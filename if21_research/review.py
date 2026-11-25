"""Anonymous, independent source-grounded reviews with pixel tie-breaking."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from random import Random
from traceback import format_exc

from if17_research.catalog import read, ref
from p1.io import dump

from .engine import Session, as_json
from .protocol import REVIEW, ROUTE, SEED, TARGETS


LABELS = {"correct", "wrong", "omitted", "uncertain"}
REVIEW_ROUTE = ROUTE | {"reasoningEffort": "off", "maxTokens": 4096}


@contextmanager
def reviewer_session(folder, snapshots):
    folder.mkdir(parents=True, exist_ok=False)
    turns = []
    entry = {"role": "independent_review", "path": str(folder), "route": REVIEW_ROUTE, "turns": turns}
    snapshots.append(entry)
    live = Session(folder, turns, "independent_review")
    try:
        entry["session_id"] = live.harness.session
        entry["initialize"] = live.harness.result(live.harness.send("initialize", REVIEW_ROUTE | {"cwd": str(folder)}))
        yield live
    finally:
        live.harness.close()


def opaque(*parts):
    return sha256("/".join(map(str, parts)).encode()).hexdigest()[:13]


def linked_claims(turn, pts):
    answer = turn.get("answer") or {}
    found = []
    for fact in answer.get("facts", []) or []:
        if not isinstance(fact, dict):
            continue
        bound = fact.get("pts")
        if bound == pts or isinstance(bound, list) and pts in bound:
            found.append(fact)
    return found


def extract(results):
    rows, finals, name_map = {}, {}, {}
    for name, path in results.items():
        if path["status"] != "completed":
            continue
        result = read(path["path"])
        tid = opaque(name)
        name_map[tid] = name
        case, arm = result["case"], result["arm"]
        for block in result["sessions"]:
            exposed = set()
            for turn in block["turns"]:
                current = {row["pts"] for row in turn["frame_map"]}
                exposed.update(current)
                if turn["stage"] == "grounds":
                    options = exposed
                elif turn["stage"] == "tool_turn":
                    options = current
                elif arm in {"C0", "C4", "C12", "C13"}:
                    options = {TARGETS[case]}
                elif turn["stage"] in {"observe", "historical_initial"}:
                    options = current
                else:
                    continue
                for pts in sorted(options):
                    claims = linked_claims(turn, pts)
                    fallback = turn["audit"]["visible_reply"][:2500] if not turn["answer"] else None
                    row_id = opaque(name, block["session_id"], turn["turn"], pts)
                    rows[row_id] = {"trial": tid, "case": case, "turn": turn["turn"], "pts": pts,
                                    "layer": turn["stage"], "claims": claims, "visible_text_if_unstructured": fallback,
                                    "was_delivered": pts in exposed, "actual_session": block["session_id"],
                                    "exposed_pts": sorted(exposed), "source_binding": path}
        final = result.get("final")
        if isinstance(final, dict):
            finals[tid] = {"case": case, "text_memory_only": arm in {"C7", "C8"},
                           "visible_pts_seen_by_master": sorted({frame["pts"] for turn in result["sessions"][-1]["turns"] for frame in turn["frame_map"]}),
                           "reply": {k: final.get(k) for k in ("facts", "hypotheses", "unresolved", "final_narration", "stop")},
                           "source_binding": path}
    return rows, finals, name_map


def packs(out, rows, finals, mode):
    root = out / "reviews" / mode
    root.mkdir(parents=True, exist_ok=True)
    already = set()
    reviewed_finals = set()
    width, finals_width = (6, 1) if mode == "tie_pixel" else (12, 2)
    known = sorted(root.glob("packet_*.json"))
    for path in known:
        old = read(path)
        already.update(old["rows"])
        reviewed_finals.update(old["finals"])
    for case in ("short", "g3"):
        available = [key for key in rows if rows[key]["case"] == case and key not in already]
        available_finals = [key for key in finals if finals[key]["case"] == case and key not in reviewed_finals]
        Random(SEED + len(known)).shuffle(available)
        for cursor in range(max((len(available) + width - 1) // width, (len(available_finals) + finals_width - 1) // finals_width)):
            path = root / f"packet_{len(known) + 1:03d}.json"
            chosen = {key: rows[key] for key in available[cursor * width : (cursor + 1) * width]}
            chosen_finals = {key: finals[key] for key in available_finals[cursor * finals_width : (cursor + 1) * finals_width]}
            dump(path, {"case": case, "rows": chosen, "finals": chosen_finals})
            known.append(path)
    return known


def review(out, packet, judge):
    document = read(packet)
    case = document["case"]
    rows = document["rows"]
    finals = document["finals"]
    frames = read(out / "input/frames.json")[case]["frames"]
    key = read(out / "evaluation_only/reference.json")["per_frame"][case]
    selected = {row["pts"] for row in rows.values()}
    if finals:
        selected.update(frame["pts"] for frame in frames)
    frames = [f for f in frames if f["pts"] in selected]
    show_originals = packet.parent.name == "tie_pixel"
    order = list(rows)
    if judge == 2:
        order.reverse()
    score_rows = [{"row": rid, **{k: row[k] for k in ("pts", "claims", "visible_text_if_unstructured", "was_delivered", "exposed_pts")}}
                  for rid in order for row in [rows[rid]]]
    score_finals = [{"id": rid, **{k: row[k] for k in ("text_memory_only", "visible_pts_seen_by_master", "reply")}}
                    for rid, row in finals.items()]
    prompt = REVIEW
    prompt += ("\nFinal right_or_gap_correct for the short clip requires explicit separated gripper and ball at original PTS 207720 or 213720; "
               "for G3 requires right-visible, sampled-not-visible, left-visible, without a conflicting side or invented observed path. "
               "Count final directly supported distinct visible state statements, explicit unknowns and unsupported statements separately."
               "\nSealed source reference keyed by exact PTS. " + ("Original PNGs attached in listed order: " if show_originals else "This audit uses the sealed pixel reference and the original delivery log; no images are attached here: "))
    prompt += as_json([{"image": i if show_originals else None, "pts": f["pts"], "seconds": f["t"], "expected_source_relation": key[str(f["pts"])]}
                       for i, f in enumerate(frames, 1)])
    prompt += "\nUse terse reasons of at most ten words; give each requested row only once. Rows to review (no method labels): " + as_json(score_rows)
    prompt += "\nFinal narrations to review: " + as_json(score_finals)
    for attempt in (1, 2):
        folder = packet.parent / packet.stem / f"judge_{judge}" / f"attempt_{attempt}"
        path = folder / "result.json"
        if path.exists():
            result = read(path)
        else:
            snapshots = []
            try:
                with reviewer_session(folder / "session", snapshots) as live:
                    image = [f["native_png"] for f in frames] if show_originals else []
                    turn = live.turn(prompt, [p["path"] for p in image], [p["image"] for p in image],
                                     [{"pts": f["pts"], "seconds": f["t"]} for f in frames] if show_originals else [], "adjudication")
                answer = turn["answer"] or {}
                observed = {v.get("row"): v for v in answer.get("facts", []) if isinstance(v, dict)}
                narrative = {v.get("id"): v for v in answer.get("finals", []) if isinstance(v, dict)}
                ok = (set(observed) == set(rows) and set(narrative) == set(finals)
                      and all(v.get("label") in LABELS for v in observed.values()))
                result = {"status": "completed" if ok else "failed", "rows": observed, "finals": narrative,
                          "missing": sorted(set(rows) - set(observed)), "missing_finals": sorted(set(finals) - set(narrative)),
                          "sessions": snapshots, "judgment": judge, "packet": ref(packet), "native_source_images_shown": show_originals}
            except Exception:
                result = {"status": "failed", "error": format_exc(), "sessions": snapshots,
                          "judgment": judge, "packet": ref(packet)}
            dump(path, result)
        if result["status"] == "completed":
            print(packet.stem, judge, "completed", len(rows), len(finals), flush=True)
            return path, result
    print(packet.stem, judge, "failed", flush=True)
    return path, result


def aggregate(out, workers, extra, resolve):
    merged = read(out / "matrix_manifest.json")["results"]
    if extra and (out / "extra_manifest.json").exists():
        merged = merged | read(out / "extra_manifest.json")["results"]
    rows, finals, names = extract(merged)
    # Rows without a bound statement can be graded by both readers without a visual assumption.
    active_rows = {k: v for k, v in rows.items() if v["claims"] or v["visible_text_if_unstructured"]}
    packets = packs(out, active_rows, finals, "review_text")
    values = defaultdict(dict)
    nvalues = defaultdict(dict)
    refs = []

    def collect(received):
        path, result = received
        refs.append(ref(path) | {"status": result["status"]})
        for key, row in result.get("rows", {}).items():
            if key in rows and row.get("label") in LABELS:
                values[key][result["judgment"]] = row
        for key, row in result.get("finals", {}).items():
            if key in finals and type(row.get("right_or_gap_correct")) is bool:
                nvalues[key][result["judgment"]] = row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(review, out, packet, judge) for packet in packets for judge in (1, 2)]
        for future in as_completed(pending):
            collect(future.result())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        for judge in (1, 2):
            missing = {key: row for key, row in active_rows.items() if judge not in values[key]}
            missing_final = {key: row for key, row in finals.items() if judge not in nvalues[key]}
            if missing or missing_final:
                pending.extend(pool.submit(review, out, packet, judge)
                               for packet in packs(out, missing, missing_final, f"text_recovery_{judge}"))
        for future in as_completed(pending):
            collect(future.result())
    gra = {key: row for key, row in active_rows.items() if len(values[key]) == 2 and
           values[key][1]["label"] != values[key][2]["label"]}
    gfa = {key: row for key, row in finals.items() if len(nvalues[key]) == 2 and
           any(nvalues[key][1].get(field) != nvalues[key][2].get(field)
               for field in ("right_or_gap_correct", "unsupported", "time_or_consistency_error"))}
    if resolve and (gra or gfa):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(review, out, packet, 3) for packet in packs(out, gra, gfa, "tie_pixel")]
            for future in as_completed(pending):
                collect(future.result())
    rated = {}
    for key, row in rows.items():
        ratings = values[key]
        if not row["claims"] and not row["visible_text_if_unstructured"]:
            label = "omitted"
            authority = "exact absence of bound text"
        elif len(ratings) == 2 and ratings[1]["label"] == ratings[2]["label"]:
            label, authority = ratings[1]["label"], "two independent reviews"
        elif 3 in ratings:
            label, authority = ratings[3]["label"], "third review with original pixels"
        else:
            label, authority = None, "awaiting matching double review or adjudication"
        rated[key] = row | {"label": label, "review": ratings, "decision_source": authority}
    narratives = {key: row | {"review": nvalues[key], "adjudicated": nvalues[key].get(3) or (
                    nvalues[key].get(1) if key not in gfa and len(nvalues[key]) == 2 else None)}
                  for key, row in finals.items()}
    manual_path = out / "evaluation_only" / "source_adjudications.json"
    if manual_path.exists():
        manual = read(manual_path)
        for key, source in manual["rows"].items():
            if key in rated:
                assert source["label"] in LABELS and rated[key]["pts"] == source["pts"]
                rated[key].update(label=source["label"], decision_source="manual exact-source adjudication",
                                  manual_ruling=source)
        for key, source in manual["narratives"].items():
            if key in narratives:
                old = narratives[key]["adjudicated"] or {}
                narratives[key].update(adjudicated=old | source, manual_ruling=source)
    package = {"rows": rated, "narratives": narratives, "trial_ids": names, "review_artifacts": refs,
               "active_rows": len(active_rows), "narrative_count": len(finals), "visual_disagreements": len(gra),
               "final_disagreements": len(gfa), "label_counts": dict(Counter(row["label"] or "unresolved_review" for row in rated.values()))}
    dump(out / "adjudication.json", package)
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--resolve", action="store_true")
    args = parser.parse_args()
    data = aggregate(args.out.resolve(), args.workers, args.extra, args.resolve)
    print("Ratings", data["label_counts"], "narratives", data["narrative_count"], flush=True)
