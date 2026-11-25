"""Materialize the scored matrix, repeated conflicts and evidence-use summaries."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

from if17_research.catalog import read
from p1.io import dump

from .protocol import TARGETS


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_extra(out):
    adjudicated = read(out / "adjudication.json")
    seen = defaultdict(dict)
    for row in adjudicated["rows"].values():
        if row["layer"] != "observe":
            continue
        name = adjudicated["trial_ids"][row["trial"]]
        case, arm, variant, rep = name.split("/")
        if int(rep) in (1, 2):
            seen[(case, arm, variant, row["pts"])][rep] = row["label"]
    finals = defaultdict(dict)
    for tid, row in adjudicated["narratives"].items():
        rating = row["adjudicated"]
        if rating:
            case, arm, variant, rep = adjudicated["trial_ids"][tid].split("/")
            if int(rep) in (1, 2):
                if rating["right_or_gap_correct"]:
                    label = "correct"
                elif rating.get("time_or_consistency_error", 0) or rating.get("unsupported", 0):
                    label = "wrong"
                else:
                    label = "omitted_or_unknown"
                finals[(case, arm, variant, "final")][rep] = label
    critical = {"short": {189720, 207720, 213720, 219720}, "g3": {2640, 2860, 3000, 3240}}
    repeats = defaultdict(list)
    for (case, arm, variant, pts), labels in (seen | finals).items():
        if pts in critical[case] | {"final"} and set(labels.values()) == {"correct", "wrong"}:
            repeats[(case, arm, variant, 3)].append({"pts_or_final": pts, "base_labels": labels})
    for case, arm, variant, n in list(repeats):
        if arm == "C9":
            repeats[(case, "C7", "isolated_then_text", n)].append({"companion_to": "C9"})
    path = out / "additional_repeats.json"
    previous = read(path) if path.exists() else {"jobs": [], "reasons": {}}
    job_set = sorted(set(map(tuple, previous["jobs"])) | set(repeats))
    reasons = previous["reasons"]
    for key, notes in repeats.items():
        name = "/".join(map(str, key))
        reasons[name] = reasons.get(name, []) + notes
    dump(path, {"jobs": job_set, "reasons": reasons})


def visual_rows(out, judgment):
    table = []
    for key, row in sorted(judgment["rows"].items(), key=lambda item: (judgment["trial_ids"][item[1]["trial"]], item[1]["turn"], item[1]["pts"])):
        case, arm, variant, trial = judgment["trial_ids"][row["trial"]].split("/")
        evaluations = row["review"]
        lead = evaluations.get("3") or evaluations.get("1") or {}
        table.append({"row": key, "case": case, "arm": arm, "variant": variant, "replicate": trial,
                      "layer": row["layer"], "turn": row["turn"], "pts": row["pts"], "delivered": row["was_delivered"],
                      "label": row["label"], "basis": row["decision_source"], "reason": lead.get("reason"),
                      "borrowed_pts_proposed": lead.get("borrowed_pts"), "source": row["source_binding"]["path"]})
    write_csv(out / "frame_error_table.csv", table, list(table[0]) if table else ["row"])
    return table


def cited_pts(fact):
    value = fact.get("pts")
    if type(value) is int:
        return [value]
    return [p for p in value if type(p) is int] if isinstance(value, list) else []


def trials(out, manifest, judgment, rows):
    notes = defaultdict(list)
    for row in rows:
        notes[(row["case"], row["arm"], row["variant"], row["replicate"])].append(row)
    by_trial = {judgment["trial_ids"][key]: val["adjudicated"] for key, val in judgment["narratives"].items()}
    items = {}
    for name, pointer in sorted(manifest.items()):
        case, arm, variant, rep = name.split("/")
        row = {"status": pointer["status"]}
        if pointer["status"] != "completed":
            items[name] = row
            continue
        trial = read(pointer["path"])
        turns = [turn for session in trial["sessions"] for turn in session["turns"]]
        attached = [a for turn in turns for a in turn["attachments"]]
        usage = Counter()
        for turn in turns:
            usage.update({k: v for k, v in turn["audit"]["usage"].items()
                          if isinstance(v, int) and k in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens")})
        layers = defaultdict(Counter)
        target = Counter()
        for visual in notes[(case, arm, variant, rep)]:
            layers[visual["layer"]][visual["label"] or "unresolved_review"] += 1
            if visual["pts"] == TARGETS[case]:
                target[visual["layer"] + "/" + (visual["label"] or "unresolved_review")] += 1
        invalid = [(turn["turn"], point) for turn in turns for fact in (turn.get("answer") or {}).get("facts", [])
                   if isinstance(fact, dict) for point in cited_pts(fact) if point not in turn_pts(trial, turn)]
        row.update(layers=layers, target=target, final=by_trial.get(name), tool=trial.get("tool_summary"),
                   visual_memory=trial.get("memory"), technical_attempts=trial["attempt"], session_count=len(trial["sessions"]),
                   turns=len(turns), image_items=len(attached), unique_attachment_hashes=len({a["sha256"] for a in attached}),
                   observed_images_within_sdk=all(turn["audit"]["sdk_user_event_verified"] for turn in turns),
                   nonempty_reasoning_turns=sum(turn["audit"]["reasoning_nonempty_blocks"] > 0 for turn in turns),
                   usage=dict(usage), seconds=round(trial["seconds"], 3), binding_outside_current_observation=invalid,
                   result=pointer)
        items[name] = row
    return items


def turn_pts(trial, turn):
    scope = set()
    for segment in trial["sessions"]:
        for other in segment["turns"]:
            if other["session"] == turn["session"] and other["turn"] <= turn["turn"]:
                scope.update(reference["pts"] for reference in other["frame_map"])
    if not scope and trial.get("memory"):
        scope.update(item["pts"] for item in read(trial["memory"]["path"]))
    return scope


def summarize(items):
    combined = {}
    prefixes = sorted({name.rsplit("/", 1)[0] for name in items})
    for prefix in prefixes:
        names = [key for key in items if key.startswith(prefix + "/")]
        done = [items[k] for k in names if items[k]["status"] == "completed"]
        visual = defaultdict(Counter)
        target = Counter()
        costs = Counter()
        final = Counter()
        tools = Counter()
        for row in done:
            for layer, counts in row["layers"].items():
                visual[layer].update(counts)
            target.update(row["target"])
            costs.update(row["usage"])
            costs.update({k: row[k] for k in ("session_count", "turns", "image_items", "nonempty_reasoning_turns", "seconds")})
            if row["final"]:
                final.update({"scored": 1, "qualified": int(row["final"]["right_or_gap_correct"])})
                final.update({field: val for field, val in row["final"].items()
                              if field in ("visible_observations", "unsupported", "time_or_consistency_error", "unknowns") and type(val) is int})
            if row["tool"]:
                tools.update({k: row["tool"][k] for k in ("calls", "distinct_read_frames", "source_range_attempts")})
        combined[prefix] = {"complete": len(done), "attempted": len(names), "layers": visual, "target": target,
                            "final": dict(final), "tools": dict(tools), "cost": dict(costs)}
    return combined


def memory_ablation(items):
    changes = {}
    for key, row in items.items():
        case, arm, variant, rep = key.split("/")
        if arm not in {"C9", "C15"} or row["status"] != "completed":
            continue
        result = read(row["result"]["path"])
        c7 = items.get(f"{case}/C7/isolated_then_text/{rep}")
        if arm == "C9" and c7 and c7["status"] == "completed":
            assert c7["visual_memory"]["sha256"] == row["visual_memory"]["sha256"]
        claims = result.get("master_answers", [])
        tool = result.get("tool_summary", {})
        changes[key] = {"source_memory": row["visual_memory"], "paired_c7_final": c7.get("final") if arm == "C9" and c7 else None,
                        "scored_final": row["final"], "first_pre_image_output": claims[0] if claims else None,
                        "final": result.get("final"), "pts_actually_received": tool.get("delivered_pts", []),
                        "new_verified_source_pts": sorted(set(tool.get("delivered_pts", [])) -
                            {item["pts"] for item in read(row["visual_memory"]["path"])}),
                        "request_reasons": [{"tool": c["tool"], "why": c.get("why"), "status": c["status"],
                            "delivered_pts": [f["pts"] for f in c["frame_receipts"]] if c["returned_to_observer"] else []}
                            for c in result.get("tool_calls", [])]}
    return changes


def run_stats(out):
    manifest = read(out / "matrix_manifest.json")["results"]
    if (out / "extra_manifest.json").exists():
        manifest |= read(out / "extra_manifest.json")["results"]
    judgment = read(out / "adjudication.json")
    graded = visual_rows(out, judgment)
    items = trials(out, manifest, judgment, graded)
    summary = summarize(items)
    dump(out / "condition_summary.json", {"conditions": summary, "trials": items})
    dump(out / "memory_deltas.json", memory_ablation(items))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra-plan", action="store_true")
    args = parser.parse_args()
    if args.extra_plan:
        classify_extra(args.out.resolve())
    else:
        run_stats(args.out.resolve())
