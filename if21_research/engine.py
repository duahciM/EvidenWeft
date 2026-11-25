"""Execute one independently scoped presentation or memory experiment."""
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter
from traceback import format_exc

from if17_research.catalog import read, ref, sha
from if18_research.observe import ACTION_NAMES, BASE, MULTI
from if18_research.video_tools import CAP
from if20_research.direct import AuditedHarness, inspect_png, prompt_audit
from if20_research.native_delivery import NativeVideoTools, source_verification
from p1.io import dump

from .protocol import BUDGET, DESCRIBE, DISTRACTORS, FINAL, GROUNDS, ADD, RECORD, ROUTE, SHARED, SYNTH, TARGETS, TASKS, orders


def as_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def frame_table(rows, previous=None):
    table = []
    for index, row in enumerate(rows, 1):
        t = row["t"]
        table.append({"image": index, "pts": row["pts"], "seconds": t, "time_base": row["time_base"],
                      "source_frame": row["source_frame"], "previous_image_seconds": previous,
                      "open_interval_between_displayed_frames": sorted([previous, t]) if previous is not None and t != previous else None})
        previous = t
    return table


class Session:
    def __init__(self, folder, turns, role):
        self.folder = folder
        self.harness = AuditedHarness(folder)
        self.turns = turns
        self.role = role

    def turn(self, prompt, images=(), expected=None, frames=None, stage=""):
        images = [Path(path) for path in images]
        expected = expected if expected is not None else [inspect_png(path.read_bytes()) for path in images]
        ordinal = len(self.turns) + 1
        receipt = self.harness.turn(prompt, images, ordinal)
        payload = self.harness.sent[-1]
        audit = prompt_audit(self.folder, payload, prompt, self.harness.session)
        answer = receipt["result"]["reply"]
        last = audit["termination"][-1]["reason"]["kind"] if audit["termination"] else None
        row = {"role": self.role, "stage": stage, "session": self.harness.session, "turn": ordinal,
               "prompt": prompt, "images": [ref(path) for path in images], "frame_map": frames or [],
               "native_paths": [str(path) for path in images], "attachments": payload, "audit": audit, "answer": answer,
               "seconds": receipt["seconds"], "native_tool_calls": receipt["result"]["tool_calls"],
               "history_image_items_before_this_turn": sum(len(item["attachments"]) for item in self.turns),
               "sdk_ack": receipt["receipt"]}
        self.turns.append(row)
        dump(self.folder / f"turn_{ordinal:02}.json", row)
        if (payload != expected or not audit["sdk_user_event_verified"] or last != "completed" or
                not audit["visible_reply"] or row["native_tool_calls"] or
                any(p != ROUTE["provider"] for p in audit["provider_context"]) or
                any(m != ROUTE["model"] for m in audit["model_context"])):
            raise RuntimeError(f"Invalid SDK turn: {self.role}, {ordinal}, {last}")
        return row


@contextmanager
def session(folder, sinks, role):
    folder.mkdir(exist_ok=False, parents=True)
    turns = []
    entry = {"role": role, "path": str(folder), "turns": turns}
    sinks.append(entry)
    live = Session(folder, turns, role)
    try:
        entry["session_id"] = live.harness.session
        entry["initialize"] = live.harness.result(live.harness.send("initialize", ROUTE | {"cwd": str(folder)}))
        yield live
    finally:
        live.harness.close()


def visual_turn(live, rows, variant="native_png", target=None, prior=None, stage="observe"):
    previous = live.turns[-1]["frame_map"][-1]["seconds"] if live.turns and live.turns[-1]["frame_map"] else None
    table = frame_table(rows, previous)
    task = "Describe every image separately." if target is None else f"Describe only the image whose original PTS is {target}."
    prompt = DESCRIBE + " " + task + "\n" + RECORD + "\nImage attachment order and source mapping: " + as_json(table)
    if prior:
        prompt += ("\nPast model-written text for context only. It may be incomplete or wrong. It is not an image or ground truth."
                   " The requested observations must come from the newly attached image. Previous records: " + as_json(prior))
    pictures = [row[variant] for row in rows]
    return live.turn(prompt, [p["path"] for p in pictures], [p["image"] for p in pictures], table, stage)


def observation(row, frame):
    answer = row.get("answer") or {}
    same = [item for item in answer.get("facts", []) if isinstance(item, dict) and
            item.get("pts") in (frame["pts"], [frame["pts"]])]
    return {"pts": frame["pts"], "seconds": frame["t"], "time_base": frame["time_base"],
            "source_frame": frame["source_frame"], "observer_session": row["session"], "observer_turn": row["turn"],
            "grounding": "revisable_model_description", "text": same,
            "unstructured_visible_text": row["audit"]["visible_reply"] if not same else None,
            "native_pointer": frame["native_png"]}


def memory_for_prompt(memory):
    return [{k: row[k] for k in ("pts", "seconds", "time_base", "source_frame", "text", "unstructured_visible_text")} for row in memory]


def isolated(folder, result, rows, chain=False):
    memory = []
    for frame in rows:
        with session(folder / f'frame_{frame["pts"]}', result["sessions"], "isolated_visual") as live:
            turn = visual_turn(live, [frame], prior=memory_for_prompt(memory) if chain else None)
        memory.append(observation(turn, frame))
        dump(folder / "memory.json", memory)
    return memory


def integrate(live, case, memory=None):
    prompt = TASKS[case] + "\n" + SYNTH
    if memory is not None:
        prompt += ("\nOnly model-written text observations are provided. Treat them as revisable and preserve only supported detail. "
                   "There are no attached images. Source-order text records:\n" + as_json(memory_for_prompt(memory)))
    return live.turn(prompt, stage="synthesis")


class MemoryTools(NativeVideoTools):
    def __init__(self, case, folder, rows):
        super().__init__(case, folder, True)
        self.memory_rows = {r["pts"]: r for r in rows}

    def dispatch(self, action, turn):
        if action["tool"] != "read_memory_png":
            return super().dispatch(action, turn)
        idx = len(self.calls) + 1
        folder = self.path / f"call_{idx:02d}"
        folder.mkdir()
        args = action.get("args") or {}
        pts = args.get("pts", []) if isinstance(args, dict) else []
        event = {"call": idx, "turn_requested": turn, "tool": "read_memory_png",
                 "why": action.get("why"), "args": args, "source_sha256": self.source["sha256"],
                 "returned_to_observer": False, "returned_on_turn": None, "seconds": 0.0}
        images = []
        frames = []
        try:
            if not isinstance(pts, list) or not 1 <= len(pts) <= CAP["image_batch"] or not action.get("why"):
                raise ValueError("Provide a bounded PTS list and reason.")
            known = [self.memory_rows[p] for p in sorted(set(pts)) if type(p) is int and p in self.memory_rows]
            aliases = [row | {"native_image": row["native_png"]["path"],
                              "native_image_sha256": row["native_png"]["sha256"]} for row in known]
            data, images, frames = self.atlas(aliases, folder)
            event.update(status="returned", reply=data | {"requested_pts": pts,
                         "not_returned_pts": [p for p in pts if type(p) is not int or p not in self.memory_rows]})
        except (ValueError, OSError, TypeError) as exc:
            event.update(status="error", reply={"error": str(exc), "images_returned": 0})
        event.update(queued_images=[ref(path) for path in images], frame_receipts=frames)
        self.calls.append(event)
        dump(folder / "tool_receipt.json", event)
        return event, images


def tool_master(folder, result, rows, memory, combo):
    case = result["case"]
    tools = MemoryTools(case, folder, rows)
    initial = BASE.replace("you have not seen any image yet", "you have not seen any image yourself yet").replace(
        "delivered tile", "delivered independent PNG")
    initial += ("\n" + MULTI + "\n6: Verify exact original PNGs named in memory. args {pts:[integer,...]}. "
                "Choose which ambiguity to verify. Text-only records below are not your own visual observations; "
                "leave any conflict or missing information explicit. Attachments arrive individually, with names in a separate receipt."
                "\nUser question: " + TASKS[case] + "\nRevisable text memory:\n" + as_json(memory_for_prompt(memory)))
    if combo:
        initial += ("\nBefore final_narration write each verified current observation to facts, with the actual source PTS "
                    "and what changed versus memory; mark unverified items and keep remaining unknowns.")
    with session(folder, result["sessions"], "combo_master" if combo else "text_with_tools") as live:
        text, files, mapping, waiting = initial, [], [], []
        answers = []
        for i in range(1, BUDGET["turns"] + 1):
            row = live.turn(text, files, [f["attachment"] for f in mapping], mapping, "tool_turn")
            tools.confirm(i, waiting, row["attachments"])
            reply = row["answer"]
            if not reply:
                raise ValueError("Expected structured master answer")
            answers.append(reply)
            actions = reply.get("actions") or []
            if not actions and reply.get("final_narration"):
                break
            if i == BUDGET["turns"]:
                break
            files, mapping, waiting = [], [], []
            feedback = []
            if isinstance(actions, list):
                for request in actions[:BUDGET["actions_per_turn"]]:
                    if not isinstance(request, dict) or len(tools.calls) >= CAP["calls"]:
                        continue
                    code = request.get("code")
                    name = (ACTION_NAMES | {6: "read_memory_png"}).get(code) if type(code) is int else None
                    if name == "read_source_range" and tools.ranges_attempted >= BUDGET["source_ranges"]:
                        feedback.append({"error": "Reread budget used; exact native pointers and directory remain available."})
                        continue
                    event, pictures = tools.dispatch(request | {"tool": name}, i)
                    offset = len(files)
                    files.extend(pictures)
                    mapping.extend(event["frame_receipts"])
                    waiting.append(event)
                    feedback.append({"call": event["call"], "tool": name, "attachment_start": offset + 1 if pictures else None,
                                     "status": event["status"], "result": event["reply"]})
            text = "Host records, images follow in tool and frame order.\n" + as_json(feedback)
            if i == BUDGET["turns"] - 1:
                text += "\nLast scheduled reply. Preserve observations, uncertainty and final_narration. Leave actions empty."
        result.update(tool_calls=tools.calls, tool_summary=tools.summary(), master_answers=answers,
                      final=answers[-1], source_verification=source_verification(tools, live.turns) if tools.delivered else None)
        result["bounded_without_final"] = not bool(answers[-1].get("final_narration"))
        if tools.delivered and not result["source_verification"]["pass"]:
            raise ValueError("Independent RGB verification failed")


def execute(folder, job, book, result, carry=None):
    case, arm, variant, rep = job
    ascending = book["frames"]
    rows = ascending
    by_pts = {r["pts"]: r for r in ascending}
    target = TARGETS[case]
    if arm in {"C8", "C12"}:
        rows = [by_pts[pts] for pts in orders(case)[variant]]
    if arm in {"C7", "C8", "C9", "C15"}:
        stage = folder / "visual"
        if arm == "C9":
            if carry is None:
                raise ValueError("C9 must have a paired real C7 memory packet")
            memory = read(carry)
            book_ref = ref(carry)
        else:
            memory = isolated(stage, result, rows, chain=arm == "C8")
            book_ref = ref(stage / "memory.json")
        result["memory"] = book_ref
        if arm in {"C9", "C15"}:
            tool_master(folder / "master", result, ascending, memory, combo=arm == "C15")
        else:
            with session(folder / "master", result["sessions"], "text_only_master") as live:
                result["final"] = integrate(live, case, memory)["answer"]
        return
    with session(folder / "session", result["sessions"], "observer") as live:
        if arm in {"C5", "C6"}:
            previous = None
            for frame in rows:
                turn = visual_turn(live, [frame], prior=memory_for_prompt([previous]) if previous and arm == "C6" else None)
                previous = observation(turn, frame)
        elif arm == "C11":
            table = [{key: row[key] for key in ("ordinal", "row", "column", "source_frame", "pts", "seconds", "time_base")}
                     for row in book["old_position_table"]]
            prompts = [TASKS[case] + "\n" + SHARED + "\nSource-time table, row-major by sheet tile:\n" + json.dumps(table),
                       FINAL + "\nUser question: " + TASKS[case]]
            if variant == "historical_A1":
                prompts = [prompt + "\n" + ADD for prompt in prompts]
            sheet = book["sheet"]
            live.turn(prompts[0], [sheet["path"]], [sheet["image"]], table, "historical_initial")
            result["final"] = live.turn(prompts[1], stage="synthesis")["answer"]
        elif arm == "C2":
            table = frame_table(rows)
            sheet = book["sheet"]
            grid = [{**stamp, "legacy_tile": raw["legacy_display_ordinal"], "xyxy": raw["sheet_cell_xyxy"]}
                    for raw, stamp in zip(rows, table, strict=True)]
            prompt = DESCRIBE + " Describe every image separately.\n" + RECORD
            prompt += "\nImage attachment order and source mapping: " + as_json(table)
            prompt += "\nThe single attachment is a sheet. Image numbers designate occupied tiles in row-major order."
            live.turn(prompt, [sheet["path"]], [sheet["image"]], grid, "observe")
        else:
            focus = target if arm in {"C0", "C4", "C12", "C13"} else None
            if arm == "C0" or (arm == "C13" and variant == "one"):
                rows = [by_pts[target]]
            elif arm == "C13" and variant == "two":
                rows = [r for r in rows if r["pts"] in (target, DISTRACTORS[case])]
            visual_turn(live, rows, variant="cell_png" if arm == "C3" else "native_png", target=focus)
            if arm == "C10":
                result["final"] = integrate(live, case)["answer"]
            elif arm == "C14":
                live.turn(GROUNDS, stage="grounds")


def run_trial(root, case, arm, variant, rep, book, carry=None):
    trial = root / "trials" / case / arm / variant / str(rep)
    for attempt in (1, 2):
        folder = trial / f"attempt_{attempt}"
        path = folder / "result.json"
        if path.exists():
            result = read(path)
        else:
            result = {"case": case, "arm": arm, "variant": variant, "replicate": rep,
                      "attempt": attempt, "route": ROUTE, "sessions": []}
            folder.mkdir(exist_ok=True, parents=True)
            started = perf_counter()
            try:
                execute(folder, (case, arm, variant, rep), book, result, carry)
                end = result["sessions"][-1]["turns"][-1]
                if not end["audit"]["visible_reply"]:
                    raise ValueError("Final visible reply missing")
                result["status"] = "completed"
            except Exception:
                result.update(status="failed", error=format_exc())
            result["seconds"] = perf_counter() - started
            dump(path, result)
        print(case, arm, variant, rep, attempt, result["status"], round(result["seconds"], 1), flush=True)
        if result["status"] == "completed":
            return ref(path), result
    return ref(path), result
