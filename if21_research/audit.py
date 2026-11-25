"""Verify P3-12 source attachments, image history, and recoverable memory."""
import argparse
from collections import Counter
import json
from pathlib import Path

from if15_research.audit import pts_ledger
from if17_research.catalog import read, ref, sha
from if20_research.direct import inspect_png
from p1.io import dump

from .protocol import CONDITIONS, TARGETS, orders


def session_events(block):
    log = Path(block["path"]) / "runtime_events.jsonl"
    ids = set()
    removals = []
    for line in log.read_text(encoding="utf-8").splitlines():
        message = json.loads(line)
        data = message.get("params", {})
        if data.get("sessionId") != block["session_id"] or message.get("method") != "session.event":
            continue
        event = data["event"]
        if event["type"] == "user/message":
            ids.update(v["attachment"]["attachmentId"].removeprefix("sha256:")
                       for v in event["data"]["content"] if v.get("type") == "image")
        if "compact" in event["type"] or "reset" in event["type"]:
            removals.append(event["type"])
    return ids, removals


def audit(out):
    frames = read(out / "input/frames.json")
    locked = read(out / "protocol_lock.json")
    assert sha(Path(locked["packets"]["path"])) == locked["packets"]["sha256"]
    assert sha(Path(locked["evaluation_only"]["path"])) == locked["evaluation_only"]["sha256"]
    for path in locked["code"]:
        assert sha(Path(path["path"])) == path["sha256"], path
    source_pts = {}
    for case, book in frames.items():
        source_pts[case] = pts_ledger(book["source"])
        assert sha(book["source"]["path"]) == book["source"]["sha256"]
        assert book["original_source_audit"]["pass"]
        for frame in book["frames"]:
            for key in ("native_png", "cell_png"):
                assert inspect_png(Path(frame[key]["path"]).read_bytes()) == frame[key]["image"]
    manifests = [read(out / "matrix_manifest.json")]
    if (out / "extra_manifest.json").exists():
        manifests.append(read(out / "extra_manifest.json"))
    items = {}
    observed_sessions = set()
    sdk_sizes = Counter()
    result_attempts = Counter()
    for path in (out / "trials").glob("*/*/*/*/attempt_*/result.json"):
        result_attempts[read(path)["status"]] += 1
    for manifest in manifests:
        for key, pointer in manifest["results"].items():
            if pointer["status"] != "completed":
                continue
            result = read(pointer["path"])
            case, arm, variant, _ = key.split("/")
            book = frames[case]
            pts, ledger = source_pts[case]
            sessions = result["sessions"]
            timeline = []
            mode_counts = []
            compaction = []
            for block in sessions:
                session_id = block["session_id"]
                assert session_id not in observed_sessions
                observed_sessions.add(session_id)
                acknowledged, events = session_events(block)
                compaction.extend(events)
                earlier = 0
                last_image_time = None
                received = []
                for turn in block["turns"]:
                    assert turn["history_image_items_before_this_turn"] == earlier
                    assert turn["audit"]["sdk_user_event_verified"]
                    earlier += len(turn["attachments"])
                    for meta, path in zip(turn["attachments"], turn["native_paths"], strict=True):
                        assert meta["sha256"] in acknowledged
                        assert sha(Path(path)) == meta["sha256"]
                        sdk_sizes[f'{meta["width"]}x{meta["height"]}'] += 1
                    for stamp in turn["frame_map"]:
                        bindings = ledger[stamp["source_frame"]]
                        assert stamp["pts"] == bindings["pts"] and stamp["time_base"] == pts
                        seconds = stamp["seconds"]
                        assert abs(seconds - float(bindings["pts_time"])) <= 1e-6
                        if "previous_image_seconds" in stamp or "previous_displayed_seconds" in stamp:
                            assert stamp.get("previous_image_seconds", stamp.get("previous_displayed_seconds")) == last_image_time
                        last_image_time = seconds
                        received.append(stamp["pts"])
                    timeline.extend(received[-len(turn["frame_map"]):] if turn["frame_map"] else [])
                mode_counts.append(earlier)
            if arm in {"C5", "C6"}:
                assert len(sessions) == 1
                assert timeline == orders(case)["ascending"]
                assert len(sessions[0]["turns"]) == len(book["frames"])
                assert all(len(turn["attachments"]) == 1 for turn in sessions[0]["turns"])
                assert all(("Past model-written text" in turn["prompt"]) == (arm == "C6" and turn["turn"] > 1)
                           for turn in sessions[0]["turns"])
            elif arm in {"C7", "C8", "C15"}:
                assert all(r == 1 for r in mode_counts[:-1]) and len(mode_counts[:-1]) == len(book["frames"])
                assert len(sessions[-1]["turns"][0]["attachments"]) == 0
                current = read(result["memory"]["path"])
                requested = orders(case)[variant] if arm == "C8" else orders(case)["ascending"]
                assert [stamp["pts"] for stamp in current] == requested
                assert all(sha(stamp["native_pointer"]["path"]) == stamp["native_pointer"]["sha256"] for stamp in current)
            if arm == "C12":
                assert timeline == orders(case)[variant]
            if arm == "C13":
                assert TARGETS[case] in timeline
                assert len(timeline) == {"one": 1, "two": 2, "full": len(book["frames"])}[variant]
            items[key] = {"session_ids": [block["session_id"] for block in sessions], "image_items_by_session": mode_counts,
                          "actual_pts_order": timeline, "sdk_projection_reset_or_compaction_events": compaction}
    metadata = {"prediction_attempts": dict(result_attempts), "verified_trials": len(items), "physical_sessions": len(observed_sessions),
                "sdk_payload_sizes": dict(sdk_sizes), "items": items,
                "required_groups": sum(len(x) for x in CONDITIONS.values()) * 2,
                "notes": "SDK user messages and session identity are verified. Local DSH code derives historical messages for each request and serializes image blocks. Provider-internal vision preprocessing remains unobserved."}
    dump(out / "matrix_source_audit.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    audit(parser.parse_args().out.resolve())
