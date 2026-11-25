"""Run one of the reusable serial or isolated-memory workflows."""
import argparse
from pathlib import Path

from if17_research.catalog import read

from .engine import TASKS, run_trial


MODES = {
    "serial": ("C5", "same_session"),
    "serial-previous-text": ("C6", "same_session_previous_text"),
    "isolated-text": ("C7", "isolated_then_text"),
    "isolated-text-chain": ("C8", "ascending"),
    "isolated-text-with-reread": ("C9", "shared_memory_plus_tools"),
    "combined": ("C15", "complete"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path, help="Audited P3-12 frames.json")
    parser.add_argument("--out", required=True, type=Path, help="Fresh output directory")
    parser.add_argument("--case", choices=("short", "g3"), required=True)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--question", help="Question used by the master for modes with a synthesis stage")
    parser.add_argument("--memory", type=Path, help="Real isolated-text memory.json for isolated-text-with-reread")
    args = parser.parse_args()
    if args.mode == "isolated-text-with-reread" and args.memory is None:
        parser.error("--memory is required for isolated-text-with-reread")
    if args.memory and args.mode != "isolated-text-with-reread":
        parser.error("--memory is only for isolated-text-with-reread")
    if args.question:
        TASKS[args.case] = args.question
    packet = read(args.inputs)[args.case]
    if args.memory:
        memory = read(args.memory)
        native = {row["pts"]: row["native_png"] for row in packet["frames"]}
        if [row["pts"] for row in memory] != list(native) or any(
            row["native_pointer"]["sha256"] != native[row["pts"]]["sha256"] for row in memory
        ):
            parser.error("Memory source pointers do not match the bound frozen packet")
    out = args.out.resolve()
    out.mkdir(exist_ok=False, parents=True)
    arm, variant = MODES[args.mode]
    reference, result = run_trial(out, args.case, arm, variant, 1, packet, args.memory)
    print(reference["path"], flush=True)
    raise SystemExit(result["status"] != "completed")
