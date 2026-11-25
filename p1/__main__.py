import argparse
import json
from dataclasses import fields
from pathlib import Path

from .baseline import run_v
from .core import Parameters, analyze
from .diagnostics import render
from .io import dump, read_manifest, read_video


def main():
    parser = argparse.ArgumentParser(description="Offline dual-anchor + motion-compensation P1")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--frames", type=Path, help="JSON manifest with explicit source t and end")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--params", type=Path, help="JSON parameter overrides")
    parser.add_argument("--coarse-fps", type=float)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    overrides = json.loads(args.params.read_text(encoding="utf-8")) if args.params else {}
    if set(overrides) - {f.name for f in fields(Parameters)}:
        parser.error("Unknown parameter name")
    for key in ("coarse_fps", "max_frames"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    p = Parameters(**overrides)
    p.validate()
    if args.video:
        frames, source = read_video(args.video, p.coarse_fps, p.width, p.max_frames)
    else:
        frames, source = read_manifest(args.frames, p.width)
    result = analyze(frames, source, p)
    baseline = run_v(frames)
    baseline["input_regime"] = "same P1 sampled source frames (controlled pool); native V uses 0.5 fps"
    dump(args.out / "intervals.json", result)
    dump(args.out / "baseline_v.json", baseline)
    render(result, baseline, args.out, frames)
    print(json.dumps({"out": str(args.out.resolve()), "segments": len(result["segments"]),
                      "local_event_candidates": len(result["local_event_candidates"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
