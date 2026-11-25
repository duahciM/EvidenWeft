"""Separate P1.2 entry. Original P1 core, parameters, and default CLI unchanged."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from .baseline import run_v
from .core import Parameters, analyze
from .diagnostics import render
from .io import dump, read_manifest, read_video
from .photometric import BrightnessParameters, brightness_metrics
from .photometric_diagnostics import render_brightness


def analyze_p12(frames, source, p=Parameters(), brightness=BrightnessParameters(), original=None):
    p.validate()
    brightness.validate()
    result = deepcopy(original if original is not None else analyze(frames, source, p))
    if result["parameters"] != asdict(p):
        raise ValueError("Base parameters differ from P1 result")
    result["schema"] = "p1.visual_continuity.v1.2"
    result["algorithm"] = "frozen_P1_plus_conservative_brightness_veto"
    result["brightness_parameters"] = asdict(brightness)
    result["interpretation"] = "Brightness evidence only downgrades edit claims to unknown; not proof of camera continuity or physical exposure cause."
    photo = [{"sample": 0, "t": frames[0]["t"], "supported": False, "reason": "no_previous_frame"}]
    for i in range(1, len(frames)):
        evidence = brightness_metrics(frames[i - 1]["gray"], frames[i]["gray"], result["pair_metrics"][i], p, brightness)
        evidence.update(sample=i, from_t=frames[i - 1]["t"], t=frames[i]["t"])
        photo.append(evidence)
    result["photometric_metrics"] = photo
    for s in result["segments"]:
        if s["kind"] == "stable":
            continue
        indices = s["evidence"].get("metric_indices", [])
        changed = [i for i in indices if result["pair_metrics"][i]["raw_mad"] >= p.activity_mad]
        count = sum(photo[i]["supported"] for i in changed)
        fraction = count / len(changed) if changed else 0.
        q = s["quality"]
        s["evidence"]["brightness_model"] = {"metric_indices": changed, "supported_count": count,
                                              "support_fraction": fraction, "source": "photometric_metrics"}
        safe = not q.get("has_sampling_gap", False) and not q.get("has_low_sharpness", False)
        if safe and changed and fraction >= brightness.min_group_support_fraction and s["motion_type"] != "continuous_motion":
            s["evidence"]["p1_original_decision"] = {"kind": s["kind"], "motion_type": s["motion_type"], "reason": s["evidence"]["reason"]}
            s.update(kind="uncertain", motion_type="unknown", photometric_type="global_gain_offset_supported")
            s["evidence"]["reason"] = "brightness_model_explains_change_without_proving_motion"
    return result


def main():
    parser = argparse.ArgumentParser(description="P1.2 conservative brightness-model extension")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path)
    group.add_argument("--frames", type=Path)
    parser.add_argument("--params", type=Path, required=True, help="frozen original P1 parameters")
    parser.add_argument("--brightness-params", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p = Parameters(**json.loads(args.params.read_text(encoding="utf-8")))
    bp = BrightnessParameters(**json.loads(args.brightness_params.read_text(encoding="utf-8"))) if args.brightness_params else BrightnessParameters()
    p.validate()
    bp.validate()
    frames, source = read_video(args.video, p.coarse_fps, p.width, p.max_frames) if args.video else read_manifest(args.frames, p.width)
    result = analyze_p12(frames, source, p, bp)
    baseline = run_v(frames)
    dump(args.out / "intervals.json", result)
    dump(args.out / "baseline_v.json", baseline)
    render(result, baseline, args.out, frames)
    render_brightness(result, args.out)
    print(json.dumps({"out": str(args.out.resolve()), "segments": len(result["segments"])}))


if __name__ == "__main__":
    main()
