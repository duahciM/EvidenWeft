"""Generate inspectable outcomes, including failures; never filter failed cases."""
import json
from pathlib import Path

import numpy as np

from p1.baseline import run_v
from p1.core import Parameters, analyze
from p1.diagnostics import render
from p1.io import ROOT, command, common, dump, read_video
from tests.fixtures import generate


def overlap(s, lo, hi):
    return s["start"] <= hi and s["end"] >= lo


def evaluate(case, result):
    candidates = [s for s in result["segments"] if s["kind"] != "stable"]
    edits = [s for s in candidates if s["motion_type"] == "likely_edit"]
    expected = case["expected"]
    lo, hi = case.get("event_range", [0, 0])
    hit = [s for s in candidates if overlap(s, lo, hi)]
    if expected == "hard_cut":
        ok = any(s["motion_type"] == "likely_edit" and s["end"] - s["start"] <= .6 for s in hit)
        rule = "likely_edit interval covers 3s and width <= 0.6s"
    elif expected == "gradual":
        ok = any(s["evidence"]["gradual_candidate"] and s["start"] <= lo + .4 and s["end"] >= hi - .4 for s in hit)
        rule = "gradual candidate covers blend, endpoint tolerance 0.4s"
    elif expected == "continuous_motion":
        ok = bool(hit) and any(s["motion_type"] == "continuous_motion" for s in hit) and not edits
        rule = "continuous_motion detected and zero likely_edit intervals"
    elif expected == "uncertain":
        ok = any(s["kind"] == "uncertain" and s["motion_type"] == "unknown" for s in hit) and not edits
        rule = "parallax proxy can abstain; zero likely_edit intervals"
    elif expected == "local_event":
        ok = not candidates and bool(result["local_event_candidates"])
        rule = "no global segment split, retain >=1 local event candidate"
    elif expected == "stable":
        ok = not candidates and all(s["end"] - s["start"] <= 30 + 1e-8 for s in result["segments"])
        rule = "stable coverage chunks <=30s, no edit/uncertain split"
    elif expected == "no_edit":
        ok = not edits
        rule = "lighting-only challenge: zero likely_edit intervals"
    elif expected == "unknown":
        ok = bool(candidates) and all(s["motion_type"] == "unknown" for s in candidates)
        rule = "low-quality/missing-stable-anchor challenge: abstain"
    else:
        ok = result["source"]["start"] == 3 and len({round(m.get("gap_s", 0), 3) for m in result["pair_metrics"][1:]}) > 1
        rule = "source start 3s retained, sampled gap durations nonuniform"
    return {"pass": bool(ok), "rule": rule,
            "intervals": [{k: s[k] for k in ("start", "end", "kind", "motion_type")} | {"reason": s["evidence"]["reason"]} for s in result["segments"]],
            "local_event_count": len(result["local_event_candidates"]),
            "likely_edit_intervals": len(edits)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any challenge check fails")
    args = parser.parse_args()
    cases = generate(ROOT / "artifacts/videos")
    p = Parameters()
    rows = []
    for case in cases:
        print("RUN", case["id"], flush=True)
        frames, source = read_video(case["path"], p.coarse_fps, p.width, p.max_frames)
        result = analyze(frames, source, p)
        baseline = run_v(frames)
        baseline["input_regime"] = "same decoded source sample pool as P1, 5 fps plus final frame"
        out = ROOT / "artifacts/results" / case["id"]
        # Run the original extractor, original thumbnail bytes, original nominal
        # grid and V functions. No P1 final-frame injection or interpolation here.
        cfg = json.loads((ROOT / "vendor/experiment.json").read_text(encoding="utf-8"))
        s = cfg["sampling"]
        pool = common.extract_candidates(case["path"], out / "native_v_pool", s["candidate_fps"],
                                         s["candidate_width"], s["candidate_height"], s["thumb_width"], s["thumb_height"])
        native_frames = []
        for c in pool:
            w, h, buf = common.read_pgm(c["thumb"])
            native_frames.append({"t": c["t"], "gray": np.frombuffer(buf, np.uint8).reshape(h, w),
                                  "source_frame": None})
        native_v = run_v(native_frames)
        native_v["input_regime"] = "unmodified original extractor 0.5fps, 560x315 display, 96x54 PGM; NO extra final frame"
        native_v["time_basis"] = "original nominal_t, NOT source PTS; never used as source-time boundary evidence"
        native_v["candidate_pool"] = pool
        # For the CFR fixtures only, independently verify the original default
        # fps filter's observed source mapping (19+40*i). V's nominal labels are
        # retained; corrected source_t is added rather than overwriting them.
        if case["id"] != "pts_offset_vfr":
            raw_thumbs = command(["ffmpeg", "-v", "error", "-i", case["path"], "-vf", "scale=96:54,format=gray",
                                  "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
            mapping = []
            for c, f in zip(pool, native_frames):
                idx = 19 + 40 * c["i"]
                matched = raw_thumbs[idx * 96 * 54:(idx + 1) * 96 * 54] == f["gray"].tobytes()
                mapping.append({"candidate": c["i"], "nominal_t": c["t"], "source_frame": idx,
                                "source_t": idx / 20, "pixel_match": matched})
            if not all(x["pixel_match"] for x in mapping):
                raise AssertionError("Original V candidate mapping failed independent pixel audit")
            native_v["source_mapping_audit"] = mapping
        else:
            native_v["source_mapping_audit"] = "not resolved for VFR; original nominal labels preserved explicitly"
        dump(out / "intervals.json", result)
        dump(out / "baseline_v.json", baseline)
        dump(out / "baseline_v_native_rate.json", native_v)
        render(result, baseline, out, frames)
        evaluation = evaluate(case, result)
        if case["id"] == "pts_offset_vfr":
            # Independent full-frame decode: compare exact selected pixels and
            # PTS records, rather than validating via the same select adapter.
            raw = command(["ffmpeg", "-v", "error", "-i", case["path"], "-fps_mode", "passthrough",
                           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"])
            pixels = 320 * 180
            pixel_ok = all(raw[f["source_frame"] * pixels:(f["source_frame"] + 1) * pixels] == f["gray"].tobytes() for f in frames)
            original = json.loads(command(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                                           "-show_entries", "frame=pts_time", "-of", "json", case["path"]]))["frames"]
            pts_ok = all(f["pts_time"] == original[f["source_frame"]]["pts_time"] for f in frames)
            evaluation.update(pixel_join_verified=pixel_ok, pts_join_verified=pts_ok, audited_frames=len(frames))
            evaluation["pass"] &= pixel_ok and pts_ok
        dump(out / "evaluation.json", {"case": case, **evaluation})
        rows.append({"case": case, "evaluation": evaluation,
                     "baseline_same_pool_peaks": baseline["change_peaks_t"],
                     "baseline_native_peaks": native_v["change_peaks_t"],
                     "baseline_same_pool_budget5": baseline["budgets"]["5"]["times"],
                     "baseline_native_budget5": native_v["budgets"]["5"]["times"],
                     "result_dir": str(out)})
        print(case["id"], evaluation, flush=True)
    dump(ROOT / "artifacts/comparison.json", {"parameters": result["parameters"], "cases": rows,
                                               "note": "No shared classification accuracy metric: frozen V selects frames, P1 segments visual continuity."})
    failed = [r["case"]["id"] for r in rows if not r["evaluation"]["pass"]]
    print(f"Acceptance checks passed {len(rows) - len(failed)}/{len(rows)}; failed={failed}; NOT an accuracy estimate", flush=True)
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
