"""Behavioral/traceability checks, including an independent original-V CLI run."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from p1.baseline import run_v
from p1.core import Parameters, analyze, pair_metrics
from p1.io import ROOT, dump, read_manifest, read_video, validate
from tests.fixtures import texture


class Contracts(unittest.TestCase):
    def test_translation_direction_and_overlap(self):
        a = texture(71)
        b = cv2.warpAffine(a, np.float32([[1, 0, 8], [0, 1, -4]]), (320, 180))
        m = pair_metrics(a, b, Parameters())
        self.assertTrue(m["motion_supported"])
        self.assertAlmostEqual(m["dx"], 8, delta=.3)
        self.assertAlmostEqual(m["dy"], -4, delta=.3)
        self.assertLess(m["residual_mad"], .01)
        self.assertLess(m["overlap"], 1)

    def test_large_shift_does_not_claim_motion(self):
        a = texture(71)
        b = cv2.warpAffine(a, np.float32([[1, 0, 140], [0, 1, 0]]), (320, 180))
        m = pair_metrics(a, b, Parameters())
        self.assertFalse(m["motion_supported"])

    def test_gap_is_unknown(self):
        frames = [{"t": t, "gray": texture(71)} for t in (5., 7., 9.)]
        result = analyze(frames, {"start": 5., "end": 10.})
        self.assertTrue(any(s["evidence"]["reason"] == "sampling_gap" for s in result["segments"]))
        self.assertTrue(all(s["motion_type"] == "unknown" for s in result["segments"]))

    def test_one_frame_is_unknown(self):
        result = analyze([{"t": 7., "gray": texture(71)}], {"start": 7., "end": 8.})
        self.assertEqual(result["segments"][0]["kind"], "uncertain")

    def test_manifest_preserves_times_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for i in range(3):
                cv2.imwrite(str(tmp / f"{i}.png"), texture(71))
            times = [2.125, 2.381, 2.997]
            dump(tmp / "frames.json", {"end": 3.2, "frames": [
                {"path": f"{i}.png", "t": t, "source_frame": 41 + i, "custom_id": f"f{i}"}
                for i, t in enumerate(times)]})
            frames, source = read_manifest(tmp / "frames.json")
            result = analyze(frames, source)
            self.assertEqual([f["t"] for f in result["frames"]], times)
            self.assertEqual(result["frames"][1]["custom_id"], "f1")
            self.assertEqual(result["segments"][-1]["end"], 3.2)

    def test_invalid_timeline_rejected(self):
        for times, end in (([1., 1.], 2.), ([2., 1.], 3.), ([1., float("nan")], 3.), ([1., 2.], 2.)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                validate([{"t": t, "gray": texture(71)} for t in times], end)

    def test_bad_parameters_rejected(self):
        for p in (Parameters(coarse_fps=0), Parameters(min_overlap=1.1), Parameters(width=4), Parameters(max_frames=1)):
            with self.assertRaises(ValueError):
                p.validate()

    def test_sample_limit_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Sample limit"):
            read_video(ROOT / "artifacts/videos/hard_cut.mkv", max_frames=2)

    def test_all_artifact_coverage_and_traceability(self):
        for path in (ROOT / "artifacts/results").glob("*/intervals.json"):
            with self.subTest(path=path):
                d = json.loads(path.read_text(encoding="utf-8"))
                segs = d["segments"]
                self.assertEqual(segs[0]["start"], d["source"]["start"])
                self.assertEqual(segs[-1]["end"], d["source"]["end"])
                for a, b in zip(segs, segs[1:]):
                    self.assertEqual(a["end"], b["start"])
                observed = {f["t"] for f in d["frames"]}
                for s in segs:
                    self.assertGreater(s["end"], s["start"])
                    self.assertTrue(set(s["evidence"].get("sample_times", [])) <= observed)
                    for side in ("anchor_left", "anchor_right"):
                        if s[side] is not None:
                            self.assertIn(s[side]["t"], observed)
                self.assertEqual(len(d["pair_metrics"]), len(d["frames"]))
                self.assertTrue(all(f["source_frame"] is not None for f in d["frames"]))

    def test_frozen_files_unchanged(self):
        for name, expected in {
            "common.py": "db0e462a8331f086580a6ed846e8c2bc4732ade9a02cf7cf270e42b401536d86",
            "select_v.py": "0649bc771823884eda9e3cca65135d230f38fa7e3b926b787160276854dfae45",
        }.items():
            self.assertEqual(hashlib.sha256((ROOT / "vendor" / name).read_bytes()).hexdigest(), expected)

    def test_baseline_adapter_matches_original_cli(self):
        # Run byte-identical source in the original directory structure. This
        # independently exercises the baseline's main(), not the adapter itself.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "scripts").mkdir()
            (tmp / "config").mkdir()
            (tmp / "work/test").mkdir(parents=True)
            for name in ("common.py", "select_v.py"):
                shutil.copy2(ROOT / "vendor" / name, tmp / "scripts" / name)
            shutil.copy2(ROOT / "vendor/experiment.json", tmp / "config/experiment.json")
            frames, candidates = [], []
            for i in range(10):
                g = cv2.resize(texture(71 if i < 4 else 83), (96, 54))
                p = tmp / f"{i}.pgm"
                cv2.imwrite(str(p), g)
                t = 3.125 + i * .31
                frames.append({"t": t, "gray": g, "source_frame": i})
                candidates.append({"i": i, "t": i * .31, "source_t": t, "source_frame": i, "thumb": str(p)})
            dump(tmp / "work/test/candidates.json", {"candidate_fps": 5, "candidates": candidates})
            subprocess.run([sys.executable, str(tmp / "scripts/select_v.py"), "--id", "test"], check=True, capture_output=True)
            native = json.loads((tmp / "work/test/selections_v.json").read_text())
            adapted = run_v(frames)
            self.assertEqual(native["change_peaks_t"], adapted["change_peaks_t"])
            for k in ("3", "5", "8"):
                self.assertEqual(native["budgets"][k]["times"], adapted["budgets"][k]["times"])
                self.assertEqual(native["budgets"][k]["source_frames"], adapted["budgets"][k]["source_frames"])
                self.assertEqual(native["budgets"][k]["diff_scores"], adapted["budgets"][k]["diff_scores"])


if __name__ == "__main__":
    unittest.main()
