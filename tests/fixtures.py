"""Deterministic synthetic fixtures. Truth is never imported by the detector."""
from pathlib import Path
import subprocess

import cv2
import numpy as np

from p1.io import dump, common

FPS, W, H = 20, 320, 180


def texture(seed, width=W, height=H):
    rng = np.random.default_rng(seed)
    a = rng.integers(35, 220, (height, width), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (5, 5), 0)
    for _ in range(70):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        cv2.circle(a, (x, y), int(rng.integers(2, 12)), int(rng.integers(0, 255)), -1)
    return a


def write_video(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lossless grayscale clips avoid codec noise dominating the minimal tests.
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
                             "-an", "-c:v", "ffv1", str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.astype(np.uint8).tobytes())
        proc.stdin.close()
        err = proc.stderr.read()
        if proc.wait() != 0:
            raise RuntimeError(err.decode(errors="replace"))
    finally:
        if proc.poll() is None:
            proc.kill()


def generate(root):
    root = Path(root)
    a, b, pano = texture(11), texture(29), texture(37, 640)
    rows = []
    definitions = [
        ("hard_cut", 6., [3., 3.], "hard_cut", "abrupt texture A -> independent B at t=3"),
        ("dissolve", 6., [2., 4.], "gradual", "linear blend of two static textured views"),
        ("pan", 6., [2., 4.], "continuous_motion", "one plane horizontal crop, 120 pixels over 2 seconds"),
        ("door_parallax", 6., [2., 4.], "uncertain", "proxy: two independently moving depth planes, occluding door posts; not real walking"),
        ("local_object", 6., [2., 4.], "local_event", "fixed camera, small person silhouette carries a box"),
        ("ppt", 36., [16., 16.05], "stable", "fixed slide 36 seconds, one small bullet appears; coverage chunks permitted"),
        ("flash_failure", 6., [2., 2.05], "no_edit", "one-frame global light flash; deliberate false-edit challenge"),
        ("exposure_failure", 6., [2., 4.], "no_edit", "gradual global exposure ramp; linear blend ambiguity"),
        ("blur_unknown", 6., [2., 4.], "unknown", "gradual blur plateau; preserve timestamps and abstain"),
        ("edge_dissolve", 2., [0., 2.], "unknown", "whole clip dissolves; no stable endpoint dwell")]
    for name, duration, interval, expected, description in definitions:
        def frames():
            for i in range(round(duration * FPS)):
                t = i / FPS
                u = float(np.clip((t - 2) / 2, 0, 1))
                if name == "hard_cut":
                    x = a.copy() if t < 3 else b.copy()
                elif name == "dissolve":
                    x = ((1 - u) * a + u * b).astype(np.uint8)
                elif name == "pan":
                    shift = round(120 * u)
                    x = pano[:, shift:shift + W].copy()
                elif name == "door_parallax":
                    # Two planes with opposing projective image motions. Global
                    # translation should fail; no claim of photorealistic depth.
                    far = cv2.warpAffine(a, np.float32([[1 + .15 * u, 0, -20 * u], [0, 1 + .15 * u, -12 * u]]), (W, H), borderMode=cv2.BORDER_REFLECT)
                    near = cv2.warpAffine(b, np.float32([[1 + .5 * u, 0, 20 * u], [0, 1 + .5 * u, -45 * u]]), (W, H), borderMode=cv2.BORDER_REFLECT)
                    x = far.copy()
                    x[:, :150] = near[:, :150]
                    post = round(145 + 90 * u)
                    cv2.rectangle(x, (post, 0), (post + 14, H), 25, -1)
                elif name == "local_object":
                    x = a.copy()
                    pos = round(65 + 130 * u)
                    cv2.circle(x, (pos, 83), 7, 35, -1)
                    cv2.rectangle(x, (pos - 6, 90), (pos + 6, 122), 35, -1)
                    cv2.rectangle(x, (pos + 7, 100), (pos + 22, 113), 230, -1)
                elif name == "ppt":
                    x = np.full((H, W), 238, np.uint8)
                    cv2.putText(x, "P1 FIXED SLIDE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, 20, 2)
                    for line in range(4):
                        cv2.putText(x, f"- stable bullet {line + 1}", (24, 65 + 22 * line), cv2.FONT_HERSHEY_SIMPLEX, .45, 45, 1)
                    if t >= 16:
                        cv2.circle(x, (285, 150), 5, 30, -1)
                elif name == "flash_failure":
                    x = np.full_like(a, 250) if i == 40 else a.copy()
                elif name == "exposure_failure":
                    x = np.clip(a.astype(float) + 55 * u, 0, 255).astype(np.uint8)
                elif name == "blur_unknown":
                    sigma = 8 * np.sin(np.pi * u)
                    x = cv2.GaussianBlur(a, (0, 0), sigma) if sigma > .1 else a.copy()
                else:
                    u = t / duration
                    x = ((1 - u) * a + u * b).astype(np.uint8)
                # Static watermark outside the main texture, never time-varying.
                cv2.putText(x, "SYNTHETIC P1", (5, H - 7), cv2.FONT_HERSHEY_SIMPLEX, .35, 0, 1)
                yield x
        path = root / (name + ".mkv")
        write_video(path, frames())
        rows.append({"id": name, "path": str(path.resolve()), "sha256": common.sha256_file(path),
                     "duration": duration, "fps": FPS, "dimensions": [W, H],
                     "event_range": interval, "expected": expected, "description": description,
                     "provenance": "deterministic synthetic; development fixtures, not held-out accuracy test"})
    # Independent timestamp stress case: nonzero start and irregular source PTS.
    path = root / "pts_offset_vfr.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(root / "hard_cut.mkv"),
                    "-vf", "setpts=3/TB+N*0.05/TB+floor(N/3)*0.025/TB",
                    "-fps_mode", "passthrough", "-enc_time_base", "1:1000", "-c:v", "ffv1", str(path)], check=True)
    rows.append({"id": "pts_offset_vfr", "path": str(path.resolve()), "sha256": common.sha256_file(path),
                 "expected": "pts_integrity", "description": "source PTS starts at 3s, alternating 50/75ms gaps",
                 "provenance": "timestamp stress transform of hard_cut"})
    dump(root / "manifest.json", {"seed_ids": [11, 29, 37], "videos": rows})
    return rows


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "artifacts/videos")
