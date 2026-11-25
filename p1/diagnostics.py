import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def render(result, baseline, out, frames):
    out.mkdir(parents=True, exist_ok=True)
    metrics = result["pair_metrics"]
    t = [m["t"] for m in metrics]
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for key in ("raw_mad", "residual_mad"):
        axes[0].plot(t, [m.get(key, np.nan) for m in metrics], label=key)
    axes[0].plot(baseline["times"], baseline["diff_series"], label="frozen V MAD (96x54)", alpha=.65)
    axes[0].axhline(result["parameters"]["activity_mad"], color="gray", ls=":", label="P1 activity")
    for key in ("phase_response", "overlap", "tile_fraction"):
        axes[1].plot(t, [m.get(key, np.nan) for m in metrics], label=key)
    for key in ("dx", "dy"):
        axes[2].plot(t, [m.get(key, np.nan) for m in metrics], label=key)
    for cid in sorted({c["candidate"] for c in result["anchor_curves"]}):
        rows = [c for c in result["anchor_curves"] if c["candidate"] == cid]
        for key in ("d_left", "d_right", "blend_error"):
            axes[3].plot([c["t"] for c in rows], [c[key] for c in rows], label=f"{cid}:{key}")
    axes[4].plot([f["t"] for f in result["frames"]], [f["sharpness"] for f in result["frames"]], label="Laplacian variance")
    axes[4].axhline(result["parameters"]["min_sharpness"], color="red", ls=":")
    colors = {"stable": "#87c99b", "transition": "#80b1e8", "uncertain": "#f3b47b"}
    for ax in axes:
        for s in result["segments"]:
            ax.axvspan(s["start"], s["end"], color=colors[s["kind"]], alpha=.13)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Original source timestamp (seconds)")
    fig.suptitle("P1 visual continuity | green=stable blue=transition orange=uncertain")
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=140)
    plt.close(fig)
    keys = [k for k in metrics[-1] if k != "tile_mad"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    curve_keys = ["candidate", "sample", "t", "d_left", "d_right", "relative_to_right", "blend_alpha", "blend_error"]
    with (out / "anchors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve_keys)
        writer.writeheader()
        writer.writerows(result["anchor_curves"])
    ids = np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int)
    canvas = Image.new("RGB", (4 * 320, 3 * 205), "white")
    draw = ImageDraw.Draw(canvas)
    for panel, i in enumerate(ids):
        x, y = (panel % 4) * 320, (panel // 4) * 205
        im = Image.fromarray(frames[i]["gray"]).resize((320, 180)).convert("RGB")
        canvas.paste(im, (x, y))
        draw.text((x + 4, y + 182), f"t={frames[i]['t']:.6f} source_frame={frames[i].get('source_frame', '?')}", fill="black")
    canvas.save(out / "contact_sheet.png")
