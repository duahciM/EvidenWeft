"""Expose P1.2 brightness evidence, including abstention reasons."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_brightness(result, out):
    rows = result["photometric_metrics"]
    keys = sorted({k for row in rows for k in row})
    with (out / "brightness.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    times = [row["t"] for row in rows]
    axes[0].plot(times, [m["raw_mad"] for m in result["pair_metrics"]], label="original raw MAD")
    axes[0].plot(times, [r.get("validation_mad") if r.get("validation_mad") is not None else np.nan for r in rows], label="brightness validation MAD")
    for key in ("gain", "offset"):
        axes[1].plot(times, [r.get(key) if r.get(key) is not None else np.nan for r in rows], label=key)
    for key in ("validation_residual_ratio", "gradient_correlation", "tile_support_fraction"):
        axes[2].plot(times, [r.get(key) if r.get(key) is not None else np.nan for r in rows], label=key)
    axes[3].step(times, [int(r["supported"]) for r in rows], where="post", label="brightness supported")
    axes[3].step(times, [int(m.get("registration_valid", False)) for m in result["pair_metrics"]], where="post", label="registration valid", alpha=.6)
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Source PTS seconds")
    fig.suptitle("P1.2 gain/offset evidence; missing values mean abstention, not zero residual")
    fig.tight_layout()
    fig.savefig(out / "brightness_diagnostics.png", dpi=140)
    plt.close(fig)
