"""
Plotting utilities for the Sustained Anomaly Recognition (SAR) module.

Produces a single figure with four stacked subplots summarising a SARResult:

    1. original time series (green) with pattern-deviation points y highlighted (red)
    2. raw distance from the periodic pattern, d
    3. smoothed deviation score s with the dynamic threshold Theta~
    4. the three run stages of the merge-then-prune procedure: raw runs (R),
       gap-bridged runs (R'), and duration-pruned final runs (R'')
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sar


def _plot_runs_row(ax, runs, t, y_row, color, label):
    """Draw one row of a run-stage timeline as thick horizontal segments."""
    for a, b in runs:
        ax.hlines(y_row, t[a], t[b], color=color, linewidth=8)
    # Always add a legend handle, even if this stage has no runs.
    return Line2D([0], [0], color=color, linewidth=8, label=label)


def plot_sar_result(result: "sar.SARResult", output_path):
    """Build the 4-subplot SAR summary figure and save it to output_path (PDF)."""

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # 1. Original data (green) with pattern-deviation points (red).
    ax1.plot(result.t, result.x, color="green", linewidth=0.8, label="original data (x)")
    anomaly_mask = result.y.astype(bool)
    ax1.scatter(
        result.t[anomaly_mask],
        result.x[anomaly_mask],
        color="red",
        s=14,
        zorder=5,
        label="pattern deviation points (y=1)",
    )
    ax1.set_ylabel("value")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # 2. Raw distance from the periodic pattern, d.
    ax2.plot(result.t, result.d, color="tab:purple", linewidth=0.8, label="distance from pattern (d)")
    ax2.set_ylabel("distance")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    # 3. Smoothed deviation score s and the dynamic threshold Theta~.
    ax3.plot(result.t, result.s, color="tab:orange", linewidth=0.8, label="smoothed deviation (s)")
    ax3.axhline(result.theta, color="black", linestyle="--", linewidth=1, label="threshold (Theta)")
    ax3.set_ylabel("deviation score")
    ax3.grid(True)
    ax3.legend(loc="upper right")

    # 4. Run stages: raw (R, top) -> merged (R', middle) -> final (R'', bottom).
    handles = [
        _plot_runs_row(ax4, result.raw_runs, result.t, 2, "tab:blue", "raw runs (R)"),
        _plot_runs_row(ax4, result.merged_runs, result.t, 1, "tab:green", "merged runs (R')"),
        _plot_runs_row(ax4, result.final_runs, result.t, 0, "tab:red", "final runs (R'')"),
    ]
    ax4.set_yticks([2, 1, 0])
    ax4.set_yticklabels(["R (raw)", "R' (merged)", "R'' (final)"])
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_xlabel("time")
    ax4.set_ylabel("run stage")
    ax4.grid(True)
    ax4.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
