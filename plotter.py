"""
Plotting utilities for the Sustained Anomaly Recognition (SAR) module.

All text (including math variables) is rendered with LaTeX, regardless of
main_config.TARGET.

plot_sar_result() produces the SAR summary figure for a SARResult. Its
layout depends on main_config.TARGET:

  - TARGET == 'view' (default): four stacked subplots —
      1. original time series (green) with pattern-deviation points y highlighted (red)
      2. raw distance from the periodic pattern, d
      3. smoothed deviation score s with the dynamic threshold Theta~
      4. the three run stages of the merge-then-prune procedure: raw runs (R),
         gap-bridged runs (R'), and duration-pruned final runs (R'')
  - TARGET == 'paper': two stacked axes, sized for an IEEE double-column
    figure —
      1. original time series (green) with pattern-deviation points y highlighted (red)
      2. a thin strip showing only the final identified anomaly regions (R'')

plot_pattern_sections() produces a separate figure overlaying every
period-length section of the input series with the periodic pattern (phi).
With TARGET == 'paper' its content is unchanged; it is only resized to fit
an IEEE double-column figure.

When main_config.SHOW_PLOT is True, both functions leave their figure open
instead of showing it immediately; call show_all() once after creating all
figures so they appear together in separate windows at the same time.
"""

import matplotlib
import numpy as np

import main_config as cfg

if not cfg.SHOW_PLOT:
    matplotlib.use("Agg")  # non-interactive backend, needed for saving without a display

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sar

# Render all plot text (including math variables) with LaTeX.
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

if cfg.TARGET == "paper":
    plt.rcParams["font.size"] = 11

# Approximate double-column (\textwidth) width of an IEEE conference paper.
_IEEE_DOUBLE_COLUMN_WIDTH_CM = 18.0

# Shared y-axis label for plots of the raw metric value.
_VALUE_LABEL = r"$n_\text{packets}$"


def _size_for_paper(fig, height_cm):
    """Resize fig to the IEEE double-column width, at the given height (cm).

    The full figure is first shrunk by 30% (preserving aspect ratio), then
    the height alone is grown back by 30% twice on top of that.
    """
    width_cm = _IEEE_DOUBLE_COLUMN_WIDTH_CM * 0.75
    height_cm = height_cm * 0.7 * 1.3 * 1.3
    fig.set_size_inches(width_cm / 2.54, height_cm / 2.54)


def _choose_time_unit(total_seconds):
    """Pick a display unit (seconds, minutes, or hours) for a duration."""
    if total_seconds < 120:
        return 1.0, "s"
    if total_seconds < 2 * 3600:
        return 60.0, "min"
    return 3600.0, "h"


def _plot_runs_row(ax, runs, t, y_row, color, label):
    """Draw one row of a run-stage timeline as thick horizontal segments."""
    for a, b in runs:
        ax.hlines(y_row, t[a], t[b], color=color, linewidth=8)
    # Always add a legend handle, even if this stage has no runs.
    return Line2D([0], [0], color=color, linewidth=8, label=label)


def _add_cwde_overlay(ax, result, cwde_mask, size):
    """Scatter the C-WDE detector-ensemble anomaly points ($y_i = 1$)."""
    ax.scatter(
        result.t[cwde_mask],
        result.x[cwde_mask],
        color="black",
        marker="x",
        s=size,
        zorder=6,
        label=r"C-WDE anomaly signals ($y_i = 1$)",
    )


def plot_sar_result(result: "sar.SARResult", output_path, cwde_mask=None):
    """Build the SAR summary figure (layout depends on main_config.TARGET).

    cwde_mask, if given, is a boolean array (same length as result.t)
    marking points flagged by the C-WDE detector ensemble; these are
    overlaid as an extra scatter series on the top axis.

    Depending on main_config.SHOW_PLOT, either leaves it open for show_all()
    to display, or saves it to output_path (PDF).
    """
    if cfg.TARGET == "paper":
        _plot_sar_result_paper(result, output_path, cwde_mask)
    else:
        _plot_sar_result_view(result, output_path, cwde_mask)


def _plot_sar_result_view(result: "sar.SARResult", output_path, cwde_mask=None):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # 1. Original data (green) with pattern-deviation points (red) and,
    #    optionally, C-WDE anomaly signals (black x).
    ax1.plot(result.t, result.x, color="green", linewidth=1.4, label=r"Original data ($\mathbf{x}$)")
    anomaly_mask = result.y.astype(bool)
    ax1.scatter(
        result.t[anomaly_mask],
        result.x[anomaly_mask],
        color="red",
        s=14,
        zorder=5,
        label=r"SAR pattern deviation points ($\tilde{y}_i = 1$)",
    )
    if cwde_mask is not None:
        _add_cwde_overlay(ax1, result, cwde_mask, size=24)
    ax1.set_ylabel(_VALUE_LABEL)
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # 2. Raw distance from the periodic pattern, d.
    ax2.plot(result.t, result.d, color="tab:purple", linewidth=0.8, label=r"Distance from pattern ($d$)")
    ax2.set_ylabel("Distance")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    # 3. Smoothed deviation score s and the dynamic threshold Theta~.
    ax3.plot(result.t, result.s, color="tab:orange", linewidth=0.8, label=r"Smoothed deviation ($s$)")
    ax3.axhline(result.theta, color="black", linestyle="--", linewidth=1, label=r"Threshold ($\Theta$)")
    ax3.set_ylabel("Deviation score")
    ax3.grid(True)
    ax3.legend(loc="upper right")

    # 4. Run stages: raw (R, top) -> merged (R', middle) -> final (R'', bottom).
    handles = [
        _plot_runs_row(ax4, result.raw_runs, result.t, 2, "tab:blue", r"Raw runs ($R$)"),
        _plot_runs_row(ax4, result.merged_runs, result.t, 1, "tab:green", r"Merged runs ($R'$)"),
        _plot_runs_row(ax4, result.final_runs, result.t, 0, "tab:red", r"Final runs ($R''$)"),
    ]
    ax4.set_yticks([2, 1, 0])
    ax4.set_yticklabels([r"$R$ (raw)", r"$R'$ (merged)", r"$R''$ (final)"])
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Run stage")
    ax4.grid(True)
    ax4.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    if not cfg.SHOW_PLOT:
        fig.savefig(output_path)
        plt.close(fig)


def _plot_sar_result_paper(result: "sar.SARResult", output_path, cwde_mask=None):
    fig, (ax_main, ax_regions) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [8, 1]})

    # 1. Original data (green) with pattern-deviation points (red) and,
    #    optionally, C-WDE anomaly signals (black x).
    ax_main.plot(result.t, result.x, color="green", linewidth=1.0, label=r"Original data ($\mathbf{x}$)")
    anomaly_mask = result.y.astype(bool)
    ax_main.scatter(
        result.t[anomaly_mask],
        result.x[anomaly_mask],
        color="red",
        s=8,
        zorder=5,
        label=r"SAR pattern deviation points ($\tilde{y}_i = 1$)",
    )
    if cwde_mask is not None:
        _add_cwde_overlay(ax_main, result, cwde_mask, size=14)
    ax_main.set_ylabel(_VALUE_LABEL)
    ax_main.grid(True)
    ax_main.legend(loc="upper left")
    ax_main.tick_params(labelbottom=False)

    # 2. Thin strip: only the final identified anomaly regions (R'').
    handle = _plot_runs_row(ax_regions, result.final_runs, result.t, 0, "tab:red", "SAR anomaly region candidates")
    ax_regions.set_yticks([])
    ax_regions.set_ylim(-0.5, 0.5)
    ax_regions.grid(True, axis="x")
    ax_regions.legend(handles=[handle], loc="center left")
    plt.setp(ax_regions.get_xticklabels(), rotation=30, ha="right")

    _size_for_paper(fig, height_cm=7)
    fig.tight_layout()
    if not cfg.SHOW_PLOT:
        fig.savefig(output_path)
        plt.close(fig)


def plot_pattern_sections(result: "sar.SARResult", output_path):
    """Plot every period-length section of the input series (thin lines)
    overlaid with the periodic pattern phi (thick line).

    Depending on main_config.SHOW_PLOT, either leaves it open for show_all()
    to display, or saves it to output_path (PDF). The x-axis is clipped
    exactly to the span of the curves (0 to P-1), and the y-axis is scaled
    to 3x the maximum of phi, so sections with much larger values may
    extend beyond the visible area. With TARGET == 'paper' the figure is
    resized to fit an IEEE double-column layout; its content is unchanged.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    divisor, unit = _choose_time_unit(result.P * result.sampling_interval_seconds)
    step = result.sampling_interval_seconds / divisor

    for i, section in enumerate(result.sections):
        ax.plot(
            np.arange(len(section)) * step,
            section,
            color="tab:gray",
            linewidth=0.5,
            alpha=0.5,
            label=r"Sections ($\tilde{\mathbf{x}}_j$)" if i == 0 else None,
        )

    ax.plot(
        np.arange(result.P) * step,
        result.phi,
        color="black",
        linewidth=2.5,
        label=r"Repeated pattern ($\mathbf{\phi}$)",
    )

    ax.set_xlabel(f"Position within the period [{unit}]")
    ax.set_ylabel(_VALUE_LABEL)
    ax.set_xlim(0, (result.P - 1) * step)
    ax.set_ylim(0, 3 * result.phi.max())
    ax.grid(True)
    ax.legend(loc="upper right")

    if cfg.TARGET == "paper":
        _size_for_paper(fig, height_cm=7)

    fig.tight_layout()
    if not cfg.SHOW_PLOT:
        fig.savefig(output_path)
        plt.close(fig)


def show_all():
    """Display every currently open matplotlib figure at once (blocking
    until all of them are closed).
    """
    plt.show()
