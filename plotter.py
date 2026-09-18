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

plot_sar_result_wide() is the same two-axis 'paper' layout, but 30% wider
than the full IEEE double-column width (instead of the single-column-like
75% used by plot_sar_result()) and with a half-height region strip (no
ground-truth row - CESNET has none) so the main axis gets that space back;
used by --mode single-wide and --mode cesnet for a single wide example
figure (e.g. the CESNET series), regardless of TARGET.

plot_pattern_sections() produces a separate figure overlaying every
period-length section of the input series with the periodic pattern (phi).
With TARGET == 'paper' its content is unchanged; it is only resized to fit
an IEEE double-column figure.

plot_metrics_per_series() and plot_mean_metrics() chart the precision,
recall, and F1 score from evaluation.py's dataset-wide scoring: one bar
group per series, and the dataset-wide mean, respectively. With
TARGET == 'paper', plot_mean_metrics() is the only one meant to be used
(the per-series breakdown is a 'view'-only diagnostic) and is resized for
an IEEE double-column figure.

When main_config.SHOW_PLOT is True, every function above leaves its figure
open instead of showing it immediately; call show_all() once after creating
all figures so they appear together in separate windows at the same time.
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


def _grow_bottom_margin(fig, extra_inches):
    """Add extra_inches of blank space below every axes in fig, without
    changing any axes' absolute size.

    Grows the figure's height and shifts all existing axes up by that same
    amount (in inches), so their absolute size is preserved exactly; the
    added space appears as new margin at the bottom, e.g. for a legend
    placed there afterwards with fig.legend(...). Must be called after the
    figure's layout (tight_layout etc.) is finalized, and nothing that
    re-flows the layout (tight_layout, subplots_adjust) should run after it.
    """
    w, h = fig.get_size_inches()
    new_h = h + extra_inches
    for ax in fig.axes:
        pos = ax.get_position()
        new_y0 = (pos.y0 * h + extra_inches) / new_h
        new_height = pos.height * h / new_h
        ax.set_position([pos.x0, new_y0, pos.width, new_height])
    fig.set_size_inches(w, new_h)


def _grow_main_axis(fig, growth_fraction):
    """Grow the first axes in fig (the main data axis) by growth_fraction
    of its own height (e.g. 0.2 for 20%), growing the figure by that same
    absolute amount and shifting the second axes (the region strip) down to
    make room below it, without changing either the gap between the two
    axes or the region strip's own size.

    The blank space above the main axis (where its legend sits, anchored
    to it via bbox_to_anchor) is kept at the same absolute size, so the
    main axis grows downward, not upward - it never encroaches on that
    legend. Must run after subplots_adjust (needs the axes' final
    pre-growth positions) and before _grow_bottom_margin() (which shifts
    every axes' position on top of whatever this leaves them at).
    """
    ax_main, ax_regions = fig.axes[0], fig.axes[1]
    w, h = fig.get_size_inches()
    pos_m, pos_r = ax_main.get_position(), ax_regions.get_position()

    extra = growth_fraction * pos_m.height * h
    new_h = h + extra

    depth_top = h - (pos_m.y0 + pos_m.height) * h  # blank space above ax_main
    gap = pos_m.y0 * h - (pos_r.y0 + pos_r.height) * h  # gap between the two axes

    new_main_height = pos_m.height * h + extra
    new_main_y0 = new_h - depth_top - new_main_height

    new_regions_height = pos_r.height * h  # unchanged
    new_regions_y0 = new_main_y0 - gap - new_regions_height

    ax_main.set_position([pos_m.x0, new_main_y0 / new_h, pos_m.width, new_main_height / new_h])
    ax_regions.set_position([pos_r.x0, new_regions_y0 / new_h, pos_r.width, new_regions_height / new_h])
    fig.set_size_inches(w, new_h)


def _size_for_paper(fig, height_cm, width_scale=0.75):
    """Resize fig to width_scale * the IEEE double-column width, at the
    given height (cm). width_scale=0.75 (default) approximates a single
    column; 1.0 spans the full double-column width.

    The full figure is first shrunk by 30% (preserving aspect ratio), then
    the height alone is grown back by 30% twice on top of that.
    """
    width_cm = _IEEE_DOUBLE_COLUMN_WIDTH_CM * width_scale
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


def plot_sar_result(result: "sar.SARResult", output_path, cwde_mask=None, ground_truth_mask=None):
    """Build the SAR summary figure (layout depends on main_config.TARGET).

    cwde_mask, if given, is a boolean array (same length as result.t)
    marking points flagged by the C-WDE detector ensemble; these are
    overlaid as an extra scatter series on the top axis.

    ground_truth_mask, if given, is a boolean array (same length as
    result.t) marking ground-truth anomalous points; only used by the
    TARGET == 'paper' layout, as an extra row on its bottom axis.

    Depending on main_config.SHOW_PLOT, either leaves it open for show_all()
    to display, or saves it to output_path (PDF).
    """
    if cfg.TARGET == "paper":
        _plot_sar_result_paper(result, output_path, cwde_mask, ground_truth_mask)
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
    ax1.set_ylabel(cfg.VALUE_LABEL)
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


def _build_paper_style_fig(
    result: "sar.SARResult", cwde_mask=None, ground_truth_mask=None, cwde_legend_ncol=2, region_height_ratio=1
):
    """Build the 2-axis (original data + thin region strip) figure shared by
    _plot_sar_result_paper() and plot_sar_result_wide(); only sizing and
    margins differ between them, applied by the caller afterward.

    cwde_legend_ncol: how many columns the top legend wraps to when a C-WDE
    overlay adds a 3rd handle (2 handles always get 1 row); the narrower
    'paper' figure needs 1 (full 3-row stack) where the wider 'wide' figure
    fits 2, since the 3rd label doesn't fit two-per-row at that width.

    region_height_ratio: height of the bottom region strip relative to the
    main axis (3); the freed-up space (a smaller ratio) goes to the main
    axis automatically, since the figure's total height is fixed.

    Returns (fig, region_handles, legend_rows) - region_handles is the
    legend handle list for the bottom strip, added by the caller once the
    figure has its final size (fig.legend() needs the final figure-relative
    position); legend_rows is how many rows the top legend wrapped to, for
    the caller to size the top margin accordingly.
    """
    fig, (ax_main, ax_regions) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, region_height_ratio]}
    )

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
    ax_main.set_ylabel(cfg.VALUE_LABEL)
    ax_main.grid(True)
    main_handles, main_labels = ax_main.get_legend_handles_labels()
    # 3 handles (with a C-WDE overlay) don't fit on one line at this width;
    # wrap to cwde_legend_ncol columns/rows instead of shrinking to fit.
    main_ncol = cwde_legend_ncol if len(main_handles) >= 3 else len(main_handles)
    ax_main.legend(
        main_handles,
        main_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=main_ncol,
        frameon=False,
    )
    ax_main.tick_params(labelbottom=False)

    # 2. Thin strip: ground truth (if available) and the final identified
    #    anomaly regions (R'').
    region_handles = []
    if ground_truth_mask is not None:
        gt_runs = sar.extract_runs(ground_truth_mask.astype(int))
        region_handles.append(_plot_runs_row(ax_regions, gt_runs, result.t, 1, "tab:gray", "Ground truth regions"))
        final_row, ylim_top = 0, 1.5
    else:
        final_row, ylim_top = 0, 0.5
    region_handles.append(
        _plot_runs_row(ax_regions, result.final_runs, result.t, final_row, "tab:red", "SAR anomaly region candidates")
    )
    ax_regions.set_yticks([])
    ax_regions.set_ylim(-0.5, ylim_top)
    ax_regions.grid(True, axis="x")
    plt.setp(ax_regions.get_xticklabels(), rotation=30, ha="right")

    legend_rows = -(-len(main_handles) // main_ncol)  # ceil division
    return fig, region_handles, legend_rows


def _finish_paper_style_fig(fig, region_handles, output_path, width_scale, left, legend_rows, main_axis_growth=0.0):
    """Size, margin, and legend-place a _build_paper_style_fig() figure,
    then save/close it per main_config.SHOW_PLOT. left is the only margin
    that depends on width_scale (see plot_sar_result_wide()); the rest were
    hand-tuned once for the shared height=7cm and reused unchanged, except
    top, which needs extra headroom for each extra row the main legend
    wraps to beyond one (see _build_paper_style_fig's cwde_legend_ncol).

    main_axis_growth: passed straight to _grow_main_axis() (0.0 - the
    default - skips it) to grow the main axis beyond its plain height=7cm
    share, e.g. for plot_sar_result_wide().
    """
    _size_for_paper(fig, height_cm=7, width_scale=width_scale)
    # Manual margins instead of tight_layout(): tight_layout's automatic
    # padding tends to leave far more whitespace than actually needed here.
    # These were tuned by hand to the current legends/labels, as tight as
    # possible without clipping any of them.
    top = 0.885 - 0.095 * (legend_rows - 1)
    fig.subplots_adjust(left=left, right=0.995, top=top, bottom=0.25, hspace=0.05)

    if main_axis_growth:
        _grow_main_axis(fig, main_axis_growth)

    # Grow the canvas to make room for the region legend below ax_regions,
    # rather than carving the space out of the margins above (which would
    # shrink the axes) - the legend goes in the newly added strip.
    _grow_bottom_margin(fig, extra_inches=0.25)
    fig.legend(
        handles=region_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01314),  # ~half the legend's own height below its previous position (0.05)
        ncol=len(region_handles),
        frameon=False,
    )

    if not cfg.SHOW_PLOT:
        fig.savefig(output_path)
        plt.close(fig)


def _plot_sar_result_paper(result: "sar.SARResult", output_path, cwde_mask=None, ground_truth_mask=None):
    # cwde_legend_ncol=1: at this narrower width even 2 columns clips the
    # 3rd (C-WDE) label, so stack all 3 into their own row instead.
    fig, region_handles, legend_rows = _build_paper_style_fig(result, cwde_mask, ground_truth_mask, cwde_legend_ncol=1)
    _finish_paper_style_fig(fig, region_handles, output_path, width_scale=0.75, left=0.108, legend_rows=legend_rows)


def plot_sar_result_wide(result: "sar.SARResult", output_path, cwde_mask=None):
    """Same layout as the 'paper' plot_sar_result() (original data plus a
    thin final-regions strip), but 30% wider than the full IEEE
    double-column width (single-column-like 75% is used there); the bottom
    region strip is half as tall, and the main axis is 30% taller than its
    plain height=7cm share on top of that (so the figure ends up somewhat
    taller overall than plot_sar_result()'s, not just wider). Meant for a
    single wide example figure (e.g. the CESNET series), not the
    per-column 'paper' plot - use plot_sar_result() for that. Ignores
    main_config.TARGET.

    No ground_truth_mask: unlike plot_sar_result(), this is CESNET-only for
    now, which has no ground-truth labels.

    cwde_mask and SHOW_PLOT behave exactly as in plot_sar_result().
    """
    # cwde_legend_ncol=2 (default): the full double-column width fits the
    # 3rd (C-WDE) label two-per-row. region_height_ratio=0.5: half the
    # normal bottom-strip height, since there's no ground-truth row to show.
    fig, region_handles, legend_rows = _build_paper_style_fig(result, cwde_mask, region_height_ratio=0.5)
    # left is the only margin that needs to change for the wider canvas
    # (see _finish_paper_style_fig); main_axis_growth grows the main axis
    # (and the figure) by another 20% on top of the shared height=7cm,
    # shifting the region strip down to keep clear of it - see
    # _grow_main_axis().
    _finish_paper_style_fig(
        fig, region_handles, output_path, width_scale=1.3, left=0.062, legend_rows=legend_rows, main_axis_growth=0.2
    )


def plot_pattern_sections(result: "sar.SARResult", output_path):
    """Plot every period-length section of the input series (thin lines)
    overlaid with the periodic pattern phi (thick line).

    Depending on main_config.SHOW_PLOT, either leaves it open for show_all()
    to display, or saves it to output_path (PDF). The x-axis is clipped
    exactly to the span of the curves (0 to P-1). With TARGET == 'view' the
    y-axis is scaled to 3x the maximum of phi; with TARGET == 'paper' it is
    set to [phi_min - 0.5*range, phi_max + 0.5*range] and the figure is
    resized to fit an IEEE double-column layout. Either way, sections with
    much larger or smaller values may extend beyond the visible area.
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
    ax.set_ylabel(cfg.VALUE_LABEL)
    ax.set_xlim(0, (result.P - 1) * step)
    if cfg.TARGET == "paper":
        phi_min, phi_max = result.phi.min(), result.phi.max()
        margin = 0.5 * (phi_max - phi_min)
        ax.set_ylim(phi_min - margin, phi_max + margin)
    else:
        ax.set_ylim(0, 3 * result.phi.max())
    ax.grid(True)
    ax.legend(loc="lower left")

    if cfg.TARGET == "paper":
        _size_for_paper(fig, height_cm=7 * 0.6)  # 40% shorter than the summary plot

    fig.tight_layout()
    if not cfg.SHOW_PLOT:
        fig.savefig(output_path)
        plt.close(fig)


def plot_metrics_per_series(metrics_df, output_path):
    """Bar chart of precision/recall/F1 score per series.

    Depending on main_config.SHOW_PLOT, either leaves it open for show_all()
    to display, or saves it to output_path (PDF). Intended for TARGET ==
    'view' only; use plot_mean_metrics() for a 'paper'-ready summary.
    """
    n = len(metrics_df)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n), 5))
    x = np.arange(n)
    width = 0.25
    ax.bar(x - width, metrics_df["precision"], width, label="Precision", color="tab:blue")
    ax.bar(x, metrics_df["recall"], width, label="Recall", color="tab:orange")
    ax.bar(x + width, metrics_df["f_score"], width, label="F1 score", color="tab:green")
    ax.set_xticks(x)
    # Escape underscores for LaTeX text mode (file names contain them).
    ax.set_xticklabels([s.replace("_", r"\_") for s in metrics_df["series"]], rotation=90)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y")

    fig.tight_layout()
    if not cfg.SHOW_PLOT:
        fig.savefig(output_path)
        plt.close(fig)


def plot_mean_metrics(mean_metrics_df, output_path):
    """Bar chart of the dataset-wide mean precision/recall/F1 score.

    Depending on main_config.SHOW_PLOT, either leaves it open for show_all()
    to display, or saves it to output_path (PDF). With TARGET == 'paper' the
    figure is resized to fit an IEEE double-column layout.
    """
    row = mean_metrics_df.iloc[0]
    labels = ["Precision", "Recall", "F1 score"]
    values = [row["precision"], row["recall"], row["f_score"]]
    stds = [row.get("precision_std", 0.0), row.get("recall_std", 0.0), row.get("f_score_std", 0.0)]
    tops = [v + s for v, s in zip(values, stds)]  # bar + error bar cap, for headroom/label placement
    if cfg.TARGET == "paper":
        colors = ["#4C72B0", "#C44E52", "#55A868"]
    else:
        colors = ["tab:blue", "tab:orange", "tab:green"]

    bar_width = 0.4 if cfg.TARGET == "paper" else 0.8  # half-width, paper only
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.bar(x, values, width=bar_width, yerr=stds, capsize=4, color=colors, ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for i, v in enumerate(values):
        # At the bar's own top (not the error bar cap), offset sideways so
        # it clears the vertical std whisker, and up a little so it isn't
        # flush with the bar top.
        ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(6, 6), textcoords="offset points", ha="left", va="center")
    ax.set_ylabel("Mean metric value")
    if cfg.TARGET == "paper":
        # Let the maximum float with the data, with a little headroom above
        # the error bars' caps.
        ax.set_ylim(0, max(tops) * 1.075 if max(tops) > 0 else 1.0)
    else:
        # Usually a plain 0-1 range, but widen it if an error bar's cap
        # would otherwise poke out past the top.
        ax.set_ylim(0, max(1.0, max(tops) * 1.025))
    # Equalize the outer margins with the inter-bar gaps, so the outer bars
    # aren't crowded against the axes edges.
    ax.set_xlim(bar_width / 2 - 1, len(labels) - bar_width / 2)
    ax.grid(True, axis="y")

    if cfg.TARGET == "paper":
        _size_for_paper(fig, height_cm=7 * 0.6)  # match plot_pattern_sections's paper height

    fig.tight_layout()

    if cfg.TARGET == "paper":
        # Shrink the axes to 48% of the figure width, centered, without
        # changing the figure size itself.
        pos = ax.get_position()
        axes_width = 0.48
        ax.set_position([(1 - axes_width) / 2, pos.y0, axes_width, pos.height])

    if not cfg.SHOW_PLOT:
        fig.savefig(output_path)
        plt.close(fig)


def show_all():
    """Display every currently open matplotlib figure at once (blocking
    until all of them are closed).
    """
    plt.show()
