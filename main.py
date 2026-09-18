"""
Command-line entry point for the Sustained Anomaly Recognition (SAR) module.

Modes (--mode):

  single      Run SAR once on --input and (optionally, --plot) show/save its
              summary figures. This is the default.

                  python main.py --input data/cesnet_ip1367_n_packets_hourly.csv \\
                                  --output-dir results --plot

  single-wide Same as 'single', but defaults --input to main_config.WIDE_INPUT
              (the CESNET example series) and its own WIDE_* SAR parameters,
              and - only when TARGET == 'paper' - plots with
              plotter.plot_sar_result_wide() instead of plot_sar_result(),
              spanning the full IEEE double-column width rather than the
              single-column-like width used by 'single'. With TARGET ==
              'view' it behaves exactly like 'single'.

                  python main.py --mode single-wide --plot

  cesnet      Run SAR on all three CESNET_INPUTS metrics (n_packets, n_flows,
              n_bytes - the same IP 1367 traffic, three different measures),
              using the WIDE_* SAR parameters for all three. There is no
              ground truth for any of them, so this saves the anomalous
              sections themselves (results/CESNET/range_candidates/, like
              --mode evaluate does for NEK) and prints/saves how many SAR
              found per metric (results/CESNET/anomaly_counts.csv); it then
              plots the n_packets series (with its C-WDE overlay) the same
              way --mode single-wide does.

                  python main.py --mode cesnet

  evaluate    Run SAR over every series in --eval-dataset-dir, save each
              series' anomaly-region candidates, then score them against
              ground-truth labels (evaluation.py, PRTS range-based metrics)
              and show/save the resulting metrics.

                  python main.py --mode evaluate

  tune        Use Optuna to search SMOOTHING_WINDOW, LAMBDA, G_MAX and D_MIN
              for the combination that maximises the mean PRTS F1 score over
              --eval-dataset-dir. No plotting, no status bar, and nothing is
              written to disk; the best parameters are only printed at the end.

                  python main.py --mode tune

All manually-tunable SAR parameters (T, W, LAMBDA, G_MAX, D_MIN) and the
evaluation-pipeline settings are read from main_config.py; they can be
overridden on the command line for quick experimentation.

In --mode single, every file written to --output-dir is named
    <datetime>_<what>_<manual-parameter-settings>.<ext>
so that a result file can always be traced back to the exact parameter
settings (and run time) that produced it, e.g.:
    20260917_143012_point_scores_T86400W12LAMBDA3.0GMAX3DMIN5.csv

In --mode evaluate, per-series anomaly-region candidates are always written
to results/<eval-dataset-name>/range_candidates/<original file name>.csv
(no date or parameter tag, so they overwrite on rerun). The metrics
(anomaly_det_metrics.csv, anomaly_det_metrics_mean.csv) and their plots
respect SHOW_PLOT like --mode single does: saved to
results/<eval-dataset-name>/ when SHOW_PLOT is False, or only displayed
when it is True. With TARGET == 'view' both a per-series breakdown and the
dataset-wide mean are plotted; with TARGET == 'paper' only the mean-metrics
plot is produced, sized for an IEEE double-column figure.
"""

import argparse
import glob
import os
from datetime import datetime

import pandas as pd

import main_config as cfg
import sar


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sustained Anomaly Recognition (SAR) on a time series CSV.")
    parser.add_argument(
        "--mode",
        choices=["single", "single-wide", "cesnet", "evaluate", "tune"],
        default=cfg.EVAL_MODE,
        help=f"'single' runs SAR on --input; 'single-wide' is the same but for a full "
        f"double-column-width example figure (default input: WIDE_INPUT); 'cesnet' "
        f"runs SAR on all three CESNET_INPUTS metrics and counts anomalous sections in "
        f"each (no ground truth), plus the single-wide n_packets plot; 'evaluate' "
        f"runs it over a whole labelled dataset; 'tune' searches "
        f"SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN for the best mean F1 score on that "
        f"dataset (default: {cfg.EVAL_MODE}).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=f"Path to the input CSV file (timestamp, value) "
        f"(default: {cfg.INPUT} for --mode single, {cfg.WIDE_INPUT} for single-wide).",
    )
    parser.add_argument("--time-column", default=None, help=f"Timestamp column name (default: {cfg.TIME_COLUMN}).")
    parser.add_argument("--value-column", default=None, help=f"Value column name (default: {cfg.VALUE_COLUMN}).")
    parser.add_argument("--output-dir", default="results", help="Directory to write outputs into (default: results).")

    parser.add_argument("--period-seconds", type=float, default=None, help=f"Override T (default: {cfg.PERIOD_SECONDS}).")
    parser.add_argument("--smoothing-window", type=int, default=None, help=f"Override W (default: {cfg.SMOOTHING_WINDOW}).")
    parser.add_argument("--lam", type=float, default=None, help=f"Override LAMBDA (default: {cfg.LAMBDA}).")
    parser.add_argument("--g-max", type=int, default=None, help=f"Override G_MAX (default: {cfg.G_MAX}).")
    parser.add_argument("--d-min", type=int, default=None, help=f"Override D_MIN (default: {cfg.D_MIN}).")

    parser.add_argument("--plot", action="store_true", help="Also save the summary figure(s) (--mode single only).")

    parser.add_argument(
        "--eval-dataset-dir", default=cfg.EVAL_DATASET_DIR, help=f"Folder of labelled series to evaluate (default: {cfg.EVAL_DATASET_DIR})."
    )
    parser.add_argument(
        "--eval-dataset-name",
        default=cfg.EVAL_DATASET_NAME,
        help=f"Name of the results/<name> output folder for --mode evaluate (default: {cfg.EVAL_DATASET_NAME}).",
    )
    parser.add_argument(
        "--eval-file-glob", default=cfg.EVAL_FILE_GLOB, help=f"Filename pattern within --eval-dataset-dir (default: {cfg.EVAL_FILE_GLOB})."
    )
    parser.add_argument(
        "--eval-label-column",
        default=cfg.EVAL_LABEL_COLUMN,
        help=f"Ground-truth binary anomaly label column name (default: {cfg.EVAL_LABEL_COLUMN}).",
    )

    return parser.parse_args()


def build_run_tag(params: dict) -> str:
    """Concatenate manual parameter names/values into a filename tag, e.g.
    {"T": 86400, "W": 12, "LAMBDA": 3.0} -> "T86400W12LAMBDA3.0"
    """
    return "".join(f"{name}{value}" for name, value in params.items())


def make_output_path(output_dir, dt_str, what, run_tag, ext):
    return os.path.join(output_dir, f"{dt_str}_{what}_{run_tag}.{ext}")


def find_eval_files(args):
    """Files matching --eval-file-glob in --eval-dataset-dir, minus any
    named in main_config.EVAL_EXCLUDE_FILES. Used by --mode evaluate/tune.
    """
    files = sorted(glob.glob(os.path.join(args.eval_dataset_dir, args.eval_file_glob)))
    files = [f for f in files if os.path.basename(f) not in cfg.EVAL_EXCLUDE_FILES]
    if not files:
        raise FileNotFoundError(f"No files matching '{args.eval_file_glob}' found in {args.eval_dataset_dir}")
    return files


def resolve_sar_params(args, wide=False):
    """Manual SAR parameters, with CLI overrides applied on top of
    main_config - the WIDE_* settings when wide=True (--mode single-wide),
    otherwise the plain ones.
    """
    if wide:
        defaults = {
            "smoothing_window": cfg.WIDE_SMOOTHING_WINDOW,
            "lam": cfg.WIDE_LAMBDA,
            "g_max": cfg.WIDE_G_MAX,
            "d_min": cfg.WIDE_D_MIN,
        }
    else:
        defaults = {
            "smoothing_window": cfg.SMOOTHING_WINDOW,
            "lam": cfg.LAMBDA,
            "g_max": cfg.G_MAX,
            "d_min": cfg.D_MIN,
        }
    return {
        "period_seconds": cfg.PERIOD_SECONDS if args.period_seconds is None else args.period_seconds,
        "smoothing_window": defaults["smoothing_window"] if args.smoothing_window is None else args.smoothing_window,
        "lam": defaults["lam"] if args.lam is None else args.lam,
        "g_max": defaults["g_max"] if args.g_max is None else args.g_max,
        "d_min": defaults["d_min"] if args.d_min is None else args.d_min,
    }


def _plot_single_wide(plotter, result, plot_path, cwde_mask):
    """Plot with plotter.plot_sar_result_wide() when TARGET == 'paper'
    (matching --mode single-wide's own rule), else fall back to the plain
    plot_sar_result() (identical to --mode single's plot in that case).
    Shared by --mode single-wide and --mode cesnet.
    """
    if cfg.TARGET == "paper":
        plotter.plot_sar_result_wide(result, plot_path, cwde_mask=cwde_mask)
    else:
        plotter.plot_sar_result(result, plot_path, cwde_mask=cwde_mask)


def run_single(args):
    import plotter  # local import: keeps plotter's backend selection out of --mode evaluate

    wide = args.mode == "single-wide"
    input_path = args.input or (cfg.WIDE_INPUT if wide else cfg.INPUT)
    params = resolve_sar_params(args, wide)

    if wide:
        # plotter.py and the ground-truth/C-WDE loading below read these
        # directly off main_config; swap in single-wide's own settings for
        # the rest of this (single-CLI-invocation) run.
        cfg.VALUE_LABEL = cfg.WIDE_VALUE_LABEL
        cfg.CWDE_SIGNALS_PATH = cfg.WIDE_CWDE_SIGNALS_PATH
        cfg.CWDE_SIGNAL_COLUMN = cfg.WIDE_CWDE_SIGNAL_COLUMN

    t, x = sar.load_time_series(input_path, args.time_column, args.value_column)
    result = sar.run_sar(t, x, **params)

    cwde_mask = None
    if cfg.CWDE_SIGNALS_PATH:
        cwde_mask = sar.load_binary_signal(cfg.CWDE_SIGNALS_PATH, result.t, signal_column=cfg.CWDE_SIGNAL_COLUMN)

    # Ground-truth labels, if the input CSV happens to carry one (e.g. the
    # labelled NEK/NAB/IOPS/Yahoo series) - shown on the paper-mode plot only.
    ground_truth_mask = None
    if cfg.EVAL_LABEL_COLUMN in pd.read_csv(input_path, nrows=0).columns:
        ground_truth_mask = sar.load_binary_signal(
            input_path, result.t, time_column=args.time_column, signal_column=cfg.EVAL_LABEL_COLUMN
        )

    print(f"Inferred period P = {result.P} timesteps")
    print(f"Dynamic threshold Theta~ = {result.theta:.4f}")
    print(f"Raw runs (R): {len(result.raw_runs)}, merged runs (R'): {len(result.merged_runs)}, "
          f"final runs (R''): {len(result.final_runs)}")
    if len(result.final_runs_df):
        print(result.final_runs_df.to_string(index=False))

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = build_run_tag(
        {
            "T": params["period_seconds"],
            "W": params["smoothing_window"],
            "LAMBDA": params["lam"],
            "GMAX": params["g_max"],
            "DMIN": params["d_min"],
        }
    )

    def outpath(what, ext):
        return make_output_path(args.output_dir, dt_str, what, run_tag, ext)

    # SHOW_PLOT means figures are only displayed interactively; nothing is
    # written to --output-dir in that case, CSVs included.
    if not cfg.SHOW_PLOT:
        os.makedirs(args.output_dir, exist_ok=True)

        # Point-level data: everything needed to redraw subplots 1-3.
        points_path = outpath("point_scores", "csv")
        points_df = pd.DataFrame(
            {
                "timestamp": result.t,
                "value": result.x,
                "distance": result.d,
                "smoothed_deviation": result.s,
                "pattern_deviation_label": result.y,
                "threshold": result.theta,
            }
        )
        if cwde_mask is not None:
            points_df["cwde_anomaly_signal"] = cwde_mask.astype(int)
        if ground_truth_mask is not None:
            points_df["ground_truth_label"] = ground_truth_mask.astype(int)
        points_df.to_csv(points_path, index=False)

        # Run-level data at each stage: everything needed to redraw subplot 4.
        raw_runs_path = outpath("runs_raw", "csv")
        result.raw_runs_df.to_csv(raw_runs_path, index=False)

        merged_runs_path = outpath("runs_merged", "csv")
        result.merged_runs_df.to_csv(merged_runs_path, index=False)

        final_runs_path = outpath("runs_final", "csv")
        result.final_runs_df.to_csv(final_runs_path, index=False)

        print(f"\nWrote point-level scores to {points_path}")
        print(f"Wrote raw runs to {raw_runs_path}")
        print(f"Wrote merged runs to {merged_runs_path}")
        print(f"Wrote final runs to {final_runs_path}")
    else:
        print("\nSHOW_PLOT is True: nothing is written to disk, results are only displayed.")

    if args.plot:
        plot_path = None if cfg.SHOW_PLOT else outpath("sar_plot", "pdf")
        if wide:
            _plot_single_wide(plotter, result, plot_path, cwde_mask)
        else:
            plotter.plot_sar_result(result, plot_path, cwde_mask=cwde_mask, ground_truth_mask=ground_truth_mask)
        if cfg.SHOW_PLOT:
            print("Displayed plot in a matplotlib window.")
        else:
            print(f"Wrote plot to {plot_path}")

        if cfg.PLOT_PATTERN:
            pattern_plot_path = None if cfg.SHOW_PLOT else outpath("pattern_plot", "pdf")
            plotter.plot_pattern_sections(result, pattern_plot_path)
            if cfg.SHOW_PLOT:
                print("Displayed pattern plot in a matplotlib window.")
            else:
                print(f"Wrote pattern plot to {pattern_plot_path}")

        if cfg.SHOW_PLOT:
            plotter.show_all()


def run_cesnet(args):
    """Run SAR on all three CESNET_INPUTS metrics (no ground truth to score
    against - just save the anomalous sections themselves, like --mode
    evaluate does for NEK, and print/save how many SAR found in each), then
    plot the n_packets series (with its C-WDE overlay) using the same
    plotting rule as --mode single-wide.
    """
    import plotter

    params = resolve_sar_params(args, wide=True)
    cfg.VALUE_LABEL = cfg.WIDE_VALUE_LABEL

    out_dir = os.path.join(args.output_dir, cfg.CESNET_DATASET_NAME)
    range_dir = os.path.join(out_dir, "range_candidates")
    os.makedirs(range_dir, exist_ok=True)

    counts = []
    packets_result, packets_cwde_mask = None, None
    for name, path in cfg.CESNET_INPUTS.items():
        t, x = sar.load_time_series(path, args.time_column, args.value_column)
        result = sar.run_sar(t, x, **params)
        n_sections = len(result.final_runs)
        counts.append({"dataset": name, "n_anomalous_sections": n_sections})
        print(f"{name}: {n_sections} anomalous section(s) (final runs R'')")

        # Anomalous sections themselves, same convention as --mode
        # evaluate's range_candidates: original file name, no date or
        # parameter tag, so reruns overwrite in place.
        result.final_runs_df.to_csv(os.path.join(range_dir, os.path.basename(path)), index=False)

        if name == "n_packets":
            packets_result = result
            if cfg.WIDE_CWDE_SIGNALS_PATH:
                packets_cwde_mask = sar.load_binary_signal(
                    cfg.WIDE_CWDE_SIGNALS_PATH, result.t, signal_column=cfg.WIDE_CWDE_SIGNAL_COLUMN
                )

    print(f"\nSaved {len(cfg.CESNET_INPUTS)} anomalous-section file(s) to {range_dir}")

    counts_df = pd.DataFrame(counts, columns=["dataset", "n_anomalous_sections"])
    print()
    print(counts_df.to_string(index=False))

    if not cfg.SHOW_PLOT:
        counts_path = os.path.join(out_dir, "anomaly_counts.csv")
        counts_df.to_csv(counts_path, index=False)
        print(f"\nWrote anomaly counts to {counts_path}")
    else:
        print("\nSHOW_PLOT is True: anomaly counts are only displayed, not written to disk.")

    plot_path = None if cfg.SHOW_PLOT else os.path.join(out_dir, "cesnet_example_plot.pdf")
    _plot_single_wide(plotter, packets_result, plot_path, packets_cwde_mask)
    if cfg.SHOW_PLOT:
        print("Displayed the example (n_packets) plot in a matplotlib window.")
        plotter.show_all()
    else:
        print(f"Wrote example plot to {plot_path}")


def run_evaluate(args):
    import sys
    import status_bar
    import evaluation
    import plotter

    # status_bar draws with a Unicode block character; some Windows consoles
    # default to a codepage that can't encode it.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    files = find_eval_files(args)

    params = resolve_sar_params(args)

    out_dir = os.path.join(args.output_dir, args.eval_dataset_name)
    range_dir = os.path.join(out_dir, "range_candidates")
    os.makedirs(range_dir, exist_ok=True)

    # Drop stale range-candidate files left over from a previous run (e.g. a
    # series later added to EVAL_EXCLUDE_FILES) so evaluate_dataset() below
    # can't silently re-include them.
    current_names = {os.path.basename(f) for f in files}
    for stale_path in glob.glob(os.path.join(range_dir, "*.csv")):
        if os.path.basename(stale_path) not in current_names:
            os.remove(stale_path)

    n = len(files)
    for i, file_path in enumerate(files):
        name = os.path.basename(file_path)
        status_bar.draw_status_bar(i, n, text=f"running SAR on {name}")

        t, x = sar.load_time_series(file_path, args.time_column, args.value_column)
        result = sar.run_sar(t, x, **params)
        # Range candidates: original file name, no date/parameter tag, so
        # reruns overwrite in place.
        result.final_runs_df.to_csv(os.path.join(range_dir, name), index=False)
    status_bar.draw_status_bar(n, n, text="done")
    print()

    print(f"\nSaved {n} range-candidate file(s) to {range_dir}")

    metrics_df = evaluation.evaluate_dataset(args.eval_dataset_dir, range_dir, args.eval_label_column)
    mean_metrics_df = evaluation.compute_mean_metrics(metrics_df)

    if not cfg.SHOW_PLOT:
        metrics_path = os.path.join(out_dir, "anomaly_det_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Wrote metrics to {metrics_path}")

        mean_metrics_path = os.path.join(out_dir, "anomaly_det_metrics_mean.csv")
        mean_metrics_df.to_csv(mean_metrics_path, index=False)
        print(f"Wrote mean metrics to {mean_metrics_path}")
    else:
        print("\nSHOW_PLOT is True: metrics are only displayed, not written to disk.")

    print(metrics_df.to_string(index=False))
    print(mean_metrics_df.to_string(index=False))

    # The per-series breakdown is a 'view'-only diagnostic; 'paper' mode
    # only produces the paper-ready mean-metrics summary.
    if cfg.TARGET == "view":
        per_series_path = None if cfg.SHOW_PLOT else os.path.join(out_dir, "anomaly_det_metrics_per_series.pdf")
        plotter.plot_metrics_per_series(metrics_df, per_series_path)

    mean_plot_path = None if cfg.SHOW_PLOT else os.path.join(out_dir, "anomaly_det_metrics_mean.pdf")
    plotter.plot_mean_metrics(mean_metrics_df, mean_plot_path)

    if cfg.SHOW_PLOT:
        plotter.show_all()


def run_tune(args):
    import optuna
    import evaluation

    files = find_eval_files(args)

    period_seconds = cfg.PERIOD_SECONDS if args.period_seconds is None else args.period_seconds

    # Preload every series once; only the tuned parameters change across
    # trials, and nothing is written to disk during tuning.
    series = []
    for file_path in files:
        t, x = sar.load_time_series(file_path, args.time_column, args.value_column)
        true_labels = pd.read_csv(file_path)[args.eval_label_column].to_numpy()
        series.append((t, x, true_labels))

    sw_low, sw_high = cfg.TUNE_SMOOTHING_WINDOW_RANGE
    lam_low, lam_high = cfg.TUNE_LAMBDA_RANGE
    gmax_low, gmax_high = cfg.TUNE_G_MAX_RANGE
    dmin_low, dmin_high = cfg.TUNE_D_MIN_RANGE

    def objective(trial):
        smoothing_window = trial.suggest_int("smoothing_window", sw_low, sw_high)
        lam = trial.suggest_float("lam", lam_low, lam_high)
        g_max = trial.suggest_int("g_max", gmax_low, gmax_high)
        d_min = trial.suggest_int("d_min", dmin_low, dmin_high)

        f_scores = []
        for t, x, true_labels in series:
            result = sar.run_sar(
                t,
                x,
                period_seconds=period_seconds,
                smoothing_window=smoothing_window,
                lam=lam,
                g_max=g_max,
                d_min=d_min,
            )
            _, _, fscore = evaluation.evaluate_series(true_labels, result.final_runs_df)
            f_scores.append(fscore)
        return float(sum(f_scores) / len(f_scores))

    print(f"Tuning on {len(series)} series from {args.eval_dataset_dir}, {cfg.TUNE_N_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.TUNE_N_TRIALS)

    print()
    print(f"Best mean F1 score: {study.best_value:.4f}")
    print("Best parameters:")
    for name, value in study.best_params.items():
        print(f"  {name} = {value}")


def main():
    args = parse_args()
    if args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "tune":
        run_tune(args)
    elif args.mode == "cesnet":
        run_cesnet(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
