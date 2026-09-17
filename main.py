"""
Command-line entry point for the Sustained Anomaly Recognition (SAR) module.

Usage
-----
    python main.py --input data/cesnet_ip1367_n_packets_hourly.csv \\
                    --output-dir results --plot

The input CSV must contain a timestamp column and a value column (see
TIME_COLUMN / VALUE_COLUMN in main_config.py). All manually-tunable
parameters (T, W, LAMBDA, G_MAX, D_MIN) are read from main_config.py; they
can be overridden on the command line for quick experimentation.

Every file written to --output-dir is named
    <datetime>_<what>_<manual-parameter-settings>.<ext>
so that a result file can always be traced back to the exact parameter
settings (and run time) that produced it, e.g.:
    20260917_143012_point_scores_T86400W12LAMBDA3.0GMAX3DMIN5.csv
"""

import argparse
import os
from datetime import datetime

import pandas as pd

import main_config as cfg
import plotter
import sar


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sustained Anomaly Recognition (SAR) on a time series CSV.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file (timestamp, value).")
    parser.add_argument("--time-column", default=None, help=f"Timestamp column name (default: {cfg.TIME_COLUMN}).")
    parser.add_argument("--value-column", default=None, help=f"Value column name (default: {cfg.VALUE_COLUMN}).")
    parser.add_argument("--output-dir", default="results", help="Directory to write outputs into (default: results).")

    parser.add_argument("--period-seconds", type=float, default=None, help=f"Override T (default: {cfg.PERIOD_SECONDS}).")
    parser.add_argument("--smoothing-window", type=int, default=None, help=f"Override W (default: {cfg.SMOOTHING_WINDOW}).")
    parser.add_argument("--lam", type=float, default=None, help=f"Override LAMBDA (default: {cfg.LAMBDA}).")
    parser.add_argument("--g-max", type=int, default=None, help=f"Override G_MAX (default: {cfg.G_MAX}).")
    parser.add_argument("--d-min", type=int, default=None, help=f"Override D_MIN (default: {cfg.D_MIN}).")

    parser.add_argument("--plot", action="store_true", help="Also save the 4-subplot summary figure as a PDF.")
    return parser.parse_args()


def build_run_tag(params: dict) -> str:
    """Concatenate manual parameter names/values into a filename tag, e.g.
    {"T": 86400, "W": 12, "LAMBDA": 3.0} -> "T86400W12LAMBDA3.0"
    """
    return "".join(f"{name}{value}" for name, value in params.items())


def make_output_path(output_dir, dt_str, what, run_tag, ext):
    return os.path.join(output_dir, f"{dt_str}_{what}_{run_tag}.{ext}")


def main():
    args = parse_args()

    t, x = sar.load_time_series(args.input, args.time_column, args.value_column)

    period_seconds = cfg.PERIOD_SECONDS if args.period_seconds is None else args.period_seconds
    smoothing_window = cfg.SMOOTHING_WINDOW if args.smoothing_window is None else args.smoothing_window
    lam = cfg.LAMBDA if args.lam is None else args.lam
    g_max = cfg.G_MAX if args.g_max is None else args.g_max
    d_min = cfg.D_MIN if args.d_min is None else args.d_min

    result = sar.run_sar(
        t,
        x,
        period_seconds=period_seconds,
        smoothing_window=smoothing_window,
        lam=lam,
        g_max=g_max,
        d_min=d_min,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = build_run_tag(
        {
            "T": period_seconds,
            "W": smoothing_window,
            "LAMBDA": lam,
            "GMAX": g_max,
            "DMIN": d_min,
        }
    )

    def outpath(what, ext):
        return make_output_path(args.output_dir, dt_str, what, run_tag, ext)

    # Point-level data: everything needed to redraw subplots 1-3.
    points_path = outpath("point_scores", "csv")
    pd.DataFrame(
        {
            "timestamp": result.t,
            "value": result.x,
            "distance": result.d,
            "smoothed_deviation": result.s,
            "pattern_deviation_label": result.y,
            "threshold": result.theta,
        }
    ).to_csv(points_path, index=False)

    # Run-level data at each stage: everything needed to redraw subplot 4.
    raw_runs_path = outpath("runs_raw", "csv")
    result.raw_runs_df.to_csv(raw_runs_path, index=False)

    merged_runs_path = outpath("runs_merged", "csv")
    result.merged_runs_df.to_csv(merged_runs_path, index=False)

    final_runs_path = outpath("runs_final", "csv")
    result.final_runs_df.to_csv(final_runs_path, index=False)

    print(f"Inferred period P = {result.P} timesteps")
    print(f"Dynamic threshold Theta~ = {result.theta:.4f}")
    print(f"Raw runs (R): {len(result.raw_runs)}, merged runs (R'): {len(result.merged_runs)}, "
          f"final runs (R''): {len(result.final_runs)}")
    if len(result.final_runs_df):
        print(result.final_runs_df.to_string(index=False))

    print(f"\nWrote point-level scores to {points_path}")
    print(f"Wrote raw runs to {raw_runs_path}")
    print(f"Wrote merged runs to {merged_runs_path}")
    print(f"Wrote final runs to {final_runs_path}")

    if args.plot:
        plot_path = outpath("sar_plot", "pdf")
        plotter.plot_sar_result(result, plot_path)
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
