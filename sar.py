"""
Sustained Anomaly Recognition (SAR).

Estimates a periodic pattern (daily template) from the median of aligned
observations, measures each point's deviation from that pattern, smooths
and thresholds the deviation, and turns the resulting binary signal into
sustained-anomaly-region candidates via a merge-then-prune procedure over
runs of deviation points.

All manually-tunable parameters (T, W, LAMBDA, G_MAX, D_MIN) are read from
main_config.py.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import main_config as cfg


@dataclass
class SARResult:
    """Container for every intermediate array and the final output of SAR."""

    t: np.ndarray            # timestamps (datetime64[ns]), length L
    x: np.ndarray            # values, length L
    P: int                   # period length in timesteps
    sampling_interval_seconds: float  # inferred spacing between timesteps, in seconds
    phi: np.ndarray          # periodic pattern (daily template), length P
    sections: list = field(repr=False)  # x split into consecutive length-P chunks (last may be shorter)
    d: np.ndarray            # raw distance to pattern, length L
    s: np.ndarray            # smoothed deviation score, length L
    theta: float             # dynamic threshold Theta~
    y: np.ndarray            # binary pattern-deviation labels, length L (0/1)

    # Sustained-anomaly-region candidates at each stage of the merge-then-prune
    # procedure, as (start_idx, end_idx) inclusive index pairs.
    raw_runs: list = field(repr=False)      # R: runs straight out of y
    merged_runs: list = field(repr=False)   # R': after gap-bridging merge
    final_runs: list = field(repr=False)    # R'': after minimum-duration pruning

    # The same three stages, mapped back to timestamps/values.
    raw_runs_df: pd.DataFrame = field(repr=False)
    merged_runs_df: pd.DataFrame = field(repr=False)
    final_runs_df: pd.DataFrame = field(repr=False)


def load_time_series(csv_path, time_column=None, value_column=None):
    """Load a (timestamp, value) time series from a CSV file.

    Returns
    -------
    t : np.ndarray of datetime64[ns], sorted ascending
    x : np.ndarray of float
    """
    time_column = time_column or cfg.TIME_COLUMN
    value_column = value_column or cfg.VALUE_COLUMN

    df = pd.read_csv(csv_path)
    if time_column not in df.columns or value_column not in df.columns:
        raise ValueError(
            f"Expected columns '{time_column}' and '{value_column}' in {csv_path}, "
            f"found {list(df.columns)}"
        )

    if pd.api.types.is_numeric_dtype(df[time_column]):
        # A numeric timestamp column is a Unix epoch in seconds (e.g. NEK);
        # pd.to_datetime would otherwise silently treat it as nanoseconds.
        df[time_column] = pd.to_datetime(df[time_column], unit="s")
    else:
        df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)
    df = df.dropna(subset=[value_column])

    t = df[time_column].to_numpy()
    x = df[value_column].to_numpy(dtype=float)
    return t, x


def generate_faux_timestamps(n, sampling_interval_seconds, start="1970-01-01"):
    """n evenly-spaced datetime64[ns] timestamps, sampling_interval_seconds
    apart, starting at start (its absolute value is arbitrary - only the
    spacing matters to SAR). For series whose own "timestamp" column is
    really just a sequential row index rather than real wall-clock time
    (e.g. Yahoo, IOPS - see load_time_series_faux_timestamps()).
    """
    offsets = pd.to_timedelta(np.arange(n) * sampling_interval_seconds, unit="s")
    return (pd.Timestamp(start) + offsets).to_numpy()


def load_time_series_faux_timestamps(csv_path, value_column, sampling_interval_seconds):
    """Like load_time_series(), but ignores the CSV's own timestamp column
    and synthesizes evenly-spaced ones sampling_interval_seconds apart
    instead, via generate_faux_timestamps() - for series (e.g. Yahoo, IOPS)
    whose "timestamp" column is really just a sequential row index, not
    real time, so treating it as a Unix epoch (load_time_series's usual
    rule) would compute a nonsensical period relative to the series length.

    Returns
    -------
    t : np.ndarray of datetime64[ns]
    x : np.ndarray of float
    """
    value_column = value_column or cfg.VALUE_COLUMN

    df = pd.read_csv(csv_path)
    if value_column not in df.columns:
        raise ValueError(f"Expected column '{value_column}' in {csv_path}, found {list(df.columns)}")

    df = df.dropna(subset=[value_column]).reset_index(drop=True)
    t = generate_faux_timestamps(len(df), sampling_interval_seconds)
    x = df[value_column].to_numpy(dtype=float)
    return t, x


def load_binary_signal(csv_path, t, time_column=None, signal_column="signal"):
    """Load a 0/1 signal column from a CSV, aligned to timestamps t by exact
    timestamp match. Any timestamp in t missing from the CSV is treated as 0.

    Matching is done tz-naive: t and the CSV's timestamps can be tz-aware or
    not independently (e.g. a source CSV that dropped its UTC offset while t
    kept it) - only the wall-clock value needs to match.
    """
    time_column = time_column or cfg.TIME_COLUMN
    df = pd.read_csv(csv_path)
    if pd.api.types.is_numeric_dtype(df[time_column]):
        # A numeric timestamp column is a Unix epoch in seconds (e.g. NEK);
        # pd.to_datetime would otherwise silently treat it as nanoseconds.
        df[time_column] = pd.to_datetime(df[time_column], unit="s")
    else:
        df[time_column] = pd.to_datetime(df[time_column])
    series = df.set_index(time_column)[signal_column]
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    t_index = pd.DatetimeIndex(t)
    if t_index.tz is not None:
        t_index = t_index.tz_localize(None)
    aligned = series.reindex(t_index).fillna(0)
    return aligned.to_numpy(dtype=bool)


def infer_sampling_interval_seconds(t):
    """Infer the (dominant) sampling interval of a timestamp array, in seconds."""
    if len(t) < 2:
        raise ValueError("Need at least two timestamps to infer a sampling interval.")
    diffs = np.diff(t).astype("timedelta64[s]").astype(float)
    values, counts = np.unique(diffs, return_counts=True)
    return float(values[np.argmax(counts)])


def compute_period_timesteps(period_seconds, sampling_interval_seconds):
    """Convert the period T (seconds) into the period P (number of timesteps)."""
    P = round(period_seconds / sampling_interval_seconds)
    if P < 2:
        raise ValueError(
            f"Computed period P={P} timesteps is too short; check T/sampling interval."
        )
    return int(P)


def compute_sections(x, P):
    """Split x into consecutive length-P sections (the trailing section may be shorter)."""
    L = len(x)
    return [x[start : start + P] for start in range(0, L, P)]


def compute_periodic_pattern(x, P):
    """Estimate the daily/periodic template phi via the median of aligned points.

    phi_l = median{ x_j*P+l : j = 0, 1, ..., N-1 }  for l = 0, ..., P-1

    This naturally covers the trailing incomplete period at the end of the
    series as well, since every index i contributes to phi[i mod P]
    regardless of whether it belongs to a full period.
    """
    L = len(x)
    phi = np.empty(P, dtype=float)
    for l in range(P):
        phi[l] = np.median(x[l::P])
    return phi


def compute_deviation(x, phi, P):
    """d_i = |x_i - phi_{i mod P}|"""
    L = len(x)
    idx = np.arange(L) % P
    return np.abs(x - phi[idx])


def smooth_deviation(d, W):
    """Smooth the raw deviation d into s~ via a centered moving average of
    width W. Near the series boundaries the window is truncated to the
    available samples rather than padded, so every s~_i remains the average
    of real observations.
    """
    L = len(d)
    half = W // 2
    s = np.empty(L, dtype=float)
    cumsum = np.concatenate(([0.0], np.cumsum(d)))
    for i in range(L):
        lo = max(0, i - half)
        hi = min(L, i + half + 1)
        s[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return s


def compute_dynamic_threshold(s, lam):
    """Theta~ = mu_s~ + LAMBDA * sigma_s~"""
    return float(np.mean(s) + lam * np.std(s))


def binarize(s, theta):
    """y~_i = 1 if s~_i > Theta~ else 0."""
    return (s > theta).astype(int)


def extract_runs(y):
    """Extract maximal runs of consecutive 1s from a binary sequence.

    Returns a list of (start_idx, end_idx) inclusive index pairs.
    """
    runs = []
    L = len(y)
    i = 0
    while i < L:
        if y[i] == 1:
            j = i
            while j + 1 < L and y[j + 1] == 1:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def merge_runs(runs, g_max):
    """Bridge runs separated by a gap of at most G_MAX timesteps."""
    if not runs:
        return []
    merged = [runs[0]]
    for a, b in runs[1:]:
        prev_a, prev_b = merged[-1]
        if a - prev_b - 1 <= g_max:
            merged[-1] = (prev_a, b)
        else:
            merged.append((a, b))
    return merged


def prune_runs(runs, d_min):
    """Discard runs shorter than D_MIN timesteps."""
    return [(a, b) for (a, b) in runs if (b - a + 1) >= d_min]


def runs_to_dataframe(runs, t, x):
    """Map (start_idx, end_idx) runs back to timestamps/values."""
    rows = []
    for a, b in runs:
        rows.append(
            {
                "start_idx": a,
                "end_idx": b,
                "start_time": t[a],
                "end_time": t[b],
                "duration_timesteps": b - a + 1,
                "max_value": float(np.max(x[a : b + 1])),
                "min_value": float(np.min(x[a : b + 1])),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "start_idx",
            "end_idx",
            "start_time",
            "end_time",
            "duration_timesteps",
            "max_value",
            "min_value",
        ],
    )


def run_sar(
    t,
    x,
    period_seconds=None,
    smoothing_window=None,
    lam=None,
    g_max=None,
    d_min=None,
):
    """Run the full SAR pipeline on a pre-processed time series (t, x).

    Parameters not passed explicitly default to the manual values in
    main_config.py.
    """
    period_seconds = cfg.PERIOD_SECONDS if period_seconds is None else period_seconds
    W = cfg.SMOOTHING_WINDOW if smoothing_window is None else smoothing_window
    lam = cfg.LAMBDA if lam is None else lam
    g_max = cfg.G_MAX if g_max is None else g_max
    d_min = cfg.D_MIN if d_min is None else d_min

    sampling_interval = infer_sampling_interval_seconds(t)
    P = compute_period_timesteps(period_seconds, sampling_interval)

    sections = compute_sections(x, P)
    phi = compute_periodic_pattern(x, P)
    d = compute_deviation(x, phi, P)
    s = smooth_deviation(d, W)
    theta = compute_dynamic_threshold(s, lam)
    y = binarize(s, theta)

    raw_runs = extract_runs(y)
    merged_runs = merge_runs(raw_runs, g_max)
    final_runs = prune_runs(merged_runs, d_min)

    raw_runs_df = runs_to_dataframe(raw_runs, t, x)
    merged_runs_df = runs_to_dataframe(merged_runs, t, x)
    final_runs_df = runs_to_dataframe(final_runs, t, x)

    return SARResult(
        t=t,
        x=x,
        P=P,
        sampling_interval_seconds=sampling_interval,
        phi=phi,
        sections=sections,
        d=d,
        s=s,
        theta=theta,
        y=y,
        raw_runs=raw_runs,
        merged_runs=merged_runs,
        final_runs=final_runs,
        raw_runs_df=raw_runs_df,
        merged_runs_df=merged_runs_df,
        final_runs_df=final_runs_df,
    )
