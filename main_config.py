"""
Manual (user-set) parameters for Sustained Anomaly Recognition (SAR).
"""

# Column names expected in the input CSV.
TIME_COLUMN = "timestamp"
VALUE_COLUMN = "value"

# T: periodicity of the input series, in seconds. The period in timesteps
# (P) is derived automatically from T and the inferred sampling interval.
PERIOD_SECONDS = 24 * 60 * 60  # 1 day

# W: width (in timesteps) of the moving-average window used to smooth the
# raw distance-to-pattern into the deviation score.
SMOOTHING_WINDOW = 12

# LAMBDA: sensitivity multiplier of the dynamic threshold
# Theta = mean(s) + LAMBDA * std(s).
LAMBDA = 3.0

# G_MAX: maximum gap (in timesteps) between two runs for them to be merged.
G_MAX = 5

# D_MIN: minimum duration (in timesteps) for a merged run to be kept.
D_MIN = 5

# SHOW_PLOT: if True, display figures in a matplotlib window; if False,
# save them to disk instead. Applies to both plots below.
SHOW_PLOT = False

# PLOT_PATTERN: if True, also produce the sections/pattern figure.
PLOT_PATTERN = True

# TARGET: 'view' for interactive/inspection-friendly figures, or 'paper' for
# camera-ready IEEE-formatted figures (LaTeX text, double-column sizing).
TARGET = "paper"

# Optional: path to a CSV of an externally computed C-WDE detector-ensemble
# signal, overlaid as points on the summary plot. Must be row-aligned with
# the main input series (same length and timestamps). Set to None to disable.
CWDE_SIGNALS_PATH = "data/C-WDE/cesnet_1367_n_packets_signals.csv"
CWDE_SIGNAL_COLUMN = "signal_cwde_raw_avg"
