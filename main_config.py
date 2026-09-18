"""
Manual (user-set) parameters for Sustained Anomaly Recognition (SAR).
"""

# INPUT: default CSV for --mode single (used when --input isn't given).
# labeled_NEK_52_data.csv is a nicely daily-periodic NEK series with two
# ground-truth anomaly ranges (17 and 87 timesteps), both correctly flagged
# by SAR at the parameters below, with no false positives.
INPUT = "data/NEK/labeled_NEK_52_data.csv"

# Column names expected in the input CSV.
TIME_COLUMN = "timestamp"
VALUE_COLUMN = "value"

# VALUE_LABEL: y-axis label for plots of the input series' value (--mode
# single). NEK's series are generic, undocumented per-series "Key
# Performance Indicator" values from production network equipment (no more
# specific unit is published) - update this when INPUT points elsewhere.
VALUE_LABEL = r"Network equipment KPI"

# T: periodicity of the input series, in seconds. The period in timesteps
# (P) is derived automatically from T and the inferred sampling interval.
PERIOD_SECONDS = 24 * 60 * 60  # 1 day

# W: width (in timesteps) of the moving-average window used to smooth the
# raw distance-to-pattern into the deviation score.
SMOOTHING_WINDOW = 3

# LAMBDA: sensitivity multiplier of the dynamic threshold
# Theta = mean(s) + LAMBDA * std(s).
LAMBDA = 1.4291222461677693

# G_MAX: maximum gap (in timesteps) between two runs for them to be merged.
G_MAX = 8

# D_MIN: minimum duration (in timesteps) for a merged run to be kept.
D_MIN = 2

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
# CWDE_SIGNALS_PATH = "data/C-WDE/cesnet_1367_n_packets_signals.csv"
CWDE_SIGNALS_PATH = None
# CWDE_SIGNAL_COLUMN = "signal_cwde_raw_avg"
CWDE_SIGNAL_COLUMN = None

# --- Wide single-column figure (--mode single-wide) ---------------------
# Same idea as INPUT/VALUE_LABEL/SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN/
# CWDE_SIGNALS_PATH/CWDE_SIGNAL_COLUMN above, but for --mode single-wide,
# whose figure spans the full IEEE double-column width (see plotter.py's
# plot_sar_result_wide()). Defaults to the CESNET example series.
WIDE_INPUT = "data/cesnet_ip1367_n_packets_hourly.csv"
WIDE_VALUE_LABEL = r"Packets per hour"
WIDE_SMOOTHING_WINDOW = 12
WIDE_LAMBDA = 3.0
WIDE_G_MAX = 5
WIDE_D_MIN = 5
WIDE_CWDE_SIGNALS_PATH = "data/C-WDE/cesnet_1367_n_packets_signals.csv"
WIDE_CWDE_SIGNAL_COLUMN = "signal_cwde_raw_avg"

# --- CESNET dataset triple (--mode cesnet) -------------------------------
# The three CESNET-TimeSeries24 IP 1367 metrics sharing the same
# timestamps (same period/traffic, three different measurements); n_packets
# is WIDE_INPUT, the series also used by --mode single-wide with its C-WDE
# overlay. There is no ground truth for any of the three, so --mode cesnet
# only counts SAR's anomalous sections per metric, rather than scoring them.
CESNET_INPUTS = {
    "n_packets": WIDE_INPUT,
    "n_flows": "data/cesnet_ip1367_n_flows_hourly.csv",
    "n_bytes": "data/cesnet_ip1367_n_bytes_hourly.csv",
}
CESNET_DATASET_NAME = "CESNET"

# --- Evaluation pipeline -----------------------------------------------
# EVAL_MODE: "single" runs SAR once on --input; "evaluate" runs SAR over
# every series in EVAL_DATASET_DIR and scores it against ground truth.
EVAL_MODE = "single"

# EVAL_DATASET_DIR / EVAL_FILE_GLOB: folder and filename pattern of the
# labelled series to evaluate, used when EVAL_MODE == "evaluate".
EVAL_DATASET_DIR = "data/NEK"
EVAL_FILE_GLOB = "labeled_NEK_*_data.csv"

# EVAL_DATASET_NAME: name of the results/[EVAL_DATASET_NAME] output folder.
EVAL_DATASET_NAME = "NEK"

# EVAL_LABEL_COLUMN: ground-truth binary anomaly label column in each file.
EVAL_LABEL_COLUMN = "label"

# EVAL_EXCLUDE_FILES: file names (matching EVAL_FILE_GLOB, not full paths)
# skipped by --mode evaluate/tune. These 8 NEK series undergo a permanent
# regime change partway through (a stable baseline, then a new pattern that
# persists for most of the rest of the series) rather than a transient
# anomaly; SAR's single global daily-median template ends up matching the
# new (majority) regime and flags the old baseline instead, collapsing
# precision/recall (F1 < 0.1) - see the investigation in the project chat.
# NEK_51/56, NEK_41/46, NEK_6/1 and NEK_36/31 are each duplicate pairs of
# the same underlying series.
# EVAL_EXCLUDE_FILES = [
#     "labeled_NEK_51_data.csv",
#     "labeled_NEK_56_data.csv",
#     "labeled_NEK_41_data.csv",
#     "labeled_NEK_46_data.csv",
#     "labeled_NEK_6_data.csv",
#     "labeled_NEK_1_data.csv",
#     "labeled_NEK_36_data.csv",
#     "labeled_NEK_31_data.csv",
# ]

EVAL_EXCLUDE_FILES = []

# PRTS_PRECISION_ALPHA: existence-reward weight for PRTS precision. Kept at
# 0 (purely overlap-based) since precision is defined to never reward mere
# existence (Tatbul et al. 2018).
PRTS_PRECISION_ALPHA = 0.0

# PRTS_RECALL_ALPHA: existence-reward weight for PRTS recall (0 = purely
# overlap-based, 1 = any overlap at all earns full credit). 0.5 gives half
# credit for detecting a real anomaly at all, half for how completely and
# accurately it was captured.
PRTS_RECALL_ALPHA = 0.5

# PRTS_CARDINALITY / PRTS_BIAS: PRTS cardinality ("one", "reciprocal",
# "udf_gamma") and positional bias ("flat", "front", "middle", "back").
# "reciprocal" penalizes a prediction that overlaps multiple ground-truth
# ranges (or vice versa) instead of rewarding it as if it were a clean match.
PRTS_CARDINALITY = "reciprocal"
PRTS_BIAS = "flat"

# --- Hyperparameter tuning (--mode tune) --------------------------------
# TUNE_N_TRIALS: number of Optuna trials. Benchmarked at ~0.1s/trial on the
# full NEK dataset, so this finishes in well under a minute.
TUNE_N_TRIALS = 500

# TUNE_*_RANGE: (low, high) search range Optuna samples SMOOTHING_WINDOW,
# LAMBDA, G_MAX and D_MIN from, replacing their main_config defaults above.
TUNE_SMOOTHING_WINDOW_RANGE = (2, 48)
TUNE_LAMBDA_RANGE = (0.5, 5.0)
TUNE_G_MAX_RANGE = (0, 20)
TUNE_D_MIN_RANGE = (1, 30)
