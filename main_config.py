"""
Manual (user-set) parameters for Sustained Anomaly Recognition (SAR).
"""

import json
import os

# SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN/POINTWISE_SAR_PARAMS live in
# sar_params.json rather than being hardcoded here, so --mode tune and
# --mode tune-point-wise (tuning.py) can update them directly by writing
# that file, without editing this one (previously they were plain
# assignments here, hand-copied in after each tuning run).
with open(os.path.join(os.path.dirname(__file__), "sar_params.json"), encoding="utf-8") as _sar_params_file:
    _sar_params = json.load(_sar_params_file)

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
SMOOTHING_WINDOW = _sar_params["main"]["smoothing_window"]

# LAMBDA: sensitivity multiplier of the dynamic threshold
# Theta = mean(s) + LAMBDA * std(s).
LAMBDA = _sar_params["main"]["lam"]

# G_MAX: maximum gap (in timesteps) between two runs for them to be merged.
G_MAX = _sar_params["main"]["g_max"]

# D_MIN: minimum duration (in timesteps) for a merged run to be kept.
D_MIN = _sar_params["main"]["d_min"]

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
# Same idea as INPUT/VALUE_LABEL/CWDE_SIGNALS_PATH/CWDE_SIGNAL_COLUMN
# above, but for --mode single-wide, whose figure spans the full IEEE
# double-column width (see plotter.py's plot_sar_result_wide()). Defaults
# to the CESNET example series. The SAR parameters themselves
# (SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN above) are shared with every other
# mode except tune/tune-point-wise/tune-combined - single-wide (and
# --mode cesnet) used to have their own separate WIDE_SMOOTHING_WINDOW/
# WIDE_LAMBDA/WIDE_G_MAX/WIDE_D_MIN, but no longer do.
WIDE_INPUT = "data/cesnet_ip1367_n_packets_hourly.csv"
WIDE_VALUE_LABEL = r"Packets per hour"
WIDE_CWDE_SIGNALS_PATH = "data/C-WDE/new_params/cesnet_1367_n_packets_cwde.csv"
WIDE_CWDE_SIGNAL_COLUMN = "cwde_alert"

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

# --- Dataset registry ----------------------------------------------------
# Every named labelled dataset --mode evaluate, evaluate-point-wise, and
# tune can draw from. Each entry: dataset_dir, file_glob, label_column, and
# optionally exclude_files (file names to skip) and sampling_seconds.
#
# Yahoo's and IOPS's own "timestamp" column is just a sequential row index,
# not real wall-clock time, so SAR needs faux timestamps synthesized at
# their real sampling interval instead (sar.generate_faux_timestamps(),
# via "sampling_seconds" below) - otherwise the row index gets misread as a
# Unix epoch (load_time_series's rule for numeric timestamps, needed for
# NEK), producing a period wildly out of proportion to the series length
# and a degenerate all-zero deviation score. NAB and NEK have real
# timestamps already, so they have no "sampling_seconds" entry.
# YAHOO_SAMPLING_SECONDS: Yahoo Webscope S5's A1 subset is hourly
# production-traffic measurements (real data, not synthetic).
YAHOO_SAMPLING_SECONDS = 60 * 60
# IOPS_SAMPLING_SECONDS: IOPS here comes via the TSB-UAD benchmark, which
# repackages the original AIOps 2018 KPI competition data - "most KPI
# curves have an interval of 1 minute...while some of them have an
# interval of 5 minutes" (no per-series metadata here to tell which, so
# 1 minute, the majority case, is used for all).
IOPS_SAMPLING_SECONDS = 60

DATASETS = {
    "NEK": {
        "dataset_dir": EVAL_DATASET_DIR,
        "file_glob": EVAL_FILE_GLOB,
        "label_column": EVAL_LABEL_COLUMN,
        "exclude_files": EVAL_EXCLUDE_FILES,
    },
    "NAB": {"dataset_dir": "data/NAB", "file_glob": "labeled_*.csv", "label_column": "label"},
    "Yahoo": {
        "dataset_dir": "data/Yahoo",
        "file_glob": "labeled_Yahoo_*_data.csv",
        "label_column": "label",
        "sampling_seconds": YAHOO_SAMPLING_SECONDS,
    },
    "IOPS": {
        "dataset_dir": "data/IOPS",
        "file_glob": "labeled_IOPS_*.csv",
        "label_column": "label",
        "sampling_seconds": IOPS_SAMPLING_SECONDS,
    },
}

# POINTWISE_DATASETS: which of the DATASETS above --mode evaluate-point-wise
# runs over. NAB/Yahoo/IOPS label individual anomalous points rather than
# NEK's sustained ranges, so on top of scoring them the same way as
# --mode evaluate, that mode also counts how many regions SAR signals per
# series and per dataset (see run_evaluate_pointwise() in main.py).
POINTWISE_DATASETS = {name: DATASETS[name] for name in ["NAB", "Yahoo", "IOPS"]}

# POINTWISE_SAR_PARAMS: each point-wise dataset's own SAR parameters
# (smoothing_window, lam, g_max, d_min), used by --mode evaluate-point-wise
# instead of the shared SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN above. Unlike
# every other mode, evaluate-point-wise doesn't use one shared parameter
# set: NAB/Yahoo/IOPS don't share a good parameter set with each other any
# better than they do with NEK (see the project chat's investigation), so
# --mode tune-point-wise tunes each of the three separately (see
# tuning.py) and writes its results straight into sar_params.json's
# "pointwise" entries.
POINTWISE_SAR_PARAMS = _sar_params["pointwise"]

# --- Hyperparameter tuning --------------------------------------------
# Three modes, all searching SMOOTHING_WINDOW, LAMBDA, G_MAX, D_MIN with
# Optuna (see tuning.py's module docstring for the full detail):
#  - --mode tune: mean F1 score over the "long-anomaly" datasets - every
#    DATASETS entry NOT in POINTWISE_DATASETS below (currently just NEK),
#    which have sustained-range ground truth PRTS F1 is meaningful for
#    (averaged equally per dataset, not pooled over all series - a
#    dataset with more series doesn't count more than one with fewer).
#  - --mode tune-point-wise: the same mean-F1 reward, but over
#    POINTWISE_DATASETS (NAB/Yahoo/IOPS) instead - their ground truth is
#    individual points, so F1 structurally collapses here regardless of
#    detection quality (see the project chat's investigation); this mode
#    exists to see/quantify that directly, not as a real tuning target.
#  - --mode tune-combined: both at once, in a single two-term reward -
#    mean F1 over the long-anomaly datasets against coverage fraction
#    (the fraction of the total timeline SAR flags, pooled across every
#    point-wise series) over the point-wise datasets. A raw region count
#    or raw mean duration alone are unbounded (can run into the
#    thousands, or many times a sensible reference, for parameter choices
#    well within TUNE_*_RANGE - verified directly in the project chat)
#    and pull against each other through G_MAX (merging trades count for
#    duration), which let this reward collapse to signalling nothing at
#    all rather than balancing the two; coverage fraction folds both into
#    one number that's naturally bounded in [0, 1] - it can never exceed
#    "the whole timeline" - so it needs no baseline normalisation to sit
#    on the same scale as F1.
# TUNE_REWARD_WEIGHT_F1/COVERAGE: relative weights for tune-combined's two
# reward terms, renormalised by tuning.py to sum to 1 in the unlikely case
# one of DATASETS/POINTWISE_DATASETS is ever empty.
TUNE_REWARD_WEIGHT_F1 = 0.55
TUNE_REWARD_WEIGHT_COVERAGE = 0.45

# TUNE_SEED_POINTS: known-good (smoothing_window, lam, g_max, d_min) points
# to seed --mode tune's Optuna study with (via study.enqueue_trial(), one
# real trial each, evaluated before any sampler-chosen ones) - not
# --mode tune-point-wise or --mode tune-combined, since we have no
# comparably verified good starting points for the point-wise datasets.
# D_MIN has a sharp, narrow optimum only visible at moderate G_MAX and
# washed out at high G_MAX (verified directly in the project chat), so
# even multivariate TPE benefits from starting with direct evidence of it
# rather than having to find it by chance. All four points score well on
# NEK (mean F1 0.5418, 0.5584, 0.5544, and 0.5596 respectively, as of the
# project chat's investigation - re-verify if NEK's data or SAR itself
# changes).
TUNE_SEED_POINTS = [
    {"smoothing_window": 3, "lam": 1.429, "g_max": 8, "d_min": 2},
    {"smoothing_window": 3, "lam": 1.429, "g_max": 35, "d_min": 2},
    {"smoothing_window": 2, "lam": 1.442, "g_max": 50, "d_min": 2},
    {"smoothing_window": 2, "lam": 1.455, "g_max": 45, "d_min": 2},
]

# TUNE_N_TRIALS: number of Optuna trials. Benchmarked at ~0.1s/trial per
# dataset on NEK, so this finishes in well under a minute per dataset
# involved in the chosen tuning mode.
TUNE_N_TRIALS = 10000

# TUNE_*_RANGE: (low, high) search range Optuna samples SMOOTHING_WINDOW,
# LAMBDA, G_MAX and D_MIN from, replacing their main_config defaults above.
TUNE_SMOOTHING_WINDOW_RANGE = (1, 50)
TUNE_LAMBDA_RANGE = (0.5, 5.0)
TUNE_G_MAX_RANGE = (1, 100)
TUNE_D_MIN_RANGE = (1, 10)
