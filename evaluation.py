"""
Evaluation of SAR anomaly-region candidates against ground-truth labels,
using the PRTS range-based precision/recall/F1 score metrics
(https://pypi.org/project/prts/).
"""

import glob
import os

import numpy as np
import pandas as pd
from prts import ts_fscore, ts_precision, ts_recall

import main_config as cfg


def runs_to_binary(runs_df, length):
    """Convert a range-candidates DataFrame (with start_idx/end_idx columns)
    into a binary 0/1 array of the given length."""
    y = np.zeros(length, dtype=int)
    for _, row in runs_df.iterrows():
        y[int(row["start_idx"]) : int(row["end_idx"]) + 1] = 1
    return y


def evaluate_series(true_labels, pred_runs_df):
    """Compute PRTS precision, recall, and F1 score for a single series."""
    real = np.asarray(true_labels, dtype=int)
    pred = runs_to_binary(pred_runs_df, len(real))

    # prts requires at least one positive in both arrays; SAR predicting no
    # anomalies at all (or a series with no ground-truth anomalies) is a
    # valid outcome, scored 0 by the usual precision/recall convention.
    if pred.sum() == 0 or real.sum() == 0:
        return 0.0, 0.0, 0.0

    # Precision never rewards mere existence (Tatbul et al. 2018), so it
    # always uses PRTS_PRECISION_ALPHA; only recall's alpha is meant to be
    # tuned for a domain-specific existence/quality trade-off.
    precision = ts_precision(
        real, pred, alpha=cfg.PRTS_PRECISION_ALPHA, cardinality=cfg.PRTS_CARDINALITY, bias=cfg.PRTS_BIAS
    )
    recall = ts_recall(real, pred, alpha=cfg.PRTS_RECALL_ALPHA, cardinality=cfg.PRTS_CARDINALITY, bias=cfg.PRTS_BIAS)
    fscore = ts_fscore(
        real,
        pred,
        p_alpha=cfg.PRTS_PRECISION_ALPHA,
        r_alpha=cfg.PRTS_RECALL_ALPHA,
        cardinality=cfg.PRTS_CARDINALITY,
        p_bias=cfg.PRTS_BIAS,
        r_bias=cfg.PRTS_BIAS,
    )
    return precision, recall, fscore


def evaluate_dataset(dataset_dir, range_dir, label_column):
    """Score every range-candidates file in range_dir against the matching
    ground-truth series (same filename) in dataset_dir.

    Returns a DataFrame with one row per series: series, precision, recall, f_score.
    """
    rows = []
    for range_path in sorted(glob.glob(os.path.join(range_dir, "*.csv"))):
        name = os.path.basename(range_path)
        source_path = os.path.join(dataset_dir, name)
        if not os.path.exists(source_path):
            continue

        true_labels = pd.read_csv(source_path)[label_column].to_numpy()
        pred_runs_df = pd.read_csv(range_path)
        precision, recall, fscore = evaluate_series(true_labels, pred_runs_df)
        rows.append({"series": name, "precision": precision, "recall": recall, "f_score": fscore})

    return pd.DataFrame(rows, columns=["series", "precision", "recall", "f_score"])


def compute_mean_metrics(metrics_df):
    """Mean and standard deviation of precision/recall/F1 score across all
    series, as a single-row DataFrame (*_std columns hold the std devs;
    0.0 rather than NaN when there are fewer than two series).
    """
    std = metrics_df[["precision", "recall", "f_score"]].std().fillna(0.0)
    return pd.DataFrame(
        [
            {
                "precision": metrics_df["precision"].mean(),
                "recall": metrics_df["recall"].mean(),
                "f_score": metrics_df["f_score"].mean(),
                "precision_std": std["precision"],
                "recall_std": std["recall"],
                "f_score_std": std["f_score"],
            }
        ]
    )
