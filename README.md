# SAR — Sustained Anomaly Recognition

SAR detects **sustained anomalous regions** in a periodic (e.g. daily) univariate time series — stretches of time where the data drifts away from its usual recurring pattern, as opposed to single-point spikes.

## How it works

1. **Periodic pattern**: the series is split by its period (e.g. one day), and for each position within the period the median across all periods gives the expected "template" value.
2. **Deviation**: each point's absolute distance from the template at its position is computed, then smoothed with a moving average to suppress noise.
3. **Thresholding**: a dynamic threshold (`mean + LAMBDA * std` of the smoothed deviation) flags points as pattern-deviation points.
4. **Run extraction**: consecutive flagged points form runs; nearby runs are merged across small gaps (`G_MAX`), and runs shorter than `D_MIN` are discarded as noise.

The surviving runs are the final sustained anomaly region candidates.

## Files

- `sar.py` — the SAR algorithm (pattern estimation, deviation, thresholding, run merging/pruning).
- `main_config.py` — manual SAR parameters (`PERIOD_SECONDS`, `SMOOTHING_WINDOW`, `LAMBDA`, `G_MAX`, `D_MIN`), input CSV column names, plot toggles (`SHOW_PLOT`, `PLOT_PATTERN`, `TARGET`), an optional C-WDE overlay (`CWDE_SIGNALS_PATH`, `CWDE_SIGNAL_COLUMN`), the `--mode single-wide` parameter set (`WIDE_INPUT`, `WIDE_VALUE_LABEL`, `WIDE_SMOOTHING_WINDOW`, `WIDE_LAMBDA`, `WIDE_G_MAX`, `WIDE_D_MIN`, `WIDE_CWDE_SIGNALS_PATH`, `WIDE_CWDE_SIGNAL_COLUMN`), the `--mode cesnet` dataset triple (`CESNET_INPUTS`, `CESNET_DATASET_NAME`), the evaluation-pipeline settings (`EVAL_MODE`, `EVAL_DATASET_DIR`, `EVAL_DATASET_NAME`, `EVAL_FILE_GLOB`, `EVAL_LABEL_COLUMN`, `EVAL_EXCLUDE_FILES`, `PRTS_*`), and the tuning settings (`TUNE_N_TRIALS`, `TUNE_*_RANGE`).
- `main.py` — CLI entry point with five modes (`--mode single|single-wide|cesnet|evaluate|tune`): `single` runs SAR on one CSV (`--input`) and optionally plots it; `single-wide` is the same but for a full double-column-width example figure (defaults to the CESNET series via the `WIDE_*` settings); `cesnet` runs SAR on all three CESNET metrics (`CESNET_INPUTS`) and counts anomalous sections in each, with no ground truth to score against, plus the `single-wide` n_packets plot; `evaluate` runs SAR over every series in a labelled dataset folder and scores it; `tune` uses Optuna to search for the SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN combination with the best mean F1 score.
- `evaluation.py` — scores a dataset's anomaly-region candidates against ground-truth labels using the [PRTS](https://pypi.org/project/prts/) range-based precision/recall/F1 score, per series and averaged; contains no plotting.
- `status_bar.py` — a terminal progress bar, used while `--mode evaluate` runs SAR over a whole dataset.
- `plotter.py` — every figure the project produces. `plot_sar_result()`'s layout depends on `TARGET`: `'view'` gives a 4-panel diagnostic figure (original data with flagged points, distance, smoothed score with threshold, run stages); `'paper'` gives a compact 2-axis figure (original data with flagged points, plus a thin strip showing only the final anomaly regions), sized for a single-column-like width and rendered with LaTeX text. `plot_sar_result_wide()` is the same 2-axis 'paper' layout at the same height, but spanning the full IEEE double-column width instead - used by `--mode single-wide` (only when `TARGET = 'paper'`; with `TARGET = 'view'` that mode's figure is identical to `single`'s). `plot_pattern_sections()` overlays every period-length section of the data with the pattern phi; `TARGET = 'paper'` only resizes it to the double-column width, content unchanged. `plot_metrics_per_series()` and `plot_mean_metrics()` chart `evaluation.py`'s output (see below); with `TARGET = 'paper'` only `plot_mean_metrics()` (resized for the double-column layout) is used.
- `data/` — example input data (CESNET-TimeSeries24, IP 1367, hourly `n_packets`, `n_flows`, and `n_bytes` - the same traffic, three measures, sharing the same timestamps), plus a `C-WDE/` folder holding externally computed detector-ensemble signals for the `n_packets` series (row-aligned; overlaid on the summary plot when `CWDE_SIGNALS_PATH`/`WIDE_CWDE_SIGNALS_PATH` is set), and labelled reference datasets (`NAB/`, `IOPS/`, `Yahoo/`, `NEK/`, `CESNET/`) with manifests under `dataset_manifest/`.
- `results/` — output of a run (created by `main.py`).

## Usage

```bash
pip install -r requirements.txt
python main.py --input data/cesnet_ip1367_n_packets_hourly.csv --output-dir results --plot
```

The input CSV needs a timestamp column and a value column (see `TIME_COLUMN` / `VALUE_COLUMN` in `main_config.py`). Any manual parameter can be overridden on the command line, e.g. `--lam 2.5`.

For a single wide (full double-column-width) example figure - e.g. to regenerate the CESNET example plot - use `--mode single-wide` instead; it defaults `--input` and the SAR parameters to `main_config.py`'s `WIDE_*` settings rather than the plain ones:

```bash
python main.py --mode single-wide --plot
```

To run SAR on all three CESNET metrics at once (n_packets, n_flows, n_bytes - `main_config.py`'s `CESNET_INPUTS`) instead of just one, use `--mode cesnet`:

```bash
python main.py --mode cesnet
```

There's no ground truth for any of the three, so this just runs SAR on each with the `WIDE_*` parameters, always writes the anomalous sections themselves to `results/CESNET/range_candidates/<original file name>.csv` (same convention as `--mode evaluate`'s per-series files: no date or parameter tag, so reruns overwrite in place), and counts how many anomalous sections it found per metric, printing the result and (when `SHOW_PLOT = False`) saving it to `results/CESNET/anomaly_counts.csv` (columns: `dataset`, `n_anomalous_sections`). It then plots the n_packets series the same way `--mode single-wide` does (with its C-WDE overlay), saving to `results/CESNET/cesnet_example_plot.pdf` or displaying it, per `SHOW_PLOT`.

## Outputs

Each run writes to `--output-dir` with filenames of the form
`<datetime>_<what>_<parameter settings>.<ext>`, so every file is traceable to the exact settings that produced it:

- `..._point_scores_....csv` — per-timestep value, distance, smoothed deviation, threshold, and pattern-deviation label.
- `..._runs_raw_....csv`, `..._runs_merged_....csv`, `..._runs_final_....csv` — the sustained-anomaly-region candidates at each stage of the merge-then-prune procedure.
- `..._sar_plot_....pdf` — the 4-panel summary figure (only with `--plot`, and only if `SHOW_PLOT = False` in `main_config.py`; if `SHOW_PLOT = True` the figure is displayed in a matplotlib window instead of being saved).
- `..._pattern_plot_....pdf` — the pattern figure (only with `--plot` and `PLOT_PATTERN = True`; also governed by `SHOW_PLOT`).

## Evaluating on a labelled dataset

```bash
python main.py --mode evaluate
```

Runs SAR over every file matching `EVAL_FILE_GLOB` in `EVAL_DATASET_DIR` (default: `data/NEK/labeled_NEK_*_data.csv`), except any file named in `EVAL_EXCLUDE_FILES` (defaults to 8 NEK series that undergo a permanent regime change partway through rather than a transient anomaly, which collapses SAR's single global daily template - see the comment in `main_config.py`), showing progress with `status_bar.py`, and always writes:

- `results/<EVAL_DATASET_NAME>/range_candidates/<original file name>.csv` — one file per series, same columns as `..._runs_final_....csv` above. Named after the source file with no date or parameter tag, so reruns overwrite in place.

The metrics behave like `single` mode's figures — governed by `SHOW_PLOT` (save when `False`, display only when `True`) and by `TARGET`:

- `results/<EVAL_DATASET_NAME>/anomaly_det_metrics.csv` — one row per series: `series`, `precision`, `recall`, `f_score`, computed by `evaluation.py` against the `EVAL_LABEL_COLUMN` ground-truth column using PRTS range-based metrics.
- `results/<EVAL_DATASET_NAME>/anomaly_det_metrics_mean.csv` — the same three metrics averaged over all series.
- `results/<EVAL_DATASET_NAME>/anomaly_det_metrics_per_series.pdf` — bar chart of precision/recall/F1 score per series (`TARGET = 'view'` only).
- `results/<EVAL_DATASET_NAME>/anomaly_det_metrics_mean.pdf` — bar chart of the three averaged metrics (both targets; the only plot produced when `TARGET = 'paper'`, sized for an IEEE double-column figure).

Every `--eval-*` setting and the SAR parameters can be overridden on the command line, same as in `single` mode.

## Tuning SMOOTHING_WINDOW, LAMBDA, G_MAX, D_MIN

```bash
python main.py --mode tune
```

Uses [Optuna](https://optuna.org/) to search for the combination of `SMOOTHING_WINDOW`, `LAMBDA`, `G_MAX` and `D_MIN` (search ranges: `TUNE_*_RANGE` in `main_config.py`) that maximises the mean PRTS F1 score over `--eval-dataset-dir`, over `TUNE_N_TRIALS` trials. Every series is loaded once up front; nothing is written to disk and no figures are shown during or after tuning — the study lives entirely in memory, and only the best parameters and F1 score are printed at the end. Optuna's own per-trial log line (`Trial N finished with value: ...`) is the only progress output.
