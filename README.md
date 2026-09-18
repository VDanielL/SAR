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
- `main_config.py` — manual parameters (`PERIOD_SECONDS`, `SMOOTHING_WINDOW`, `LAMBDA`, `G_MAX`, `D_MIN`), input CSV column names, plot toggles (`SHOW_PLOT`, `PLOT_PATTERN`, `TARGET`), and an optional C-WDE overlay (`CWDE_SIGNALS_PATH`, `CWDE_SIGNAL_COLUMN`).
- `main.py` — CLI entry point: loads a CSV, runs SAR, saves results.
- `plotter.py` — builds the summary figure and a separate pattern figure (every period-length section of the data overlaid with the pattern phi). The summary figure's layout depends on `TARGET`: `'view'` gives a 4-panel diagnostic figure (original data with flagged points, distance, smoothed score with threshold, run stages); `'paper'` gives a compact 2-axis figure (original data with flagged points, plus a thin strip showing only the final anomaly regions), sized for an IEEE double-column layout and rendered with LaTeX text. `TARGET = 'paper'` also resizes the pattern figure to the same double-column width; its content is unchanged.
- `data/` — example input data (CESNET-TimeSeries24, IP 1367, hourly `n_packets`), plus a `C-WDE/` folder holding externally computed detector-ensemble signals for the same series (row-aligned; overlaid on the summary plot when `CWDE_SIGNALS_PATH` is set), and labelled reference datasets (`NAB/`, `IOPS/`, `Yahoo/`, `NEK/`, `CESNET/`) with manifests under `dataset_manifest/`.
- `results/` — output of a run (created by `main.py`).

## Usage

```bash
pip install -r requirements.txt
python main.py --input data/cesnet_ip1367_n_packets_hourly.csv --output-dir results --plot
```

The input CSV needs a timestamp column and a value column (see `TIME_COLUMN` / `VALUE_COLUMN` in `main_config.py`). Any manual parameter can be overridden on the command line, e.g. `--lam 2.5`.

## Outputs

Each run writes to `--output-dir` with filenames of the form
`<datetime>_<what>_<parameter settings>.<ext>`, so every file is traceable to the exact settings that produced it:

- `..._point_scores_....csv` — per-timestep value, distance, smoothed deviation, threshold, and pattern-deviation label.
- `..._runs_raw_....csv`, `..._runs_merged_....csv`, `..._runs_final_....csv` — the sustained-anomaly-region candidates at each stage of the merge-then-prune procedure.
- `..._sar_plot_....pdf` — the 4-panel summary figure (only with `--plot`, and only if `SHOW_PLOT = False` in `main_config.py`; if `SHOW_PLOT = True` the figure is displayed in a matplotlib window instead of being saved).
- `..._pattern_plot_....pdf` — the pattern figure (only with `--plot` and `PLOT_PATTERN = True`; also governed by `SHOW_PLOT`).
