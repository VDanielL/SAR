# SAR — Sustained Anomaly Recognition

SAR detects **sustained anomalous regions** in a periodic (e.g. daily) univariate time series — stretches of time where the data drifts away from its usual recurring pattern, as opposed to single-point spikes.

## How it works

1. **Periodic pattern**: the series is split by its period (e.g. one day), and for each position within the period the median across all periods gives the expected "template" value.
2. **Deviation**: each point's absolute distance from the template at its position is computed, then smoothed with a moving average to suppress noise.
3. **Thresholding**: a dynamic threshold (`mean + LAMBDA * std` of the smoothed deviation) flags points as pattern-deviation points.
4. **Run extraction**: consecutive flagged points form runs; nearby runs are merged across small gaps (`G_MAX`), and runs shorter than `D_MIN` are discarded as noise.

The surviving runs are the final sustained anomaly region candidates.

## Files

- `sar.py` — the SAR algorithm (pattern estimation, deviation, thresholding, run merging/pruning), plus `load_time_series()`/`load_binary_signal()` (CSV loading) and `generate_faux_timestamps()`/`load_time_series_faux_timestamps()` (synthesizing evenly-spaced timestamps for a series whose own timestamp column is really just a sequential row index - see `--mode evaluate-point-wise`).
- `main_config.py` — manual SAR parameters (`PERIOD_SECONDS`, `SMOOTHING_WINDOW`, `LAMBDA`, `G_MAX`, `D_MIN`) shared by every mode except `tune`/`tune-point-wise`/`tune-combined` (which search over them instead) and `evaluate-point-wise` (which uses its own per-dataset `POINTWISE_SAR_PARAMS` instead), input CSV column names, plot toggles (`SHOW_PLOT`, `PLOT_PATTERN`, `TARGET`), an optional C-WDE overlay (`CWDE_SIGNALS_PATH`, `CWDE_SIGNAL_COLUMN`), the `--mode single-wide`/`cesnet` dataset settings (`WIDE_INPUT`, `WIDE_VALUE_LABEL`, `WIDE_CWDE_SIGNALS_PATH`, `WIDE_CWDE_SIGNAL_COLUMN`, `CESNET_INPUTS`, `CESNET_DATASET_NAME`), the labelled-dataset registry `DATASETS` (dataset_dir/file_glob/label_column, plus optionally exclude_files/sampling_seconds - `YAHOO_SAMPLING_SECONDS`, `IOPS_SAMPLING_SECONDS`) that `--mode evaluate` (`EVAL_MODE`, `EVAL_DATASET_DIR`, `EVAL_DATASET_NAME`, `EVAL_FILE_GLOB`, `EVAL_LABEL_COLUMN`, `EVAL_EXCLUDE_FILES`), `--mode evaluate-point-wise` (`POINTWISE_DATASETS`, `POINTWISE_SAR_PARAMS`), and the three tuning modes all draw from, the shared PRTS settings (`PRTS_*`), and the tuning settings (`TUNE_REWARD_WEIGHT_F1`, `TUNE_REWARD_WEIGHT_COVERAGE`, `TUNE_SEED_POINTS`, `TUNE_N_TRIALS`, `TUNE_*_RANGE`).
- `main.py` — CLI entry point with eight modes (`--mode single|single-wide|cesnet|evaluate|evaluate-point-wise|tune|tune-point-wise|tune-combined`): `single` runs SAR on one CSV (`--input`) and optionally plots it; `single-wide` is the same (same `SMOOTHING_WINDOW`/`LAMBDA`/`G_MAX`/`D_MIN`) but for a full double-column-width example figure (defaults `--input` to the CESNET series via `WIDE_INPUT`); `cesnet` runs SAR (again, the same parameters) on all three CESNET metrics (`CESNET_INPUTS`) and counts anomalous sections in each, with no ground truth to score against, plus the `single-wide` n_packets plot; `evaluate` runs SAR over every series in a labelled dataset folder and scores it; `evaluate-point-wise` runs it over NAB/Yahoo/IOPS (`POINTWISE_DATASETS`), each with its own `POINTWISE_SAR_PARAMS` entry rather than the shared parameters, scores each dataset the same way `evaluate` does, and also counts signalled regions per series and per dataset (mean/std of count and duration), since their point-wise labels don't say much through a range-based score alone; `tune`/`tune-point-wise`/`tune-combined` (see `tuning.py`) are the only modes that don't use a fixed parameter set - they search for the best combination against, respectively, mean F1 over the long-anomaly datasets, mean F1 over each point-wise dataset separately, or a combined F1-vs-coverage reward over both groups.
- `tuning.py` — the three tuning modes' shared implementation: uses [Optuna](https://optuna.org/) to search SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN for the combination maximising mean PRTS F1 (`tune`: over the "long-anomaly" datasets - every `DATASETS` entry not in `POINTWISE_DATASETS`, currently just NEK; `tune-point-wise`: the same reward, but as three separate studies, one per `POINTWISE_DATASETS` entry - NAB/Yahoo/IOPS each tuned independently, not pooled, and run in parallel in their own process) or a reward combining both (`tune-combined`: mean F1 over the long-anomaly datasets against coverage fraction over the point-wise ones pooled together, weighted by `TUNE_REWARD_WEIGHT_F1`/`TUNE_REWARD_WEIGHT_COVERAGE`).
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

For a single wide (full double-column-width) example figure - e.g. to regenerate the CESNET example plot - use `--mode single-wide` instead; it uses the same `SMOOTHING_WINDOW`/`LAMBDA`/`G_MAX`/`D_MIN` as `single`, just defaulting `--input` to `main_config.py`'s `WIDE_INPUT`:

```bash
python main.py --mode single-wide --plot
```

To run SAR on all three CESNET metrics at once (n_packets, n_flows, n_bytes - `main_config.py`'s `CESNET_INPUTS`) instead of just one, use `--mode cesnet`:

```bash
python main.py --mode cesnet
```

There's no ground truth for any of the three, so this just runs SAR on each with the same `SMOOTHING_WINDOW`/`LAMBDA`/`G_MAX`/`D_MIN` as every other mode, always writes the anomalous sections themselves to `results/CESNET/range_candidates/<original file name>.csv` (same convention as `--mode evaluate`'s per-series files: no date or parameter tag, so reruns overwrite in place), and counts how many anomalous sections it found per metric, printing the result and (when `SHOW_PLOT = False`) saving it to `results/CESNET/anomaly_counts.csv` (columns: `dataset`, `n_anomalous_sections`). It then plots the n_packets series the same way `--mode single-wide` does (with its C-WDE overlay), saving to `results/CESNET/cesnet_example_plot.pdf` or displaying it, per `SHOW_PLOT`.

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

## Evaluating point-wise labelled datasets

```bash
python main.py --mode evaluate-point-wise
```

NAB, Yahoo, and IOPS (`main_config.py`'s `POINTWISE_DATASETS`) label individual anomalous points rather than NEK's sustained ranges - PRTS handles a single-timestep ground-truth anomaly fine (recall/precision/F1 all come out sensible, verified directly), so this mode scores each dataset the same way `--mode evaluate` does for NEK, into its own `results/<dataset>/` folder (`anomaly_det_metrics.csv`, `anomaly_det_metrics_mean.csv`, and the same plots, governed by `SHOW_PLOT`/`TARGET` exactly as in `--mode evaluate`). Unlike every other mode, this one doesn't use the shared `SMOOTHING_WINDOW`/`LAMBDA`/`G_MAX`/`D_MIN` - each dataset gets its own entry from `main_config.py`'s `POINTWISE_SAR_PARAMS`, since NAB/Yahoo/IOPS don't share a good parameter set with each other any better than they do with NEK (see `--mode tune-point-wise` below). On top of that, since a range-based score alone doesn't say much when the ground truth is single points rather than ranges, this mode also writes the anomalous sections themselves to `results/<dataset>/range_candidates/<original file name>.csv` (same convention as `--mode evaluate`), and counts how many regions SAR signalled - printing and (when `SHOW_PLOT = False`) saving both:

- `results/pointwise_signalled_regions.csv` — one row per series: `dataset`, `series`, `n_signalled_regions`.
- `results/NAB_Yahoo_IOPS_signalled_region_stats.csv` — per dataset (`dataset`): the total, mean, and (population) standard deviation of `n_signalled_regions` per series (`total`, `mean`, `std`), plus the mean and standard deviation of the regions' own duration in timesteps, pooled across every region in the dataset rather than averaged per series first (`mean_duration`, `std_duration`). Also printed as e.g. "NAB: 2446 signalled regions total (mean 47.04, std 39.47 per series; region duration mean 9.91, std 14.33 timesteps)".

Finally, it prints a ready-to-paste booktabs-style LaTeX table (`main.py`'s `format_latex_metrics_table()`) of each dataset's mean precision/recall/F1 score, one row per dataset.

Yahoo's and IOPS's own "timestamp" column is really just a sequential row index, not real time, so both get faux timestamps synthesized at their real sampling interval instead (`sar.generate_faux_timestamps()`, via `YAHOO_SAMPLING_SECONDS`/`IOPS_SAMPLING_SECONDS`) - Yahoo Webscope S5 A1 is hourly production traffic, and IOPS (via the TSB-UAD benchmark, originally the AIOps 2018 KPI challenge) is mostly 1-minute sampling. Left as their literal row index, `load_time_series`'s usual numeric-timestamp rule (needed for NEK's real Unix-epoch-second timestamps) would misread them as a Unix epoch, computing a period wildly out of proportion to the series length and collapsing the deviation score to zero everywhere. NAB has real timestamps already, so it's unaffected.

## Tuning SMOOTHING_WINDOW, LAMBDA, G_MAX, D_MIN

`tuning.py` uses [Optuna](https://optuna.org/) to search `SMOOTHING_WINDOW`, `LAMBDA`, `G_MAX` and `D_MIN` (search ranges: `TUNE_*_RANGE` in `main_config.py`) for the combination maximising a reward, over `TUNE_N_TRIALS` trials. There are three tuning modes, all built from the same two building blocks - mean PRTS F1 score (averaged equally per dataset, not pooled over all series, so a dataset with more series doesn't count more than one with fewer) and coverage fraction (the fraction of the total timeline SAR flags, pooled across every series in a group - naturally bounded in `[0, 1]`, unlike a raw region count or raw mean duration on their own, which pull against each other through `G_MAX` and are otherwise unbounded, verified directly in the project chat) - just scoring different dataset groups:

```bash
python main.py --mode tune               # mean F1 over the "long-anomaly" datasets (every DATASETS entry not in POINTWISE_DATASETS - currently just NEK)
python main.py --mode tune-point-wise    # the same mean-F1 reward, but as three SEPARATE studies - one per POINTWISE_DATASETS entry (NAB/Yahoo/IOPS), each tuned independently
python main.py --mode tune-combined      # both at once: mean F1 over the long-anomaly datasets weighted by TUNE_REWARD_WEIGHT_F1, against coverage fraction over the point-wise datasets (pooled together) weighted by TUNE_REWARD_WEIGHT_COVERAGE
```

`tune-point-wise` runs Optuna once per point-wise dataset rather than pooling all three under one shared parameter set, since NAB/Yahoo/IOPS don't share a good parameter set with each other any better than they do with NEK. Since the three studies don't interact at all (different dataset, different Optuna study, no shared state), they run in parallel - one OS process per dataset, via `concurrent.futures.ProcessPoolExecutor` - rather than one after another; each study loads only its own dataset's series, in its own process. Progress lines from all three studies (each Optuna trial, plus a `==== <dataset> ====` banner at the start of each) print as they arrive and can interleave, but each line still names its own dataset. It exists to see/quantify directly that F1 structurally collapses on point-wise ground truth regardless of detection quality (see the project chat's precision investigation) - it's not a meaningful tuning target on its own, unlike `tune` and `tune-combined`.

`tune` and `tune-point-wise` (not `tune-combined`) write their result straight to `sar_params.json` - `tune` overwrites its `"main"` entry (read by `main_config.py`'s `SMOOTHING_WINDOW`/`LAMBDA`/`G_MAX`/`D_MIN`) as soon as its single study finishes; `tune-point-wise` overwrites `"pointwise"[dataset]` for each dataset (read by `POINTWISE_SAR_PARAMS`) from the main process, once every dataset's worker process has returned its result (so three concurrent workers never read-modify-write the same file at once). Either way, the next process that imports `main_config` (including a subsequent tuning run) picks the new values up automatically, with no manual copy-paste step. `tune-combined`'s result isn't written anywhere automatically, since it optimises a weighted blend rather than either file section directly.

All three modes use multivariate TPE rather than Optuna's default univariate TPE, which models each parameter's effect independently and can miss a parameter whose effect is *conditional* on another - `D_MIN` has a sharp, narrow optimum that only shows up at moderate `G_MAX` and washes out at high `G_MAX` (verified directly in the project chat), so the default sampler can settle on "`G_MAX` high, `D_MIN` irrelevant" and never revisit `D_MIN` where it actually matters.

`tune` additionally seeds its study with `main_config.py`'s `TUNE_SEED_POINTS` - known-good parameter points (verified in the project chat) enqueued as the first trials before any sampler-chosen ones, so the search starts with direct evidence of `D_MIN`'s narrow optimum instead of having to stumble onto it by chance. `tune-point-wise` and `tune-combined` don't seed anything, since there's no comparably verified good starting point for the point-wise datasets.
