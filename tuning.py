"""
Hyperparameter tuning for the Sustained Anomaly Recognition (SAR) module.
Three modes, all searching SMOOTHING_WINDOW, LAMBDA, G_MAX, D_MIN with
Optuna, differing only in which datasets they score and how:

  - --mode tune: mean PRTS F1 score (averaged equally per dataset) over
    every "long-anomaly" dataset - every main_config.DATASETS entry NOT
    also in main_config.POINTWISE_DATASETS (currently just NEK), which has
    sustained-range ground truth PRTS F1 is meaningful for. Its result is
    written straight to sar_params.json's "main" entry once the study
    finishes - main_config.py's SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN load
    from there, so no manual copy-paste step is needed.
  - --mode tune-point-wise: the same mean-F1 reward, but run as three
    separate studies, one per main_config.POINTWISE_DATASETS entry
    (NAB/Yahoo/IOPS), each tuning that dataset's own SMOOTHING_WINDOW/
    LAMBDA/G_MAX/D_MIN independently rather than pooling all three under
    one shared/compromise set - they don't share a good parameter set
    with each other any better than with NEK (see the project chat's
    investigation). Their ground truth is individual points rather than
    ranges, so F1 structurally collapses regardless of detection quality
    - this mode is for seeing/quantifying that per dataset, not because a
    high score here is a meaningful tuning target. The three studies are
    fully independent (different dataset, different search), so they run
    in separate processes in parallel (see run_tune_pointwise()) rather
    than one after another. Each result is written straight to
    sar_params.json's "pointwise"[dataset] entry once every study has
    finished - main_config.py's POINTWISE_SAR_PARAMS (what --mode
    evaluate-point-wise actually uses) loads from there, so no manual
    copy-paste step is needed.
  - --mode tune-combined: both groups at once, in a single two-term
    reward - mean F1 over the long-anomaly datasets, weighted by
    main_config.TUNE_REWARD_WEIGHT_F1, against coverage fraction (the
    fraction of the total timeline SAR flags, pooled across every
    point-wise series - naturally bounded in [0, 1], unlike a raw region
    count or raw mean duration on their own, which pull against each
    other through G_MAX and are otherwise unbounded - see the project
    chat's investigation) over the point-wise datasets, weighted by
    main_config.TUNE_REWARD_WEIGHT_COVERAGE.

Each study loads its own series once up front (tune-point-wise's three
studies each load only their own dataset, in their own worker process) and
shows no figures - a study lives entirely in memory while it runs, and
only the best parameters and score are printed at the end. tune and
tune-point-wise write their result(s) to sar_params.json once finished
(see update_main_sar_params()/update_pointwise_sar_params()); tune-combined
writes nothing to disk.

All three use multivariate TPE (see _make_study()) rather than Optuna's
default univariate TPE, which models each parameter's marginal effect
independently and can miss a parameter whose effect is conditional on
another - e.g. D_MIN has a sharp, narrow optimum that only shows up at
moderate G_MAX and washes out at high G_MAX (verified directly in the
project chat), so a sampler that ignores that interaction can converge on
"G_MAX high, D_MIN irrelevant" without ever re-examining D_MIN in the
region where it actually matters.
"""

import concurrent.futures
import glob
import json
import os

import pandas as pd

import evaluation
import main_config as cfg
import sar

SAR_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sar_params.json")


def find_dataset_files(dataset_dir, file_glob, exclude_files=()):
    """Files matching file_glob in dataset_dir, minus any named in
    exclude_files (file names, not full paths).
    """
    files = sorted(glob.glob(os.path.join(dataset_dir, file_glob)))
    files = [f for f in files if os.path.basename(f) not in exclude_files]
    if not files:
        raise FileNotFoundError(f"No files matching '{file_glob}' found in {dataset_dir}")
    return files


def load_dataset_series(dataset_name, args):
    """Load every (t, x, true_labels) triple for one main_config.DATASETS
    entry, using faux timestamps when the dataset defines "sampling_seconds"
    (see main_config.py's comment on DATASETS).
    """
    dataset_cfg = cfg.DATASETS[dataset_name]
    files = find_dataset_files(
        dataset_cfg["dataset_dir"], dataset_cfg["file_glob"], dataset_cfg.get("exclude_files", ())
    )
    sampling_seconds = dataset_cfg.get("sampling_seconds")

    series = []
    for file_path in files:
        if sampling_seconds is not None:
            t, x = sar.load_time_series_faux_timestamps(file_path, args.value_column, sampling_seconds)
        else:
            t, x = sar.load_time_series(file_path, args.time_column, args.value_column)
        true_labels = pd.read_csv(file_path)[dataset_cfg["label_column"]].to_numpy()
        series.append((t, x, true_labels))
    return series


def load_dataset_groups(dataset_names, args):
    """{name: [(t, x, true_labels), ...]} for every name in dataset_names,
    printing how many series each one loaded.
    """
    datasets = {name: load_dataset_series(name, args) for name in dataset_names}
    for name, series in datasets.items():
        print(f"Loaded {len(series)} series from {name} ({cfg.DATASETS[name]['dataset_dir']})")
    return datasets


def _load_sar_params_file():
    with open(SAR_PARAMS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_sar_params_file(data):
    with open(SAR_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def update_main_sar_params(best_params):
    """Overwrite sar_params.json's "main" entry (read by
    main_config.SMOOTHING_WINDOW/LAMBDA/G_MAX/D_MIN) with best_params (a
    dict with those four keys - e.g. --mode tune's study.best_params).
    """
    data = _load_sar_params_file()
    data["main"] = {
        "smoothing_window": best_params["smoothing_window"],
        "lam": best_params["lam"],
        "g_max": best_params["g_max"],
        "d_min": best_params["d_min"],
    }
    _save_sar_params_file(data)
    print(f"Wrote main SAR parameters to {SAR_PARAMS_PATH}: {data['main']}")


def update_pointwise_sar_params(dataset_name, best_params):
    """Overwrite sar_params.json's "pointwise"[dataset_name] entry (read
    by main_config.POINTWISE_SAR_PARAMS) with best_params (a dict with
    smoothing_window/lam/g_max/d_min keys - e.g. --mode tune-point-wise's
    study.best_params for that dataset).
    """
    data = _load_sar_params_file()
    data["pointwise"][dataset_name] = {
        "smoothing_window": best_params["smoothing_window"],
        "lam": best_params["lam"],
        "g_max": best_params["g_max"],
        "d_min": best_params["d_min"],
    }
    _save_sar_params_file(data)
    print(
        f"Wrote pointwise SAR parameters for {dataset_name!r} to "
        f"{SAR_PARAMS_PATH}: {data['pointwise'][dataset_name]}"
    )


def mean_f1_over_datasets(datasets, period_seconds, smoothing_window, lam, g_max, d_min):
    """Mean PRTS F1 per dataset, averaged equally across datasets (not
    pooled over all series - a dataset with more series doesn't count more
    than one with fewer). datasets: {name: [(t, x, true_labels), ...]}.
    """
    dataset_means = []
    for series in datasets.values():
        f_scores = []
        for t, x, true_labels in series:
            result = sar.run_sar(
                t, x, period_seconds=period_seconds, smoothing_window=smoothing_window, lam=lam, g_max=g_max, d_min=d_min
            )
            _, _, fscore = evaluation.evaluate_series(true_labels, result.final_runs_df)
            f_scores.append(fscore)
        dataset_means.append(sum(f_scores) / len(f_scores))
    return sum(dataset_means) / len(dataset_means)


def pointwise_coverage_over_datasets(datasets, period_seconds, smoothing_window, lam, g_max, d_min):
    """Coverage fraction (0-1): total signalled-region duration divided by
    total series length, pooled across every series in every dataset - "in
    total", not per-dataset or per-series. Naturally bounded in [0, 1]
    (never more than "the whole timeline"), unlike a raw region count or
    raw mean duration on their own, which have no upper bound - see the
    project chat's investigation - so it needs no baseline normalisation
    to sit on the same scale as F1. datasets: {name: [(t, x, true_labels),
    ...]}.

    Also returns the raw region count and mean duration - not used in the
    reward itself, but handy diagnostics for what a given coverage
    actually looks like (many short regions vs. a few long ones).

    Returns (coverage, n_regions, mean_duration).
    """
    n_regions = 0
    total_duration = 0
    total_length = 0
    for series in datasets.values():
        for t, x, _ in series:
            result = sar.run_sar(
                t, x, period_seconds=period_seconds, smoothing_window=smoothing_window, lam=lam, g_max=g_max, d_min=d_min
            )
            n_regions += len(result.final_runs)
            total_duration += sum(b - a + 1 for a, b in result.final_runs)
            total_length += len(x)
    coverage = total_duration / total_length if total_length > 0 else 0.0
    mean_duration = total_duration / n_regions if n_regions > 0 else 0.0
    return coverage, n_regions, mean_duration


def _search_ranges():
    return (
        cfg.TUNE_SMOOTHING_WINDOW_RANGE,
        cfg.TUNE_LAMBDA_RANGE,
        cfg.TUNE_G_MAX_RANGE,
        cfg.TUNE_D_MIN_RANGE,
    )


def _print_best_params(study):
    print("Best parameters:")
    for name, value in study.best_params.items():
        print(f"  {name} = {value}")


def _make_study():
    """A maximising study using multivariate TPE, which models parameter
    interactions jointly rather than each parameter's marginal effect
    independently (Optuna's TPESampler default). D_MIN's effect is sharply
    conditional on G_MAX - its narrow, isolated optimum only shows up at
    moderate G_MAX, and washes out at high G_MAX where merged runs are
    already long enough that pruning barely bites regardless of D_MIN (see
    the project chat's investigation) - so the default univariate sampler
    can settle on "G_MAX high, D_MIN irrelevant" without ever revisiting
    D_MIN at the G_MAX region where it actually matters.
    """
    import optuna

    return optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(multivariate=True))


def run_tune_f1_only(args, dataset_names, group_label, seed_points=None):
    """Shared engine for --mode tune and --mode tune-point-wise: pure mean
    F1 score (mean_f1_over_datasets) over dataset_names, no coverage term.
    group_label is only used in the printed progress/summary text.

    seed_points, if given, is a list of {"smoothing_window", "lam",
    "g_max", "d_min"} dicts enqueued as the study's first trials (via
    study.enqueue_trial()) before any sampler-chosen ones - see
    main_config.TUNE_SEED_POINTS.
    """
    period_seconds = cfg.PERIOD_SECONDS if args.period_seconds is None else args.period_seconds
    datasets = load_dataset_groups(dataset_names, args)

    (sw_low, sw_high), (lam_low, lam_high), (gmax_low, gmax_high), (dmin_low, dmin_high) = _search_ranges()

    def objective(trial):
        smoothing_window = trial.suggest_int("smoothing_window", sw_low, sw_high)
        lam = trial.suggest_float("lam", lam_low, lam_high)
        g_max = trial.suggest_int("g_max", gmax_low, gmax_high)
        d_min = trial.suggest_int("d_min", dmin_low, dmin_high)
        return mean_f1_over_datasets(datasets, period_seconds, smoothing_window, lam, g_max, d_min)

    n_total_series = sum(len(series) for series in datasets.values())
    print(
        f"\nTuning on {n_total_series} series across {len(datasets)} {group_label} dataset(s) "
        f"({', '.join(dataset_names)}), {cfg.TUNE_N_TRIALS} trials..."
    )
    print(f"  maximise mean F1 over: {', '.join(dataset_names)}")

    study = _make_study()
    for params in seed_points or []:
        study.enqueue_trial(params)
        print(f"  seeded with {params}")
    study.optimize(objective, n_trials=cfg.TUNE_N_TRIALS)

    print()
    print(f"Best mean F1 score: {study.best_value:.4f}")
    _print_best_params(study)
    return study.best_params


def run_tune(args):
    """--mode tune: pure mean F1 over the long-anomaly datasets (every
    main_config.DATASETS entry NOT in POINTWISE_DATASETS - currently just
    NEK). Seeded with main_config.TUNE_SEED_POINTS. Writes the result to
    sar_params.json's "main" entry (read by main_config.SMOOTHING_WINDOW/
    LAMBDA/G_MAX/D_MIN).
    """
    long_anomaly_names = [name for name in cfg.DATASETS if name not in cfg.POINTWISE_DATASETS]
    best_params = run_tune_f1_only(args, long_anomaly_names, "long-anomaly", seed_points=cfg.TUNE_SEED_POINTS)
    update_main_sar_params(best_params)


def _run_single_pointwise_study(dataset_name, args):
    """Run in a worker process by run_tune_pointwise(): one dataset's whole
    tune-point-wise study, returning (dataset_name, best_params) rather than
    writing to sar_params.json itself - all three worker processes would
    otherwise read-modify-write the same file concurrently and could lose
    each other's updates, so the actual writes happen back in the parent
    process, once every worker has finished.
    """
    print(f"\n{'=' * 20} {dataset_name} {'=' * 20}")
    best_params = run_tune_f1_only(args, [dataset_name], "point-wise")
    return dataset_name, best_params


def run_tune_pointwise(args):
    """--mode tune-point-wise: runs Optuna separately for EACH dataset in
    main_config.POINTWISE_DATASETS (NAB, Yahoo, IOPS) - one independent
    study per dataset, each maximising just that dataset's own mean F1,
    rather than pooling all three under one shared/compromise parameter
    set (a dataset with more series doesn't drag down or dilute another's
    optimum this way; NAB/Yahoo/IOPS don't share a good parameter set with
    each other any better than they do with NEK - see the project chat's
    investigation). F1 structurally collapses on point-wise ground truth
    regardless of detection quality, so this is for seeing/quantifying
    that per dataset, not because a high score here is a meaningful
    tuning target on its own.

    The three studies don't interact at all (different dataset, different
    Optuna study, no shared state), so they run in parallel, one OS process
    per dataset (ProcessPoolExecutor - real parallelism, unlike threads,
    since each trial's numpy-heavy sar.run_sar() calls are CPU-bound and
    would otherwise contend for the GIL) rather than one after another.
    Progress lines from the three studies (each Optuna trial, plus the
    "====" banner above) print as they arrive and can interleave; each
    line still names its own dataset. Once every study has finished, each
    result is written to sar_params.json's "pointwise"[dataset_name] entry
    (read by main_config.POINTWISE_SAR_PARAMS) from the parent process.
    """
    dataset_names = list(cfg.POINTWISE_DATASETS)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(dataset_names)) as executor:
        futures = [executor.submit(_run_single_pointwise_study, name, args) for name in dataset_names]
        for future in concurrent.futures.as_completed(futures):
            dataset_name, best_params = future.result()
            update_pointwise_sar_params(dataset_name, best_params)


def run_tune_combined(args):
    """--mode tune-combined: the two-term reward - mean F1 over the
    long-anomaly datasets weighted by TUNE_REWARD_WEIGHT_F1, against
    coverage fraction over POINTWISE_DATASETS weighted by
    TUNE_REWARD_WEIGHT_COVERAGE (renormalised to sum to 1 in the
    unlikely case one of DATASETS/POINTWISE_DATASETS is ever empty).
    """
    period_seconds = cfg.PERIOD_SECONDS if args.period_seconds is None else args.period_seconds

    long_anomaly_names = [name for name in cfg.DATASETS if name not in cfg.POINTWISE_DATASETS]
    pointwise_names = list(cfg.POINTWISE_DATASETS.keys())

    long_anomaly_series = load_dataset_groups(long_anomaly_names, args)
    pointwise_series = load_dataset_groups(pointwise_names, args)

    (sw_low, sw_high), (lam_low, lam_high), (gmax_low, gmax_high), (dmin_low, dmin_high) = _search_ranges()

    raw_weights = {}
    if long_anomaly_series:
        raw_weights["f1"] = cfg.TUNE_REWARD_WEIGHT_F1
    if pointwise_series:
        raw_weights["coverage"] = cfg.TUNE_REWARD_WEIGHT_COVERAGE
    total_weight = sum(raw_weights.values())
    weights = {name: w / total_weight for name, w in raw_weights.items()}
    print(f"Reward weights (renormalised): {weights}")

    def objective(trial):
        smoothing_window = trial.suggest_int("smoothing_window", sw_low, sw_high)
        lam = trial.suggest_float("lam", lam_low, lam_high)
        g_max = trial.suggest_int("g_max", gmax_low, gmax_high)
        d_min = trial.suggest_int("d_min", dmin_low, dmin_high)

        f1_term = None
        if long_anomaly_series:
            f1_term = mean_f1_over_datasets(long_anomaly_series, period_seconds, smoothing_window, lam, g_max, d_min)
            trial.set_user_attr("mean_f1", f1_term)

        coverage_term = None
        if pointwise_series:
            coverage_term, n_regions, mean_duration = pointwise_coverage_over_datasets(
                pointwise_series, period_seconds, smoothing_window, lam, g_max, d_min
            )
            trial.set_user_attr("coverage", coverage_term)
            trial.set_user_attr("total_signals", n_regions)
            trial.set_user_attr("mean_duration", mean_duration)

        reward = 0.0
        if f1_term is not None:
            reward += weights["f1"] * f1_term
        if pointwise_series:
            reward -= weights["coverage"] * coverage_term
        return reward

    n_total_series = sum(len(series) for series in long_anomaly_series.values()) + sum(
        len(series) for series in pointwise_series.values()
    )
    print(
        f"\nTuning on {n_total_series} series across "
        f"{len(long_anomaly_series) + len(pointwise_series)} dataset(s), {cfg.TUNE_N_TRIALS} trials..."
    )
    if long_anomaly_names:
        print(f"  maximise mean F1 over: {', '.join(long_anomaly_names)}")
    if pointwise_names:
        print(f"  minimise coverage fraction over: {', '.join(pointwise_names)}")

    study = _make_study()
    study.optimize(objective, n_trials=cfg.TUNE_N_TRIALS)

    print()
    print(f"Best combined reward: {study.best_value:.4f}")
    best_attrs = study.best_trial.user_attrs
    if "mean_f1" in best_attrs:
        print(f"  mean F1 (long-anomaly datasets): {best_attrs['mean_f1']:.4f}")
    if "coverage" in best_attrs:
        print(
            f"  coverage fraction (point-wise datasets): {best_attrs['coverage']:.4%} "
            f"({best_attrs['total_signals']} signalled regions, "
            f"{best_attrs['mean_duration']:.2f} mean duration in timesteps)"
        )
    _print_best_params(study)
