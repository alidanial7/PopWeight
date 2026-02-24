"""PopWeight pipeline CLI orchestration."""

import argparse
import sys

import pandas as pd

import popweight.config as config
from popweight.baseline import run_baseline_evaluation
from popweight.cleaning import clean_core_columns
from popweight.diagnostics import run_diagnostics, save_diagnostics_report
from popweight.evaluation import classification_metrics, evaluate_regression
from popweight.features import add_segment_key, add_transforms
from popweight.io_excel import load_working_file
from popweight.models import predict_trending, train_trending_classifier
from popweight.reporting import (
    aggregate_classification_metrics,
    aggregate_regression_metrics,
    aggregate_weights,
    save_reports,
)
from popweight.schema import normalize_columns, validate_required_columns
from popweight.scoring import apply_scores
from popweight.splits import make_splits
from popweight.storage import init_db, read_df, write_df
from popweight.trending import apply_trending_label, compute_segment_thresholds
from popweight.weights import fit_segment_weights


def _ensure_init() -> None:
    """Ensure DB is initialized."""
    init_db(config.SQLITE_PATH)


def cmd_load() -> pd.DataFrame:
    """Load Excel and validate schema."""
    df = load_working_file(config.DATA_PATH, config.SHEET_NAME)
    df = normalize_columns(df)
    validate_required_columns(df)
    _ensure_init()
    write_df("raw_working_file", df)
    print("load: ok")
    return df


def cmd_clean() -> pd.DataFrame:
    """Clean core columns."""
    _ensure_init()
    df = read_df("raw_working_file")
    df_clean, report = clean_core_columns(df, config)
    write_df("clean_data", df_clean)
    print("clean: ok", report)
    return df_clean


def cmd_features() -> pd.DataFrame:
    """Add transforms and segment key."""
    _ensure_init()
    df = read_df("clean_data")
    df = add_transforms(df)
    df = add_segment_key(df, keys=config.SEGMENT_KEYS)
    write_df("features_data", df)
    print("features: ok")
    return df


def cmd_split() -> list:
    """Build repeated splits."""
    _ensure_init()
    df = read_df("features_data")
    splits = make_splits(df, config.RANDOM_SEEDS, config.TRAIN_RATIO)
    print("split: ok, n_splits =", len(splits))
    return splits


def cmd_fit_weights() -> pd.DataFrame:
    """Learn weights per segment per split."""
    splits = cmd_split()
    weights_list = []
    for split in splits:
        w = fit_segment_weights(
            split.train_df,
            seed=split.seed,
            min_segment_samples=config.MIN_SEGMENT_SAMPLES,
        )
        weights_list.append(w)
    w_df = aggregate_weights(weights_list)
    _ensure_init()
    write_df("weights", w_df)
    print("fit_weights: ok")
    return w_df


def cmd_score() -> None:
    """Compute Score on train/test (for verification)."""
    splits = cmd_split()
    w_df = read_df("weights")
    w0 = w_df[w_df["seed"] == 0]
    s0 = [s for s in splits if s.seed == 0][0]
    test_sc = apply_scores(s0.test_df, w0)
    print("score: ok, sample Score mean:", test_sc["Score"].mean())


def cmd_trend_label() -> None:
    """Build trending labels."""
    splits = cmd_split()
    s0 = [s for s in splits if s.seed == 0][0]
    thr = compute_segment_thresholds(s0.train_df, config.TREND_PERCENTILE)
    tr_lab = apply_trending_label(s0.train_df, thr)
    apply_trending_label(s0.test_df, thr)
    print("trend_label: ok, train rate:", tr_lab["Trending"].mean())


def cmd_train_classifier() -> None:
    """Train and evaluate classifier."""
    splits = cmd_split()
    w_df = read_df("weights")
    s0 = [s for s in splits if s.seed == 0][0]
    w0 = w_df[w_df["seed"] == 0]
    train_sc = apply_scores(s0.train_df, w0)
    test_sc = apply_scores(s0.test_df, w0)
    thr = compute_segment_thresholds(s0.train_df, config.TREND_PERCENTILE)
    tr_lab = apply_trending_label(s0.train_df, thr)
    te_lab = apply_trending_label(s0.test_df, thr)
    train_for_clf = tr_lab.assign(Score=train_sc["Score"])
    test_for_clf = te_lab.assign(Score=test_sc["Score"])
    mdl, chosen_thr = train_trending_classifier(train_for_clf)
    print("train_classifier: chosen threshold =", chosen_thr)
    yp, _, _ = predict_trending(mdl, test_for_clf, threshold=chosen_thr)
    m = classification_metrics(te_lab["Trending"].to_numpy(), yp)
    print("train_classifier: ok", m)


def cmd_baseline() -> None:
    """Run baseline pipeline."""
    splits = cmd_split()
    s0 = [s for s in splits if s.seed == 0][0]
    bl = run_baseline_evaluation(s0, config.TREND_PERCENTILE, seed=0)
    print("baseline regression:", bl["baseline_regression"])
    print("baseline classification:", bl["baseline_classification"])


def cmd_report() -> None:
    """Aggregate and export CSVs."""
    splits = cmd_split()
    reg_list = []
    cls_list = []
    weights_list = []
    for split in splits:
        w = fit_segment_weights(
            split.train_df,
            seed=split.seed,
            min_segment_samples=config.MIN_SEGMENT_SAMPLES,
        )
        train_sc = apply_scores(split.train_df, w)
        test_sc = apply_scores(split.test_df, w)
        thr = compute_segment_thresholds(split.train_df, config.TREND_PERCENTILE)
        tr_lab = apply_trending_label(split.train_df, thr)
        te_lab = apply_trending_label(split.test_df, thr)
        train_for_clf = tr_lab.assign(Score=train_sc["Score"])
        test_for_clf = te_lab.assign(Score=test_sc["Score"])
        mdl, chosen_thr = train_trending_classifier(train_for_clf)
        yp, _, _ = predict_trending(mdl, test_for_clf, threshold=chosen_thr)
        reg_list.append(evaluate_regression(test_sc, seed=split.seed))
        cls = classification_metrics(te_lab["Trending"].to_numpy(), yp)
        cls["seed"] = split.seed
        cls_list.append(cls)
        weights_list.append(w)

    reg_df = aggregate_regression_metrics(reg_list)
    cls_df = aggregate_classification_metrics(cls_list)
    w_df = aggregate_weights(weights_list)
    save_reports(reg_df, cls_df, w_df, output_dir="outputs")
    _ensure_init()
    write_df("weights", w_df)
    write_df("regression_metrics", reg_df)
    write_df("classification_metrics", cls_df)
    print("report: ok, saved outputs/*.csv")


def cmd_diagnostics() -> None:
    """Run diagnostics."""
    _ensure_init()
    df = read_df("features_data")
    splits = cmd_split()
    w_df = read_df("weights")
    report = run_diagnostics(
        df,
        splits,
        w_df,
        min_segment_samples=config.MIN_SEGMENT_SAMPLES,
    )
    save_diagnostics_report(report, "outputs/diagnostics_report.json")
    print("diagnostics: ok, saved outputs/diagnostics_report.json")
    print("fatal_issues:", report["fatal_issues"])
    print("warnings:", report["warnings"])
    print("ok:", report["ok"])


COMMANDS = {
    "load": (cmd_load, []),
    "clean": (cmd_clean, ["load"]),
    "features": (cmd_features, ["clean"]),
    "split": (cmd_split, ["features"]),
    "fit_weights": (cmd_fit_weights, ["split"]),
    "score": (cmd_score, ["fit_weights"]),
    "trend_label": (cmd_trend_label, ["split"]),
    "train_classifier": (cmd_train_classifier, ["fit_weights"]),
    "baseline": (cmd_baseline, ["split"]),
    "report": (cmd_report, ["features"]),
    "diagnostics": (cmd_diagnostics, ["fit_weights"]),
}


def run_commands(cmds: list[str]) -> None:
    """Run commands in order, executing prerequisites."""
    done = set()
    for cmd in cmds:
        if cmd not in COMMANDS:
            print(f"Unknown command: {cmd}", file=sys.stderr)
            sys.exit(1)
        func, deps = COMMANDS[cmd]
        for d in deps:
            if d not in done:
                run_commands([d])
                done.add(d)
        func()
        done.add(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="PopWeight pipeline CLI")
    parser.add_argument(
        "commands",
        nargs="+",
        choices=list(COMMANDS.keys()) + ["all"],
        help="Commands to run (or 'all' for full pipeline)",
    )
    args = parser.parse_args()
    cmds = args.commands
    if "all" in cmds:
        cmds = [
            "load",
            "clean",
            "features",
            "split",
            "fit_weights",
            "score",
            "trend_label",
            "train_classifier",
            "baseline",
            "report",
            "diagnostics",
        ]
    run_commands(cmds)


if __name__ == "__main__":
    main()
