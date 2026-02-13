import os

import pandas as pd

from popweight.baseline import compare_with_popweight, run_baseline_evaluation
from popweight.cleaning import clean_core_columns
from popweight.config import (
    DATA_PATH,
    MIN_SEGMENT_SAMPLES,
    RANDOM_SEEDS,
    SEGMENT_KEYS,
    SHEET_NAME,
    SQLITE_PATH,
    TRAIN_RATIO,
    TREND_PERCENTILE,
)
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
from popweight.scoring import apply_scores
from popweight.splits import make_splits
from popweight.storage import init_db, read_df, write_df
from popweight.trending import apply_trending_label, compute_segment_thresholds
from popweight.weights import fit_segment_weights


def main() -> None:
    df = load_working_file(DATA_PATH, SHEET_NAME)
    df_clean, report = clean_core_columns(df=df)
    df_features = add_transforms(df_clean)
    df_features = add_segment_key(df_features, keys=SEGMENT_KEYS)

    init_db(SQLITE_PATH)
    write_df("features_data", df_features)

    splits = make_splits(df_features, RANDOM_SEEDS, TRAIN_RATIO)
    s0 = [s for s in splits if s.seed == 0][0]
    w0 = fit_segment_weights(
        s0.train_df,
        seed=0,
        min_segment_samples=MIN_SEGMENT_SAMPLES,
    )
    train_scored = apply_scores(s0.train_df, w0)
    test_scored = apply_scores(s0.test_df, w0)
    thresholds = compute_segment_thresholds(s0.train_df, TREND_PERCENTILE)
    train_labeled = apply_trending_label(s0.train_df, thresholds)
    test_labeled = apply_trending_label(s0.test_df, thresholds)

    model = train_trending_classifier(train_labeled.assign(Score=train_scored["Score"]))
    y_pred, _ = predict_trending(
        model,
        test_labeled.assign(Score=test_scored["Score"]),
    )

    y_true = test_labeled["Trending"].to_numpy()
    pw_reg = evaluate_regression(test_scored, seed=0)
    pw_cls = classification_metrics(y_true, y_pred)
    pw_cls["seed"] = 0

    bl0 = run_baseline_evaluation(s0, TREND_PERCENTILE, seed=0)
    print("baseline regression:", bl0["baseline_regression"])
    print("baseline classification:", bl0["baseline_classification"])
    d0 = compare_with_popweight(
        pw_reg,
        pw_cls,
        bl0["baseline_regression"],
        bl0["baseline_classification"],
    )
    print("deltas:", d0)

    # Step 15: Aggregation + reporting across all seeds
    reg_list = []
    cls_list = []
    weights_list = []
    for split in splits:
        w = fit_segment_weights(
            split.train_df,
            seed=split.seed,
            min_segment_samples=MIN_SEGMENT_SAMPLES,
        )
        train_sc = apply_scores(split.train_df, w)
        test_sc = apply_scores(split.test_df, w)
        thr = compute_segment_thresholds(split.train_df, TREND_PERCENTILE)
        tr_lab = apply_trending_label(split.train_df, thr)
        te_lab = apply_trending_label(split.test_df, thr)
        mdl = train_trending_classifier(tr_lab.assign(Score=train_sc["Score"]))
        yp, _ = predict_trending(mdl, te_lab.assign(Score=test_sc["Score"]))
        reg_list.append(evaluate_regression(test_sc, seed=split.seed))
        cls = classification_metrics(te_lab["Trending"].to_numpy(), yp)
        cls["seed"] = split.seed
        cls_list.append(cls)
        weights_list.append(w)

    reg_df = aggregate_regression_metrics(reg_list)
    cls_df = aggregate_classification_metrics(cls_list)
    w_df = aggregate_weights(weights_list)
    save_reports(reg_df, cls_df, w_df, output_dir="outputs")
    write_df("weights", w_df)
    write_df("regression_metrics", reg_df)
    write_df("classification_metrics", cls_df)
    print(
        "Saved outputs/metrics_regression.csv, "
        "metrics_classification.csv, weights.csv"
    )

    # Verification
    print(os.path.exists("outputs/metrics_regression.csv"))
    print(os.path.exists("outputs/metrics_classification.csv"))
    print(os.path.exists("outputs/weights.csv"))

    reg = pd.read_csv("outputs/metrics_regression.csv")
    cls = pd.read_csv("outputs/metrics_classification.csv")
    w = pd.read_csv("outputs/weights.csv")

    print("reg shape:", reg.shape)
    print("cls shape:", cls.shape)
    print("weights shape:", w.shape)

    print(reg.tail(3))
    print(cls.tail(3))

    print("weights table:", read_df("weights").shape)
    print("regression_metrics table:", read_df("regression_metrics").shape)
    print("classification_metrics table:", read_df("classification_metrics").shape)


if __name__ == "__main__":
    main()
