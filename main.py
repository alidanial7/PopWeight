import numpy as np

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
from popweight.features import add_segment_key, add_transforms
from popweight.io_excel import load_working_file
from popweight.models import predict_trending, train_trending_classifier
from popweight.scoring import apply_scores
from popweight.splits import make_splits
from popweight.storage import init_db, write_df
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
    y_pred, y_prob = predict_trending(
        model,
        test_labeled.assign(Score=test_scored["Score"]),
    )

    print("pred shape:", y_pred.shape, "prob shape:", y_prob.shape)
    print("pred positive rate:", y_pred.mean())
    print("prob min/mean/max:", y_prob.min(), y_prob.mean(), y_prob.max())
    print("unique preds:", np.unique(y_pred, return_counts=True))


if __name__ == "__main__":
    main()
