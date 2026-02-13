from popweight.cleaning import clean_core_columns
from popweight.config import (
    DATA_PATH,
    RANDOM_SEEDS,
    SEGMENT_KEYS,
    SHEET_NAME,
    SQLITE_PATH,
    TRAIN_RATIO,
    TREND_PERCENTILE,
)
from popweight.features import add_segment_key, add_transforms
from popweight.io_excel import load_working_file
from popweight.splits import make_splits
from popweight.storage import init_db, write_df
from popweight.trending import apply_trending_label, compute_segment_thresholds


def main() -> None:
    df = load_working_file(DATA_PATH, SHEET_NAME)
    df_clean, report = clean_core_columns(df=df)
    df_features = add_transforms(df_clean)
    df_features = add_segment_key(df_features, keys=SEGMENT_KEYS)

    init_db(SQLITE_PATH)
    write_df("features_data", df_features)

    splits = make_splits(df_features, RANDOM_SEEDS, TRAIN_RATIO)
    s0 = [s for s in splits if s.seed == 0][0]

    thresholds = compute_segment_thresholds(s0.train_df, TREND_PERCENTILE)
    train_labeled = apply_trending_label(s0.train_df, thresholds)
    test_labeled = apply_trending_label(s0.test_df, thresholds)

    print("train trending rate:", train_labeled["Trending"].mean())
    print("test trending rate:", test_labeled["Trending"].mean())

    seg_rate_train = train_labeled.groupby("Segment")["Trending"].mean().sort_values()
    seg_rate_test = test_labeled.groupby("Segment")["Trending"].mean().sort_values()
    print("train per-seg min/max:", seg_rate_train.min(), seg_rate_train.max())
    print("test per-seg min/max:", seg_rate_test.min(), seg_rate_test.max())
    print(seg_rate_train.head())
    print(seg_rate_train.tail())

    print("threshold rows:", thresholds.shape)
    print("unique segments thresholds:", thresholds["Segment"].nunique())
    print("segments in train:", s0.train_df["Segment"].nunique())


if __name__ == "__main__":
    main()
