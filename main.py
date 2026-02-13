import pandas as pd

from popweight.cleaning import clean_core_columns
from popweight.config import (
    DATA_PATH,
    RANDOM_SEEDS,
    SEGMENT_KEYS,
    SHEET_NAME,
    SQLITE_PATH,
    TRAIN_RATIO,
)
from popweight.features import add_segment_key, add_transforms
from popweight.io_excel import load_working_file
from popweight.splits import make_splits
from popweight.storage import init_db, write_df


def main() -> None:
    df = load_working_file(DATA_PATH, SHEET_NAME)
    df_clean, report = clean_core_columns(df=df)
    df_features = add_transforms(df_clean)
    df_features = add_segment_key(df_features, keys=SEGMENT_KEYS)

    init_db(SQLITE_PATH)
    write_df("features_data", df_features)

    splits = make_splits(df_features, RANDOM_SEEDS, TRAIN_RATIO)
    s0 = [s for s in splits if s.seed == 0][0]
    print("train shape:", s0.train_df.shape)
    print("test shape:", s0.test_df.shape)
    print("total:", s0.train_df.shape[0] + s0.test_df.shape[0])

    train_segs = set(s0.train_df["Segment"].unique())
    test_segs = set(s0.test_df["Segment"].unique())
    print("segments train:", len(train_segs))
    print("segments test:", len(test_segs))
    print("test minus train:", test_segs - train_segs)

    train_dist = s0.train_df["Segment"].value_counts(normalize=True).sort_index()
    test_dist = s0.test_df["Segment"].value_counts(normalize=True).sort_index()
    dist_diff = (train_dist - test_dist).abs()
    print("max abs diff:", dist_diff.max())
    print(pd.concat([train_dist, test_dist, dist_diff], axis=1).head(12))


if __name__ == "__main__":
    main()
