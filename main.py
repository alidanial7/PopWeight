from sklearn.metrics import r2_score

from popweight.cleaning import clean_core_columns
from popweight.config import (
    DATA_PATH,
    MIN_SEGMENT_SAMPLES,
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
        seed=s0.seed,
        min_segment_samples=MIN_SEGMENT_SAMPLES,
    )
    print(w0.shape)
    print(w0[["Segment", "n_train", "strategy"]].head())
    print("unique segments:", w0["Segment"].nunique())
    print("strategy counts:\n", w0["strategy"].value_counts())
    print(w0[["alpha", "beta", "gamma", "intercept", "r2_train"]].describe())
    print(
        "Any NaN:",
        w0[["alpha", "beta", "gamma", "intercept", "r2_train"]].isna().any().to_dict(),
    )

    seg = "Instagram__Video"
    seg_df = s0.train_df[s0.train_df["Segment"] == seg].copy()
    row = w0[w0["Segment"] == seg].iloc[0]
    y_true = seg_df["Reach_log"]
    y_pred = (
        row["intercept"]
        + row["alpha"] * seg_df["Likes_ll"]
        + row["beta"] * seg_df["Comments_ll"]
        + row["gamma"] * seg_df["Shares_ll"]
    )
    print("manual r2:", r2_score(y_true, y_pred))
    print("stored r2:", row["r2_train"])


if __name__ == "__main__":
    main()
