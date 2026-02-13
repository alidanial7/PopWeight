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
from popweight.scoring import apply_scores
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
        seed=0,
        min_segment_samples=MIN_SEGMENT_SAMPLES,
    )
    train_scored = apply_scores(s0.train_df, w0)
    test_scored = apply_scores(s0.test_df, w0)

    print("train_scored shape:", train_scored.shape)
    print("test_scored shape:", test_scored.shape)
    print(
        "Score NaNs train/test:",
        train_scored["Score"].isna().sum(),
        test_scored["Score"].isna().sum(),
    )
    print(train_scored["Score"].describe())
    print(test_scored["Score"].describe())


if __name__ == "__main__":
    main()
