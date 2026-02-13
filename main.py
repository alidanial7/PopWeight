from popweight.cleaning import clean_core_columns
from popweight.config import DATA_PATH, SEGMENT_KEYS, SHEET_NAME
from popweight.features import add_segment_key, add_transforms
from popweight.io_excel import load_working_file


def main() -> None:
    df = load_working_file(DATA_PATH, SHEET_NAME)
    df_clean, report = clean_core_columns(df=df)
    df_features = add_transforms(df_clean)
    df_features = add_segment_key(df_features, keys=SEGMENT_KEYS)
    print("shape:", df_features.shape)
    print(
        "has columns:",
        all(
            c in df_features.columns
            for c in ["Likes_ll", "Comments_ll", "Shares_ll", "Reach_log", "Segment"]
        ),
    )
    print(
        df_features[
            [
                "Likes",
                "Likes_ll",
                "Comments",
                "Comments_ll",
                "Shares",
                "Shares_ll",
                "Reach",
                "Reach_log",
                "Platform",
                "Post Type",
                "Segment",
            ]
        ].head(5)
    )
    import numpy as np

    cols = ["Likes_ll", "Comments_ll", "Shares_ll", "Reach_log"]
    print("NaNs:", df_features[cols].isna().sum().to_dict())
    print("infs:", np.isinf(df_features[cols]).sum())
    print(df_features[["Likes_ll", "Comments_ll", "Shares_ll", "Reach_log"]].describe())
    print("Platforms:", df_features["Platform"].nunique())
    print("Post Types:", df_features["Post Type"].nunique())
    print("Segments:", df_features["Segment"].nunique())

    seg_counts = df_features["Segment"].value_counts()
    print("Smallest segments:\n", seg_counts.tail(10))
    print("Segments with < 20 samples:", (seg_counts < 20).sum())


if __name__ == "__main__":
    main()
