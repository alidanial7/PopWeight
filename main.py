import os

from popweight.cleaning import clean_core_columns
from popweight.config import DATA_PATH, SEGMENT_KEYS, SHEET_NAME, SQLITE_PATH
from popweight.features import add_segment_key, add_transforms
from popweight.io_excel import load_working_file
from popweight.storage import init_db, read_df, write_df


def main() -> None:
    df = load_working_file(DATA_PATH, SHEET_NAME)
    df_clean, report = clean_core_columns(df=df)
    df_features = add_transforms(df_clean)
    df_features = add_segment_key(df_features, keys=SEGMENT_KEYS)

    init_db(SQLITE_PATH)
    write_df("features_data", df_features)

    print("sqlite exists:", os.path.exists(SQLITE_PATH), SQLITE_PATH)
    df_db = read_df("features_data")
    print("db shape:", df_db.shape)
    print("same columns:", df_db.columns.tolist() == df_features.columns.tolist())
    df_inst = read_df("features_data", where="Platform = 'Instagram'")
    print(df_inst.shape, df_inst["Platform"].unique())


if __name__ == "__main__":
    main()
