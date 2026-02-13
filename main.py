from popweight.cleaning import clean_core_columns
from popweight.config import DATA_PATH, SHEET_NAME
from popweight.io_excel import load_working_file


def main() -> None:
    df = load_working_file(DATA_PATH, SHEET_NAME)
    df_clean, report = clean_core_columns(df=df)
    print(df_clean.shape)
    print(report)
    print(
        df_clean["Reach"].quantile(0.995),
        df["Reach"].quantile(0.995),
    )


if __name__ == "__main__":
    main()
