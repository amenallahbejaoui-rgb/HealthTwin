from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def main():

    conditions = pd.read_csv(
        RAW_DIR / "conditions.csv"
    )

    print(
        "\n========== CONDITION INVENTORY ==========\n"
    )

    print(
        "Total condition records:",
        len(conditions)
    )

    print(
        "Unique patients:",
        conditions["PATIENT"].nunique()
    )

    print(
        "Unique codes:",
        conditions["CODE"].nunique()
    )

    print(
        "Unique descriptions:",
        conditions["DESCRIPTION"].nunique()
    )

    print(
        "\n========== MOST COMMON CONDITIONS ==========\n"
    )

    common = (
        conditions
        .groupby(
            ["CODE", "DESCRIPTION"],
            dropna=False
        )
        .size()
        .sort_values(
            ascending=False
        )
        .head(150)
    )

    print(
        common.to_string()
    )


if __name__ == "__main__":
    main()