from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"


def main():

    observations = pd.read_csv(
        RAW_DIR / "observations.csv"
    )

    print("\n========== OBSERVATION INVENTORY ==========\n")

    print(
        "Total observations:",
        len(observations)
    )

    print(
        "Unique patients:",
        observations["PATIENT"].nunique()
    )

    print(
        "Unique codes:",
        observations["CODE"].nunique()
    )

    print(
        "Unique descriptions:",
        observations["DESCRIPTION"].nunique()
    )

    print("\n========== MOST COMMON OBSERVATIONS ==========\n")

    common = (
        observations
        .groupby(
            ["CODE", "DESCRIPTION"],
            dropna=False
        )
        .size()
        .sort_values(ascending=False)
        .head(100)
    )

    print(common.to_string())

    print("\n========== OBSERVATION TYPES ==========\n")

    print(
        observations["TYPE"]
        .value_counts(dropna=False)
    )
    print("\n========== UNITS ==========\n")

    print(
        observations["UNITS"]
        .value_counts(dropna=False)
        .head(50)
    )


if __name__ == "__main__":
    main()