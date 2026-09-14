from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def main():
    conditions = pd.read_csv(RAW_DIR / "conditions.csv")
    cohort = pd.read_csv(PROCESSED_DIR / "covid_cohort.csv")

    conditions["START"] = pd.to_datetime(conditions["START"], errors="coerce")
    cohort["COVID_DATE"] = pd.to_datetime(cohort["COVID_DATE"], errors="coerce")

    df = conditions.merge(
        cohort[["PATIENT", "COVID_DATE"]],
        on="PATIENT",
        how="inner"
    )

    df["DAYS_FROM_COVID"] = (
        df["START"] - df["COVID_DATE"]
    ).dt.days

    print("\n========== CONDITION TIMING ==========\n")

    print("Condition records for COVID cohort:", len(df))

    print("\nBefore COVID:")
    print((df["DAYS_FROM_COVID"] < 0).sum())

    print("\nSame day as COVID:")
    print((df["DAYS_FROM_COVID"] == 0).sum())

    print("\nAfter COVID:")
    print((df["DAYS_FROM_COVID"] > 0).sum())

    print("\n========== MOST COMMON PRE-COVID CONDITIONS ==========\n")

    pre = df[df["DAYS_FROM_COVID"] < 0]

    common = (
        pre.groupby(["CODE", "DESCRIPTION"])
        .size()
        .sort_values(ascending=False)
        .head(80)
    )

    print(common.to_string())

    print("\n========== COVID-ADJACENT CONDITIONS ==========\n")

    adjacent = df[
        (df["DAYS_FROM_COVID"] >= -7) &
        (df["DAYS_FROM_COVID"] <= 7)
    ]

    common_adjacent = (
        adjacent.groupby(["CODE", "DESCRIPTION"])
        .size()
        .sort_values(ascending=False)
        .head(50)
    )

    print(common_adjacent.to_string())


if __name__ == "__main__":
    main()