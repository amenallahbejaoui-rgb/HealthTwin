import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def main():

    path = PROCESSED_DIR / "covid_cohort.csv"

    df = pd.read_csv(path)

    df["COVID_DATE"] = pd.to_datetime(
        df["COVID_DATE"],
        errors="coerce"
    )

    df["DEATHDATE"] = pd.to_datetime(
        df["DEATHDATE"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # TIME FROM COVID TO DEATH
    # ---------------------------------------------------------

    df["DAYS_TO_DEATH"] = (
        df["DEATHDATE"] - df["COVID_DATE"]
    ).dt.days

    print("\n========== MORTALITY VALIDATION ==========\n")

    print("Total positive patients:", len(df))

    print(
        "Patients with death date:",
        df["DEATHDATE"].notna().sum()
    )

    print(
        "Deaths after COVID:",
        (df["DAYS_TO_DEATH"] > 0).sum()
    )

    print(
        "Deaths on COVID date:",
        (df["DAYS_TO_DEATH"] == 0).sum()
    )

    print(
        "Deaths before COVID:",
        (df["DAYS_TO_DEATH"] < 0).sum()
    )

    print("\nDays from COVID to death:")

    print(
        df.loc[
            df["DAYS_TO_DEATH"].notna(),
            "DAYS_TO_DEATH"
        ].describe()
    )

    print("\nDeath timing buckets:")

    buckets = pd.cut(
        df["DAYS_TO_DEATH"],
        bins=[
            -float("inf"),
            -1,
            0,
            1,
            7,
            14,
            30,
            90,
            float("inf")
        ],
        labels=[
            "Before COVID",
            "Same day",
            "1 day",
            "2-7 days",
            "8-14 days",
            "15-30 days",
            "31-90 days",
            "90+ days"
        ]
    )

    print(
        buckets.value_counts(
            sort=False,
            dropna=False
        )
    )

    # ---------------------------------------------------------
    # AGE AT COVID
    # ---------------------------------------------------------

    df["BIRTHDATE"] = pd.to_datetime(
        df["BIRTHDATE"],
        errors="coerce"
    )

    df["AGE_AT_COVID"] = (
        (
            df["COVID_DATE"] -
            df["BIRTHDATE"]
        ).dt.days / 365.25
    )

    print("\nAge at COVID:")

    print(
        df["AGE_AT_COVID"].describe()
    )

    print("\nGender:")

    print(
        df["GENDER"].value_counts(
            dropna=False
        )
    )


if __name__ == "__main__":
    main()