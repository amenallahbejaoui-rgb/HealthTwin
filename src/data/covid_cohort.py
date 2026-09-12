from pathlib import Path
import pandas as pd

from load import load_all


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


COVID_TEST_CODE = "94531-1"


def inspect_covid_observations(observations):

    print("\n========== COVID OBSERVATIONS ==========\n")

    covid = observations[
        observations["CODE"].astype(str) == COVID_TEST_CODE
    ].copy()

    print("COVID test rows:", len(covid))

    print("\nUnique descriptions:")
    print(covid["DESCRIPTION"].value_counts())

    print("\nUnique values:")
    print(covid["VALUE"].value_counts(dropna=False))

    print("\nPatients:")
    print(covid["PATIENT"].nunique())

    return covid


def build_cohort(data):

    patients = data["patients"].copy()
    observations = data["observations"].copy()

    covid = observations[
        observations["CODE"].astype(str) == COVID_TEST_CODE
    ].copy()

    covid["DATE"] = pd.to_datetime(
        covid["DATE"],
        errors="coerce"
    )

    # Positive COVID tests.
    #
    # We inspect the actual VALUE strings first instead of
    # hardcoding a potentially incorrect value.
    positive_values = [
        "Detected (qualifier value)",
        "Detected",
        "Positive",
        "positive",
    ]

    positive = covid[
        covid["VALUE"]
        .astype(str)
        .isin(positive_values)
    ].copy()

    print("\n========== COHORT ==========\n")

    print("Positive COVID observations:", len(positive))
    print("Positive COVID patients:", positive["PATIENT"].nunique())

    # First positive COVID test for each patient
    first_positive = (
        positive
        .sort_values("DATE")
        .groupby("PATIENT", as_index=False)
        .first()
    )

    first_positive = first_positive[
        ["PATIENT", "DATE", "VALUE", "DESCRIPTION"]
    ]

    first_positive = first_positive.rename(
        columns={
            "DATE": "COVID_DATE",
            "VALUE": "COVID_TEST_RESULT",
        }
    )

    # Patient demographics
    cohort = first_positive.merge(
        patients,
        left_on="PATIENT",
        right_on="Id",
        how="left"
    )

    # Mortality target
    cohort["DEATHDATE"] = pd.to_datetime(
        cohort["DEATHDATE"],
        errors="coerce"
    )

    cohort["DIED"] = cohort["DEATHDATE"].notna()

    print("\nOutcome distribution:")
    print(cohort["DIED"].value_counts())

    print("\nCohort columns:")
    print(cohort.columns.tolist())

    output = PROCESSED_DIR / "covid_cohort.csv"

    cohort.to_csv(
        output,
        index=False
    )

    print("\nSaved:")
    print(output)

    return cohort


def main():

    data = load_all()

    covid = inspect_covid_observations(
        data["observations"]
    )

    build_cohort(data)


if __name__ == "__main__":
    main()