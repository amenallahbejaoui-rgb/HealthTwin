from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_data():

    patients = pd.read_csv(RAW_DIR / "patients.csv")
    cohort = pd.read_csv(PROCESSED_DIR / "covid_cohort.csv")

    conditions = pd.read_csv(RAW_DIR / "conditions.csv")
    medications = pd.read_csv(RAW_DIR / "medications.csv")
    encounters = pd.read_csv(RAW_DIR / "encounters.csv")
    observations = pd.read_csv(RAW_DIR / "observations.csv")

    return (
        patients,
        cohort,
        conditions,
        medications,
        encounters,
        observations,
    )


def prepare_dates(cohort, conditions, medications, encounters, observations):

    cohort["COVID_DATE"] = pd.to_datetime(
        cohort["COVID_DATE"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    cohort["BIRTHDATE"] = pd.to_datetime(
        cohort["BIRTHDATE"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    cohort["DEATHDATE"] = pd.to_datetime(
        cohort["DEATHDATE"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    conditions["START"] = pd.to_datetime(
        conditions["START"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    medications["START"] = pd.to_datetime(
        medications["START"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    encounters["START"] = pd.to_datetime(
        encounters["START"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    observations["DATE"] = pd.to_datetime(
        observations["DATE"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    return cohort, conditions, medications, encounters, observations


def build_base_features(cohort):

    df = cohort[
        [
            "PATIENT",
            "COVID_DATE",
            "DEATHDATE",
            "BIRTHDATE",
            "GENDER",
            "MARITAL",
            "RACE",
            "ETHNICITY",
            "DIED",
        ]
    ].copy()

    # ---------------------------------------------------------
    # AGE
    # ---------------------------------------------------------

    df["AGE"] = (
        (df["COVID_DATE"] - df["BIRTHDATE"]).dt.days
        / 365.25
    )

    # ---------------------------------------------------------
    # 30-DAY MORTALITY TARGET
    # ---------------------------------------------------------

    df["DAYS_TO_DEATH"] = (
        df["DEATHDATE"] - df["COVID_DATE"]
    ).dt.days

    df["DEATH_30D"] = (
        df["DAYS_TO_DEATH"].between(1, 30)
    ).astype(int)

    return df


def add_condition_history(df, conditions):

    history = (
        conditions
        .merge(
            df[["PATIENT", "COVID_DATE"]],
            left_on="PATIENT",
            right_on="PATIENT",
            how="inner"
        )
    )

    history = history[
        history["START"] < history["COVID_DATE"]
    ]

    features = (
        history
        .groupby("PATIENT")
        .agg(
            previous_conditions=("DESCRIPTION", "nunique"),
            condition_events=("DESCRIPTION", "count"),
        )
        .reset_index()
    )

    df = df.merge(
        features,
        on="PATIENT",
        how="left"
    )

    return df


def add_medication_history(df, medications):

    history = (
        medications
        .merge(
            df[["PATIENT", "COVID_DATE"]],
            on="PATIENT",
            how="inner"
        )
    )

    history = history[
        history["START"] < history["COVID_DATE"]
    ]

    features = (
        history
        .groupby("PATIENT")
        .agg(
            previous_medications=("DESCRIPTION", "nunique"),
            medication_events=("DESCRIPTION", "count"),
        )
        .reset_index()
    )

    df = df.merge(
        features,
        on="PATIENT",
        how="left"
    )

    return df


def add_encounter_history(df, encounters):

    history = (
        encounters
        .merge(
            df[["PATIENT", "COVID_DATE"]],
            on="PATIENT",
            how="inner"
        )
    )

    history = history[
        history["START"] < history["COVID_DATE"]
    ]

    features = (
        history
        .groupby("PATIENT")
        .agg(
            previous_encounters=("Id", "count"),
            previous_encounter_types=(
                "ENCOUNTERCLASS",
                "nunique"
            ),
        )
        .reset_index()
    )

    df = df.merge(
        features,
        on="PATIENT",
        how="left"
    )

    return df


def add_observation_history(df, observations):

    history = (
        observations
        .merge(
            df[["PATIENT", "COVID_DATE"]],
            on="PATIENT",
            how="inner"
        )
    )

    history = history[
        history["DATE"] < history["COVID_DATE"]
    ]

    features = (
        history
        .groupby("PATIENT")
        .agg(
            previous_observations=("DESCRIPTION", "count"),
            unique_observations=("DESCRIPTION", "nunique"),
        )
        .reset_index()
    )

    df = df.merge(
        features,
        on="PATIENT",
        how="left"
    )

    return df


def main():

    (
        patients,
        cohort,
        conditions,
        medications,
        encounters,
        observations,
    ) = load_data()

    (
        cohort,
        conditions,
        medications,
        encounters,
        observations,
    ) = prepare_dates(
        cohort,
        conditions,
        medications,
        encounters,
        observations
    )

    df = build_base_features(cohort)

    print("\n========== BASE DATASET ==========\n")

    print("Patients:", len(df))

    print("\nTarget:")
    print(df["DEATH_30D"].value_counts())

    print("\nMortality rate:")
    print(df["DEATH_30D"].mean())

    print("\nAdding condition history...")
    df = add_condition_history(df, conditions)

    print("Adding medication history...")
    df = add_medication_history(df, medications)

    print("Adding encounter history...")
    df = add_encounter_history(df, encounters)

    print("Adding observation history...")
    df = add_observation_history(df, observations)

    # Missing historical counts mean zero events.
    count_columns = [
        "previous_conditions",
        "condition_events",
        "previous_medications",
        "medication_events",
        "previous_encounters",
        "previous_encounter_types",
        "previous_observations",
        "unique_observations",
    ]

    for column in count_columns:
        df[column] = df[column].fillna(0)

    output = PROCESSED_DIR / "mortality_30d_dataset.csv"

    df.to_csv(
        output,
        index=False
    )

    print("\n========== FINAL DATASET ==========\n")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(df["DEATH_30D"].value_counts())

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()