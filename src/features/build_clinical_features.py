from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# ============================================================
# CLINICAL OBSERVATION CODES
# ============================================================
OBSERVATION_CODES = {
    # Vital signs
    "systolic_bp": "8480-6",
    "diastolic_bp": "8462-4",
    "respiratory_rate": "9279-1",
    "heart_rate": "8867-4",
    "oxygen_saturation": "2708-6",
    "temperature": "8310-5",

    # Body measurements
    "weight": "29463-7",
    "height": "8302-2",
    "bmi": "39156-5",

    # Kidney
    "egfr": "33914-3",
    "creatinine": "38483-4",
    "urea_nitrogen": "6299-2",

    # Metabolic / electrolytes
    "glucose": "2339-0",
    "hba1c": "4548-4",
    "sodium": "2947-0",
    "potassium": "2823-3",
    "chloride": "2069-3",
    "calcium": "49765-1",

    # Liver
    "ast": "1920-8",
    "alt": "1742-6",
    "bilirubin": "1975-2",
    "albumin": "1751-7",
    "alkaline_phosphatase": "6768-6",

    # Blood / CBC
    "hemoglobin": "718-7",
    "hematocrit": "4544-3",
    "platelets": "777-3",
    "wbc": "6690-2",

    # Lipids
    "triglycerides": "2571-8",
    "total_cholesterol": "2093-3",
    "ldl": "18262-6",
    "hdl": "2085-9",
}

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    cohort = pd.read_csv(
        PROCESSED_DIR / "covid_cohort.csv"
    )

    observations = pd.read_csv(
        RAW_DIR / "observations.csv"
    )

    return cohort, observations


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(cohort, observations):

    cohort["COVID_DATE"] = pd.to_datetime(
        cohort["COVID_DATE"],
        errors="coerce"
    )

    cohort["BIRTHDATE"] = pd.to_datetime(
        cohort["BIRTHDATE"],
        errors="coerce"
    )

    cohort["DEATHDATE"] = pd.to_datetime(
        cohort["DEATHDATE"],
        errors="coerce"
    )

    observations["DATE"] = pd.to_datetime(
        observations["DATE"],
        errors="coerce"
    )

    # Convert VALUE to numeric.
    # Text observations become NaN and will be ignored.
    observations["VALUE_NUMERIC"] = pd.to_numeric(
        observations["VALUE"],
        errors="coerce"
    )

    return cohort, observations


# ============================================================
# BUILD TARGET + DEMOGRAPHICS
# ============================================================

def build_base_dataset(cohort):

    df = cohort[
        [
            "PATIENT",
            "COVID_DATE",
            "BIRTHDATE",
            "GENDER",
            "MARITAL",
            "RACE",
            "ETHNICITY",
            "DEATHDATE",
        ]
    ].copy()

    # Age at COVID diagnosis
    df["AGE"] = (
        (df["COVID_DATE"] - df["BIRTHDATE"]).dt.days
        / 365.25
    )

    # Days between COVID diagnosis and death
    df["DAYS_TO_DEATH"] = (
        df["DEATHDATE"] - df["COVID_DATE"]
    ).dt.days

    # Target:
    # 1 = death between 1 and 30 days after COVID
    # 0 = otherwise
    df["DEATH_30D"] = (
        df["DAYS_TO_DEATH"]
        .between(1, 30)
    ).astype(int)

    return df


# ============================================================
# EXTRACT FEATURES FOR ONE CLINICAL MEASUREMENT
# ============================================================

def extract_measurement_features(
    base_df,
    observations,
    feature_name,
    code
):

    # Select this measurement
    subset = observations[
        observations["CODE"].astype(str) == str(code)
    ].copy()

    # Keep numeric values only
    subset = subset.dropna(
        subset=["VALUE_NUMERIC", "DATE"]
    )

    if subset.empty:

        print(
            f"  {feature_name}: no numeric observations"
        )

        return base_df

    # Join with each patient's COVID date
    subset = subset.merge(
        base_df[
            ["PATIENT", "COVID_DATE"]
        ],
        on="PATIENT",
        how="inner"
    )
    subset = subset.sort_values(
    ["PATIENT", "DATE"]
)

    # --------------------------------------------------------
    # CRITICAL:
    # KEEP ONLY OBSERVATIONS BEFORE COVID
    # --------------------------------------------------------

    subset = subset[
        subset["DATE"] < subset["COVID_DATE"]
    ].copy()

    if subset.empty:

        print(
            f"  {feature_name}: no pre-COVID observations"
        )

        return base_df

    # Days between latest measurement and COVID date
    subset["DAYS_BEFORE_COVID"] = (
        subset["COVID_DATE"] - subset["DATE"]
    ).dt.days

    # --------------------------------------------------------
    # AGGREGATE FEATURES
    # --------------------------------------------------------

    aggregated = (
        subset
        .groupby("PATIENT")
        .agg(
            **{
                f"{feature_name}_latest": (
                    "VALUE_NUMERIC",
                    "last"
                ),
                f"{feature_name}_mean": (
                    "VALUE_NUMERIC",
                    "mean"
                ),
                f"{feature_name}_min": (
                    "VALUE_NUMERIC",
                    "min"
                ),
                f"{feature_name}_max": (
                    "VALUE_NUMERIC",
                    "max"
                ),
                f"{feature_name}_count": (
                    "VALUE_NUMERIC",
                    "count"
                ),
                f"{feature_name}_days_since_last": (
                    "DAYS_BEFORE_COVID",
                    "min"
                ),
            }
        )
        .reset_index()
    )

    # IMPORTANT:
    # "last" only works correctly if sorted by date.
    # We will fix latest below using an explicit sort.

    latest = (
        subset
        .sort_values("DATE")
        .groupby("PATIENT")
        .tail(1)
        [
            [
                "PATIENT",
                "VALUE_NUMERIC"
            ]
        ]
        .rename(
            columns={
                "VALUE_NUMERIC":
                    f"{feature_name}_latest"
            }
        )
    )

    # Remove the earlier "last" version
    aggregated = aggregated.drop(
        columns=[
            f"{feature_name}_latest"
        ]
    )

    aggregated = aggregated.merge(
        latest,
        on="PATIENT",
        how="left"
    )

    base_df = base_df.merge(
        aggregated,
        on="PATIENT",
        how="left"
    )

    patient_count = aggregated["PATIENT"].nunique()

    print(
        f"  {feature_name}: "
        f"{patient_count} patients"
    )

    return base_df


# ============================================================
# BUILD ALL CLINICAL FEATURES
# ============================================================

def build_clinical_features(
    base_df,
    observations
):

    print(
        "\n========== BUILDING CLINICAL FEATURES ==========\n"
    )

    for feature_name, code in OBSERVATION_CODES.items():

        base_df = extract_measurement_features(
            base_df=base_df,
            observations=observations,
            feature_name=feature_name,
            code=code
        )

    return base_df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========== HEALTH TWIN FEATURE ENGINEERING V2 ==========\n"
    )

    print("Loading data...")

    cohort, observations = load_data()

    cohort, observations = prepare_data(
        cohort,
        observations
    )

    print(
        f"COVID cohort: {len(cohort)} patients"
    )

    print(
        f"Observations: {len(observations)} rows"
    )

    # --------------------------------------------------------
    # BUILD BASE DATASET
    # --------------------------------------------------------

    print(
        "\nBuilding demographic and outcome data..."
    )

    df = build_base_dataset(cohort)

    # --------------------------------------------------------
    # BUILD CLINICAL FEATURES
    # --------------------------------------------------------

    df = build_clinical_features(
        df,
        observations
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        PROCESSED_DIR /
        "mortality_features_v2.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        "\n========== FINAL DATASET ==========\n"
    )

    print("Shape:", df.shape)

    print(
        "\nTarget distribution:"
    )

    print(
        df["DEATH_30D"]
        .value_counts()
    )

    print(
        "\nMissing values in clinical features:"
    )

    clinical_columns = [
        column
        for column in df.columns
        if column not in [
            "PATIENT",
            "COVID_DATE",
            "BIRTHDATE",
            "DEATHDATE",
            "DAYS_TO_DEATH",
            "DEATH_30D",
            "GENDER",
            "MARITAL",
            "RACE",
            "ETHNICITY",
            "AGE",
        ]
    ]

    missing = (
        df[clinical_columns]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )

    print(
        missing
        .head(30)
    )

    print(
        "\nSaved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()