from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# Conditions that represent chronic/pre-existing risk factors.
CONDITION_GROUPS = {
    "has_obesity": [
        "162864005",
        "408512008",
    ],

    "has_prediabetes": [
        "15777000",
    ],

    "has_diabetes": [
        "44054006",
    ],

    "has_hypertension": [
        "59621000",
    ],

    "has_hyperlipidemia": [
        "55822004",
    ],

    "has_hypertriglyceridemia": [
        "302870006",
    ],

    "has_metabolic_syndrome": [
        "237602007",
    ],

    "has_anemia": [
        "271737000",
    ],

    "has_coronary_heart_disease": [
        "53741008",
    ],

    "has_atrial_fibrillation": [
        "49436004",
    ],

    "has_heart_failure": [
        "88805009",
        "84114007",
    ],

    "has_stroke": [
        "230690007",
    ],

    "has_ckd": [
        "431855005",
        "431856006",
    ],

    "has_copd": [
        "87433001",
        "185086009",
    ],

    "has_asthma": [
        "233678006",
    ],

    "has_dementia": [
        "26929004",
        "230265002",
    ],

    "has_osteoporosis": [
        "64859006",
    ],

    "has_smoking": [
        "449868002",
    ],

    "has_thrombotic_history": [
        "132281000119108",
        "706870000",
    ],

    "has_cancer": [
        "126906006",
        "92691004",
        "254637007",
        "254837009",
        "424132000",
        "67811000119102",
        "254632001",
        "363406005",
        "109838007",
        "94260004",
        "314994000",
    ],
}


def main():

    conditions = pd.read_csv(
        RAW_DIR / "conditions.csv"
    )

    cohort = pd.read_csv(
        PROCESSED_DIR / "covid_cohort.csv"
    )

    conditions["START"] = pd.to_datetime(
        conditions["START"],
        errors="coerce"
    )

    cohort["COVID_DATE"] = pd.to_datetime(
        cohort["COVID_DATE"],
        errors="coerce"
    )

    # Keep only patients in the COVID-positive cohort.
    df = conditions.merge(
        cohort[["PATIENT", "COVID_DATE"]],
        on="PATIENT",
        how="inner"
    )

    # CRITICAL:
    # Only use information available BEFORE COVID.
    df = df[
        df["START"].notna() &
        (df["START"] < df["COVID_DATE"])
    ].copy()

    print("\n========== PRE-COVID CONDITIONS ==========\n")
    print("Pre-COVID condition records:", len(df))
    print("Patients with pre-COVID conditions:", df["PATIENT"].nunique())

    # ---------------------------------------------------------
    # General condition history
    # ---------------------------------------------------------

    condition_counts = (
        df.groupby("PATIENT")
        .agg(
            condition_event_count=("CODE", "size"),
            condition_count=("CODE", "nunique"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Disease-group features
    # ---------------------------------------------------------

    features = condition_counts.copy()

    for feature_name, codes in CONDITION_GROUPS.items():

        patients = set(
            df.loc[
                df["CODE"].astype(str).isin(codes),
                "PATIENT"
            ]
        )

        features[feature_name] = (
            features["PATIENT"]
            .isin(patients)
            .astype(int)
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = PROCESSED_DIR / "condition_features.csv"

    features.to_csv(
        output,
        index=False
    )

    print("\n========== FEATURE SUMMARY ==========\n")

    print("Shape:", features.shape)

    print("\nFeatures:")
    print(features.columns.tolist())

    print("\nDisease prevalence:")

    prevalence = (
        features
        .drop(columns=["PATIENT"])
        .sum()
        .sort_values(ascending=False)
    )

    print(prevalence.to_string())

    print("\nSaved to:")
    print(output)


if __name__ == "__main__":
    main()