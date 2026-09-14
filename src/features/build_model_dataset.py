from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def main():

    clinical = pd.read_csv(
        PROCESSED_DIR / "mortality_features_v2.csv"
    )

    conditions = pd.read_csv(
        PROCESSED_DIR / "condition_features.csv"
    )

    print("\n========== INPUTS ==========\n")
    print("Clinical:", clinical.shape)
    print("Conditions:", conditions.shape)

    # ---------------------------------------------------------
    # Remove useless / zero-information condition feature
    # ---------------------------------------------------------

    conditions = conditions.drop(
        columns=["has_thrombotic_history"],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # Merge onto ALL COVID patients
    # ---------------------------------------------------------

    df = clinical.merge(
        conditions,
        on="PATIENT",
        how="left"
    )

    # ---------------------------------------------------------
    # Patients without recorded pre-COVID conditions
    # ---------------------------------------------------------

    condition_cols = [
        c for c in conditions.columns
        if c != "PATIENT"
    ]

    df[condition_cols] = df[condition_cols].fillna(0)

    # ---------------------------------------------------------
    # Leakage / audit columns
    # ---------------------------------------------------------

    target = "DEATH_30D"

    excluded = [
        "PATIENT",
        "COVID_DATE",
        "DEATHDATE",
        "BIRTHDATE",
        "DAYS_TO_DEATH",
        "DIED",
    ]

    # ---------------------------------------------------------
    # Acute COVID biomarkers
    # ---------------------------------------------------------

    acute_features = [
        "crp",
        "d_dimer",
        "ferritin",
        "ldh",
        "procalcitonin",
        "troponin",
        "oxygen_saturation",
        "temperature",
        "potassium",
    ]

    # Remove every generated feature belonging to those groups.
    acute_prefixes = tuple(
        feature + "_"
        for feature in acute_features
    )

    model_features = []

    for col in df.columns:

        if col in excluded:
            continue

        if col == target:
            continue

        if col.startswith(acute_prefixes):
            continue

        model_features.append(col)

    # ---------------------------------------------------------
    # Remove extremely sparse clinical variables
    #
    # Keep features with at least 20% patient coverage.
    # ---------------------------------------------------------

    coverage = (
        df[model_features]
        .notna()
        .mean()
    )

    sparse_features = coverage[
        coverage < 0.20
    ].index.tolist()

    model_features = [
        c for c in model_features
        if c not in sparse_features
    ]

    # ---------------------------------------------------------
    # Build final dataset
    # ---------------------------------------------------------

    final_columns = model_features + [target]

    model_df = df[final_columns].copy()

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    print("\n========== MODEL DATASET ==========\n")

    print("Rows:", len(model_df))
    print("Features:", len(model_features))

    print("\nTarget:")
    print(
        model_df[target]
        .value_counts()
        .sort_index()
    )

    print("\nMissingness:")
    print(
        model_df
        .isna()
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nExcluded sparse features:", len(sparse_features))

    print("\nFinal features:")

    for feature in model_features:
        print(" -", feature)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output = PROCESSED_DIR / "mortality_model_dataset.csv"

    model_df.to_csv(
        output,
        index=False
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()