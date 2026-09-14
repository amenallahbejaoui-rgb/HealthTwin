from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def main():

    path = PROCESSED_DIR / "mortality_features_v2.csv"

    df = pd.read_csv(path)

    print("\n========== FEATURE COVERAGE ==========\n")

    target = "DEATH_30D"

    excluded = {
        "PATIENT",
        "COVID_DATE",
        "DEATHDATE",
        "BIRTHDATE",
        "DAYS_TO_DEATH",
        "DIED",
        target,
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    coverage = []

    for col in feature_cols:

        non_null = df[col].notna().sum()
        total = len(df)

        coverage.append({
            "feature": col,
            "non_null": non_null,
            "missing": total - non_null,
            "coverage_pct": 100 * non_null / total,
        })

    coverage = pd.DataFrame(coverage)

    coverage = coverage.sort_values(
        "coverage_pct",
        ascending=True
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    print("\n========== COVERAGE GROUPS ==========\n")

    print(
        ">= 90%:",
        (coverage["coverage_pct"] >= 90).sum()
    )

    print(
        ">= 75%:",
        (coverage["coverage_pct"] >= 75).sum()
    )

    print(
        ">= 50%:",
        (coverage["coverage_pct"] >= 50).sum()
    )

    print(
        ">= 25%:",
        (coverage["coverage_pct"] >= 25).sum()
    )

    print(
        "< 25%:",
        (coverage["coverage_pct"] < 25).sum()
    )

    output = PROCESSED_DIR / "feature_coverage.csv"

    coverage.to_csv(
        output,
        index=False
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()