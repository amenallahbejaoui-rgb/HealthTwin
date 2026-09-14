from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


CODES = [
    "2160-0", "38483-4",      # Creatinine
    "2345-7", "2339-0",       # Glucose
    "2951-2", "2947-0",       # Sodium
    "2823-3",                 # Potassium
    "3094-0", "6299-2",       # Urea nitrogen
    "2075-0", "2069-3",       # Chloride
    "17861-6", "49765-1",     # Calcium
]


def main():

    observations = pd.read_csv(
        RAW_DIR / "observations.csv",
        usecols=[
            "DATE",
            "PATIENT",
            "CODE",
            "DESCRIPTION",
            "VALUE",
            "UNITS",
            "TYPE",
        ]
    )

    observations["VALUE_NUMERIC"] = pd.to_numeric(
        observations["VALUE"],
        errors="coerce"
    )

    selected = observations[
        observations["CODE"].astype(str).isin(CODES)
    ].copy()

    print("\n========== LAB CODE VARIANTS ==========\n")

    summary = (
        selected
        .groupby(
            ["CODE", "DESCRIPTION", "TYPE", "UNITS"],
            dropna=False
        )
        .agg(
            observations=("CODE", "size"),
            patients=("PATIENT", "nunique"),
            numeric_values=("VALUE_NUMERIC", "count"),
            minimum=("VALUE_NUMERIC", "min"),
            maximum=("VALUE_NUMERIC", "max"),
            mean=("VALUE_NUMERIC", "mean"),
        )
        .reset_index()
        .sort_values(["DESCRIPTION", "CODE"])
    )

    print(summary.to_string(index=False))

    print("\n========== SAMPLE VALUES ==========\n")

    for code in CODES:

        subset = selected[
            selected["CODE"].astype(str) == code
        ].copy()

        if subset.empty:
            continue

        print(
            f"\n--- {code} | "
            f"{subset['DESCRIPTION'].iloc[0]} ---"
        )

        print(
            subset[
                [
                    "VALUE",
                    "UNITS",
                    "TYPE"
                ]
            ]
            .drop_duplicates()
            .head(15)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()