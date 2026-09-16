from pathlib import Path
import pandas as pd


RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    print(f"{name:25} {len(df):>8} rows | {len(df.columns):>3} columns")

    return df


def load_all():
    files = [
        "allergies.csv",
        "careplans.csv",
        "conditions.csv",
        "devices.csv",
        "encounters.csv",
        "imaging_studies.csv",
        "immunizations.csv",
        "medications.csv",
        "observations.csv",
        "organizations.csv",
        "patients.csv",
        "payer_transitions.csv",
        "payers.csv",
        "procedures.csv",
        "providers.csv",
        "supplies.csv",
    ]

    data = {}

    for file in files:
        data[file.replace(".csv", "")] = load_csv(file)

    return data


if __name__ == "__main__":
    data = load_all()

    print("\nLoaded datasets:")
    for name, df in data.items():
        print(f"- {name}: {df.shape}")