from pathlib import Path
import pandas as pd

from load import load_all


PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def prepare_events(data):

    events = []

    # ---------------------------------------------------------
    # CONDITIONS
    # ---------------------------------------------------------
    conditions = data["conditions"][
        ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]
    ].copy()

    conditions = conditions.rename(
        columns={
            "START": "DATE",
            "PATIENT": "PATIENT_ID",
            "ENCOUNTER": "ENCOUNTER_ID",
        }
    )

    conditions["EVENT_TYPE"] = "condition"

    events.append(
        conditions[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # MEDICATIONS
    # ---------------------------------------------------------
    medications = data["medications"][
        ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]
    ].copy()

    medications = medications.rename(
        columns={
            "START": "DATE",
            "PATIENT": "PATIENT_ID",
            "ENCOUNTER": "ENCOUNTER_ID",
        }
    )

    medications["EVENT_TYPE"] = "medication"

    events.append(
        medications[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # PROCEDURES
    # ---------------------------------------------------------
    procedures = data["procedures"][
        ["DATE", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]
    ].copy()

    procedures = procedures.rename(
        columns={
            "PATIENT": "PATIENT_ID",
            "ENCOUNTER": "ENCOUNTER_ID",
        }
    )

    procedures["EVENT_TYPE"] = "procedure"

    events.append(
        procedures[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # OBSERVATIONS
    # ---------------------------------------------------------
    observations = data["observations"][
        [
            "DATE",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
            "VALUE",
        ]
    ].copy()

    observations = observations.rename(
        columns={
            "PATIENT": "PATIENT_ID",
            "ENCOUNTER": "ENCOUNTER_ID",
        }
    )

    observations["EVENT_TYPE"] = "observation"

    events.append(
        observations[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # ENCOUNTERS
    # ---------------------------------------------------------
    encounters = data["encounters"][
        [
            "START",
            "PATIENT",
            "Id",
            "CODE",
            "DESCRIPTION",
        ]
    ].copy()

    encounters = encounters.rename(
        columns={
            "START": "DATE",
            "PATIENT": "PATIENT_ID",
            "Id": "ENCOUNTER_ID",
        }
    )

    encounters["EVENT_TYPE"] = "encounter"

    events.append(
        encounters[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # CARE PLANS
    # ---------------------------------------------------------
    careplans = data["careplans"][
        [
            "START",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
        ]
    ].copy()

    careplans = careplans.rename(
        columns={
            "START": "DATE",
            "PATIENT": "PATIENT_ID",
            "ENCOUNTER": "ENCOUNTER_ID",
        }
    )

    careplans["EVENT_TYPE"] = "careplan"

    events.append(
        careplans[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # IMAGING
    # ---------------------------------------------------------
    imaging = data["imaging_studies"][
        [
            "DATE",
            "PATIENT",
            "ENCOUNTER",
            "BODYSITE_DESCRIPTION",
            "MODALITY_DESCRIPTION",
        ]
    ].copy()

    imaging["DESCRIPTION"] = (
        imaging["MODALITY_DESCRIPTION"].fillna("")
        + " - "
        + imaging["BODYSITE_DESCRIPTION"].fillna("")
    )

    imaging["CODE"] = None

    imaging = imaging.rename(
        columns={
            "PATIENT": "PATIENT_ID",
            "ENCOUNTER": "ENCOUNTER_ID",
        }
    )

    imaging["EVENT_TYPE"] = "imaging"

    events.append(
        imaging[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # IMMUNIZATIONS
    # ---------------------------------------------------------
    immunizations = data["immunizations"][
        [
            "DATE",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
        ]
    ].copy()

    immunizations = immunizations.rename(
        columns={
            "PATIENT": "PATIENT_ID",
            "ENCOUNTER": "ENCOUNTER_ID",
        }
    )

    immunizations["EVENT_TYPE"] = "immunization"

    events.append(
        immunizations[
            [
                "PATIENT_ID",
                "DATE",
                "EVENT_TYPE",
                "CODE",
                "DESCRIPTION",
                "ENCOUNTER_ID",
            ]
        ]
    )

    # ---------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------
    timeline = pd.concat(events, ignore_index=True)

    timeline["DATE"] = pd.to_datetime(
        timeline["DATE"],
        errors="coerce"
    )

    timeline = timeline.dropna(
        subset=["PATIENT_ID", "DATE"]
    )

    timeline = timeline.sort_values(
        ["PATIENT_ID", "DATE"]
    ).reset_index(drop=True)

    return timeline


def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data = load_all()

    timeline = prepare_events(data)

    output = PROCESSED_DIR / "patient_timeline.csv"

    timeline.to_csv(
        output,
        index=False
    )

    print("\n========== TIMELINE ==========")

    print("Rows:", len(timeline))
    print("Patients:", timeline["PATIENT_ID"].nunique())

    print("\nEvents by type:")

    print(
        timeline["EVENT_TYPE"]
        .value_counts()
    )

    print("\nDate range:")

    print("From:", timeline["DATE"].min())
    print("To:", timeline["DATE"].max())

    print("\nSaved to:")
    print(output)


if __name__ == "__main__":
    main()