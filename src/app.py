import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="HealthTwin",
    page_icon="🏥",
    layout="wide"
)

DATA_PATH = "data/processed/mortality_model_dataset.csv"
COHORT_PATH = "data/processed/covid_cohort.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_cohort():
    return pd.read_csv(COHORT_PATH, usecols=["PATIENT"])


@st.cache_resource
def train_model(df):
    X = df.drop(columns=["DEATH_30D", "PATIENT"], errors="ignore")
    y = df["DEATH_30D"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        exclude=["object", "str"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ])

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


# ============================================================
# PAGE
# ============================================================

st.title("🏥 HealthTwin")

st.subheader("Explainable Patient Risk Analysis")

st.write(
    "A research and educational prototype using synthetic "
    "healthcare data to analyze patient risk."
)

st.divider()


# ============================================================
# DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data()

if not os.path.exists(COHORT_PATH):
    st.error(f"Cohort file not found: {COHORT_PATH}")
    st.stop()

cohort = load_cohort()

if len(cohort) != len(df):
    st.error(
        "The cohort and model dataset contain different numbers of patients. "
        "Rebuild the processed datasets before running the app."
    )
    st.stop()

df.insert(0, "PATIENT", cohort["PATIENT"].astype(str).to_numpy())

model = train_model(df)


# ============================================================
# PATIENT SELECTION
# ============================================================

patient_ids = df["PATIENT"].astype(str).tolist()

selected_patient = st.selectbox(
    "Select patient",
    patient_ids
)

patient = df[
    df["PATIENT"].astype(str) == selected_patient
].iloc[0]


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.divider()

st.header("👤 Patient Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Age",
        f"{patient['AGE']:.1f}"
    )

with col2:
    st.metric(
        "Gender",
        str(patient["GENDER"])
    )

with col3:
    st.metric(
        "Conditions",
        int(patient["condition_count"])
    )

with col4:
    st.metric(
        "Condition events",
        int(patient["condition_event_count"])
    )


# ============================================================
# RISK PREDICTION
# ============================================================

X_patient = pd.DataFrame(
    [patient.drop(labels=["DEATH_30D"])]
)

# Remove target-related information if present
X_patient = X_patient.drop(
    columns=["DEATH_30D"],
    errors="ignore"
)

probability = model.predict_proba(X_patient)[0, 1]

st.divider()

st.header("🧠 30-Day Mortality Risk")

risk_col1, risk_col2 = st.columns([1, 2])

with risk_col1:
    st.metric(
        "Estimated risk",
        f"{probability * 100:.1f}%"
    )

with risk_col2:
    st.progress(float(probability))

st.caption(
    "Model output from a Random Forest trained on synthetic "
    "Synthea healthcare records. This is not a clinical diagnosis."
)


# ============================================================
# TOP MODEL FEATURES
# ============================================================

st.divider()

st.header("🔎 Model Features")

rf_model = model.named_steps["model"]
preprocessor = model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
).head(10)

# Clean feature names
importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

st.bar_chart(
    importance_df.set_index("Feature")
)


# ============================================================
# PATIENT CLINICAL VALUES
# ============================================================

st.divider()

st.header("📊 Clinical Information")

clinical_columns = [
    "AGE",
    "systolic_bp_latest",
    "diastolic_bp_latest",
    "heart_rate_latest",
    "respiratory_rate_latest",
    "weight_latest",
    "height_latest",
    "bmi_latest",
    "creatinine_latest",
    "glucose_latest",
    "sodium_latest",
    "calcium_latest",
]

available_columns = [
    col for col in clinical_columns
    if col in df.columns
]

clinical_data = pd.DataFrame({
    "Feature": available_columns,
    "Value": [
        patient[col]
        for col in available_columns
    ]
})

st.dataframe(
    clinical_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CONDITIONS
# ============================================================

st.divider()

st.header("🩺 Previous Conditions")

condition_columns = [
    col for col in df.columns
    if col.startswith("has_")
]

conditions = []

for col in condition_columns:
    if patient[col] == 1:
        name = (
            col
            .replace("has_", "")
            .replace("_", " ")
            .title()
        )
        conditions.append(name)

if conditions:
    for condition in conditions:
        st.write(f"• {condition}")
else:
    st.write("No tracked pre-COVID conditions.")


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.header("📁 Dataset")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Patients", len(df))

with c2:
    st.metric("Features", len(df.columns) - 1)

with c3:
    st.metric(
        "30-Day deaths",
        int(df["DEATH_30D"].sum())
    )

st.caption(
    "HealthTwin uses synthetic Synthea data for research "
    "and educational purposes."
)
