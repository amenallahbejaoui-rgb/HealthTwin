import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/processed/mortality_model_dataset.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["DEATH_30D"])
y = df["DEATH_30D"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# ============================================================
# FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    ))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ============================================================
# RANDOM FOREST
# ============================================================

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


# ============================================================
# TRAIN
# ============================================================

print("Training Random Forest...")

pipeline.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# TRANSFORM DATA FOR SHAP
# ============================================================

X_test_transformed = pipeline.named_steps[
    "preprocessor"
].transform(X_test)

rf_model = pipeline.named_steps["model"]


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()


# ============================================================
# SHAP
# ============================================================

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test_transformed)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):
    values = shap_values[1]
else:
    values = shap_values

if values.ndim == 3:
    values = values[:, :, 1]


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

importance = np.abs(values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": importance
})

importance_df = importance_df.sort_values(
    "mean_abs_shap",
    ascending=False
)

print("\n========== TOP FEATURES ==========\n")

print(
    importance_df.head(20).to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

importance_df.to_csv(
    "data/processed/shap_feature_importance.csv",
    index=False
)

print(
    "\nSaved: data/processed/shap_feature_importance.csv"
)