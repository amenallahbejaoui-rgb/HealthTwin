from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


def evaluate_model(name, model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    # Default threshold = 0.5
    predictions = (probabilities >= 0.5).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0
        )
    )

    return {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    path = DATA_DIR / "mortality_model_dataset.csv"

    df = pd.read_csv(path)

    target = "DEATH_30D"

    X = df.drop(columns=[target])
    y = df[target]

    print("\n========== DATA ==========\n")

    print("Rows:", len(df))
    print("Features:", X.shape[1])

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nMortality rate:", y.mean())

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("\n========== SPLIT ==========\n")

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    print("\nTrain target:")
    print(y_train.value_counts())

    print("\nTest target:")
    print(y_test.value_counts())

    # ---------------------------------------------------------
    # Column types
    # ---------------------------------------------------------

    numeric_features = [
        c for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c])
    ]

    categorical_features = [
        c for c in X.columns
        if c not in numeric_features
    ]

    print("\nCategorical features:")
    print(categorical_features)

    print("\nNumeric features:", len(numeric_features))

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                ),
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # ---------------------------------------------------------
    # Model 1: Logistic Regression
    # ---------------------------------------------------------

    logistic = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic.fit(
        X_train,
        y_train
    )

    results = []

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic,
            X_test,
            y_test,
        )
    )

    # ---------------------------------------------------------
    # Model 2: Random Forest
    # ---------------------------------------------------------

    rf_numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            )
        ]
    )

    rf_preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                rf_numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessing",
                rf_preprocessing
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced",
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest.fit(
        X_train,
        y_train
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_test,
            y_test,
        )
    )

    # ---------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON ==========\n")

    print(
        results_df
        .sort_values(
            "pr_auc",
            ascending=False
        )
        .to_string(index=False)
    )

    output = DATA_DIR / "baseline_results.csv"

    results_df.to_csv(
        output,
        index=False
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()