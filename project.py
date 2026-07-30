"""
Heart Disease Prediction - Machine Learning Project
=====================================================
Dataset: UCI Heart Disease Dataset (Cleveland, 14 attributes)
Goal: Predict presence of heart disease (target: 0 = no disease, 1 = disease)

Pipeline:
1. Load data (via ucimlrepo, with a CSV fallback)
2. Explore & clean data
3. Preprocess (encode, scale, split)
4. Train multiple models
5. Evaluate & compare
6. Save the best model
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    """Load the UCI Heart Disease dataset. Tries the official ucimlrepo
    package first; falls back to a hosted CSV mirror if that isn't
    available (e.g. no internet / package not installed)."""
    try:
        from ucimlrepo import fetch_ucirepo
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        print("Data loaded via ucimlrepo (official UCI source).")
    except Exception as e:
        print(f"ucimlrepo failed ({e}); falling back to CSV mirror...")
        url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
        df = pd.read_csv(url)
        print("Data loaded via CSV fallback.")

    # In the raw UCI data, target can be 0-4 (severity). Convert to binary:
    # 0 = no disease, 1 = disease present (any of 1-4).
    df["target"] = (df["target"] > 0).astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. EXPLORE & CLEAN
# ---------------------------------------------------------------------------
def explore_and_clean(df):
    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Missing values ---")
    print(df.isnull().sum())

    # UCI data sometimes stores missing values as '?' strings in ca/thal
    df = df.replace("?", np.nan)
    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill remaining missing values with column median
    df = df.fillna(df.median(numeric_only=True))

    print("\n--- Target distribution ---")
    print(df["target"].value_counts())

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()
    print("\nSaved correlation_heatmap.png")

    return df


# ---------------------------------------------------------------------------
# 3. PREPROCESS
# ---------------------------------------------------------------------------
def preprocess(df):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


# ---------------------------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

        results.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1 Score": f1, "ROC-AUC": auc, "CV Accuracy": cv_score
        })
        trained_models[name] = model

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    return results_df, trained_models


# ---------------------------------------------------------------------------
# 5. PLOTS FOR BEST MODEL
# ---------------------------------------------------------------------------
def plot_best_model(best_model, best_name, X_test, y_test):
    y_pred = best_model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ROC curve
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{best_name} (AUC = {auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

    print("\nSaved confusion_matrix.png and roc_curve.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    df = explore_and_clean(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    models = get_models()
    results_df, trained_models = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print("\n\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    plot_best_model(best_model, best_name, X_test, y_test)

    # Save best model + scaler for later use (e.g. in a web app)
    joblib.dump(best_model, "heart_disease_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved heart_disease_model.pkl and scaler.pkl")

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("Saved model_comparison_results.csv")


if __name__ == "__main__":
    main()