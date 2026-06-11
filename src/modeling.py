"""Stage 3 - match outcome model training.

Trains baseline classifiers (Logistic Regression, Random Forest) to predict the
3-way match outcome (home_win / draw / away_win) from the leakage-safe features
produced in Stage 2.

Design choices that preserve the no-leakage guarantee:
  * Only the explicitly approved ``config.MODEL_FEATURES`` are used as inputs.
  * A strictly chronological train/test split (no shuffling).
  * Missing-value handling (Elo missingness indicators + median imputation) is
    fit on the TRAIN split only and then applied to TEST.
  * The 2026 fixtures are never loaded here.

No XGBoost and no tournament simulation in this stage.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# Fixed class order so every metric / matrix / probability column lines up.
CLASS_ORDER = ["home_win", "draw", "away_win"]


# --------------------------------------------------------------------------- #
# Data loading & chronological split
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    """Load the Stage 2 training feature table with parsed dates."""
    df = pd.read_parquet(config.PROCESSED_FILES["train_features"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into train/test windows by date (no shuffling)."""
    train_mask = (df["date"] >= config.TRAIN_START) & (df["date"] <= config.TRAIN_END)
    test_mask = (df["date"] >= config.TEST_START) & (df["date"] <= config.TEST_END)
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


# --------------------------------------------------------------------------- #
# Missing-value handling (fit on train only)
# --------------------------------------------------------------------------- #
def prepare_xy(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], dict]:
    """Build model matrices with Elo missingness indicators + median imputation.

    Returns ``(X_train, X_test, y_train, y_test, feature_names, impute_state)``.
    The imputation medians and feature list are computed from TRAIN only so they
    can be reused at inference time.
    """
    features = list(config.MODEL_FEATURES)

    X_train = train[features].copy()
    X_test = test[features].copy()

    # Explicit missingness indicators for Elo columns (built before imputation).
    indicator_cols: list[str] = []
    for col in config.ELO_FEATURES:
        if col in X_train.columns:
            ind = f"{col}_missing"
            X_train[ind] = X_train[col].isna().astype("int64")
            X_test[ind] = X_test[col].isna().astype("int64")
            indicator_cols.append(ind)

    # Median imputation learned on TRAIN only.
    medians = X_train[features].median(numeric_only=True)
    X_train[features] = X_train[features].fillna(medians)
    X_test[features] = X_test[features].fillna(medians)

    feature_names = features + indicator_cols
    X_train = X_train[feature_names]
    X_test = X_test[feature_names]

    y_train = train["result"].astype("object")
    y_test = test["result"].astype("object")

    impute_state = {
        "features": features,
        "indicator_cols": indicator_cols,
        "feature_names": feature_names,
        "medians": medians.to_dict(),
    }
    return X_train, X_test, y_train, y_test, feature_names, impute_state


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models() -> dict[str, Pipeline]:
    """Return the candidate models. LogReg is scaled; RF needs no scaling."""
    logreg = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )
    rf = Pipeline(
        steps=[
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=config.RANDOM_SEED,
                ),
            )
        ]
    )
    return {"LogisticRegression": logreg, "RandomForest": rf}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute accuracy, macro F1, log loss and a confusion matrix."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    # Align probability columns to CLASS_ORDER for a stable log loss.
    class_index = {c: i for i, c in enumerate(model.classes_)}
    proba_ordered = proba[:, [class_index[c] for c in CLASS_ORDER]]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", labels=CLASS_ORDER),
        "log_loss": log_loss(y_test, proba_ordered, labels=CLASS_ORDER),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASS_ORDER),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, path) -> bool:
    """Save a confusion-matrix heatmap. Returns False if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}")
        return False

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> pd.DataFrame:
    """Train, evaluate, persist artifacts; return the comparison table."""
    config.ensure_dirs()

    df = load_dataset()
    train, test = chronological_split(df)
    X_train, X_test, y_train, y_test, feature_names, impute_state = prepare_xy(train, test)

    print(f"train rows: {len(X_train):,}  ({config.TRAIN_START} .. {config.TRAIN_END})")
    print(f"test  rows: {len(X_test):,}  ({config.TEST_START} .. {config.TEST_END})")
    print(f"features ({len(feature_names)}): {feature_names}\n")

    results = []
    fitted: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        fitted[name] = model
        metrics[name] = m
        results.append(
            {
                "model": name,
                "accuracy": round(m["accuracy"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "log_loss": round(m["log_loss"], 4),
            }
        )
        print(f"[{name}] confusion matrix (rows=actual {CLASS_ORDER}):")
        print(m["confusion_matrix"], "\n")

    comparison = pd.DataFrame(results)

    # Best model: lowest log loss (probability quality matters for simulation).
    best_name = comparison.sort_values("log_loss").iloc[0]["model"]

    comparison["is_best"] = comparison["model"] == best_name
    comparison.to_csv(config.OUTPUT_FILES["model_results"], index=False)

    plotted = plot_confusion_matrix(
        metrics[best_name]["confusion_matrix"],
        f"Confusion Matrix - {best_name} (test)",
        config.OUTPUT_FILES["confusion_matrix"],
    )

    # Persist the best model together with its preprocessing state.
    joblib.dump(
        {
            "model": fitted[best_name],
            "model_name": best_name,
            "feature_names": feature_names,
            "impute_state": impute_state,
            "class_order": CLASS_ORDER,
        },
        config.OUTPUT_FILES["best_model"],
    )

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_name} (selected by lowest log loss)")
    print(f"Saved comparison -> {config.OUTPUT_FILES['model_results']}")
    if plotted:
        print(f"Saved confusion  -> {config.OUTPUT_FILES['confusion_matrix']}")
    print(f"Saved best model -> {config.OUTPUT_FILES['best_model']}")

    return comparison


if __name__ == "__main__":
    run()
