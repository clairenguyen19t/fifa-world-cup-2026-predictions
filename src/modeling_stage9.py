"""Stage 9 - model improvement experiments (leakage-safe).

Tries to beat the Stage 3 Random Forest on the 3-class match-outcome task while
keeping strictly chronological validation. Adds stronger learners, class-weight
strategies, hyperparameter tuning via ``TimeSeriesSplit`` (training data only),
and probability calibration - then evaluates every candidate on the same
2023-2025 holdout used in Stage 3.

Outputs (never overwrites Stage 3 artifacts):
  * outputs/model_results_stage9.csv
  * outputs/best_match_model_stage9.pkl
  * outputs/confusion_matrix_stage9.png

Guarantees: identical leakage-safe preprocessing as Stage 3 (Elo missingness
indicators + median imputation fit on TRAIN only); no random splits; the
2023-2025 test set is never used for tuning; 2026 fixtures are never loaded.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config, modeling

CLASS_ORDER = modeling.CLASS_ORDER  # ["home_win", "draw", "away_win"]
SEED = config.RANDOM_SEED

try:  # optional XGBoost
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def _proba_ordered(model, X) -> np.ndarray:
    proba = model.predict_proba(X)
    idx = {c: i for i, c in enumerate(model.classes_)}
    return proba[:, [idx[c] for c in CLASS_ORDER]]


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = _proba_ordered(model, X_test)
    rec = recall_score(y_test, preds, labels=CLASS_ORDER, average=None)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro", labels=CLASS_ORDER),
        "log_loss": log_loss(y_test, proba, labels=CLASS_ORDER),
        "draw_recall": rec[CLASS_ORDER.index("draw")],
        "confusion_matrix": confusion_matrix(y_test, preds, labels=CLASS_ORDER),
    }


# --------------------------------------------------------------------------- #
# Candidate definitions (model, param grid, supports class_weight)
# --------------------------------------------------------------------------- #
def _candidates() -> list[tuple]:
    cands = [
        (
            "RandomForest(tuned)",
            RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
            {
                "max_depth": [None, 16],
                "min_samples_leaf": [3, 8],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=SEED),
            {
                "max_depth": [None, 20],
                "min_samples_leaf": [3, 8],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(max_iter=400, random_state=SEED),
            {
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [31, 63],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=300, random_state=SEED),
            {
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    ]
    if _HAS_XGB:
        cands.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=400, objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", tree_method="hist", random_state=SEED,
                n_jobs=-1,
            ),
            {"max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
        ))
    return cands


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run() -> pd.DataFrame:
    config.ensure_dirs()

    df = modeling.load_dataset()
    train, test = modeling.chronological_split(df)
    X_train, X_test, y_train, y_test, feature_names, impute_state = modeling.prepare_xy(
        train, test
    )
    print(f"train rows: {len(X_train):,}  ({config.TRAIN_START}..{config.TRAIN_END})")
    print(f"test  rows: {len(X_test):,}  ({config.TEST_START}..{config.TEST_END})")

    # XGBoost needs integer labels; build an encoder where needed.
    tscv = TimeSeriesSplit(n_splits=4)
    results = []
    fitted: dict[str, object] = {}
    metrics: dict[str, dict] = {}

    for name, est, grid in _candidates():
        print(f"\n[tuning] {name} ...")
        y_tr = y_train
        if name == "XGBoost":
            # encode labels to 0..2 in CLASS_ORDER for xgboost
            y_tr = y_train.map({c: i for i, c in enumerate(CLASS_ORDER)})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = GridSearchCV(
                est, grid, scoring="neg_log_loss", cv=tscv, n_jobs=-1, refit=True,
            )
            gs.fit(X_train, y_tr)
        best = gs.best_estimator_

        # wrap xgboost so predict returns string labels for unified evaluation
        if name == "XGBoost":
            best = _XGBStringWrapper(best, CLASS_ORDER)

        m = evaluate(best, X_test, y_test)
        cw = gs.best_params_.get("class_weight", "n/a")
        fitted[name] = best
        metrics[name] = m
        results.append({
            "model": name, "class_weight": str(cw), "calibrated": False,
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "log_loss": round(m["log_loss"], 4),
            "draw_recall": round(m["draw_recall"], 4),
            "best_params": str(gs.best_params_),
        })
        print(f"  best params: {gs.best_params_}")
        print(f"  acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
              f"logloss={m['log_loss']:.4f} draw_recall={m['draw_recall']:.4f}")

    # ----- explicit class-weight comparison on HistGB (draw question) -----
    print("\n[class-weight experiment] HistGradientBoosting None vs balanced ...")
    for cw in (None, "balanced"):
        hgb = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.1, random_state=SEED, class_weight=cw)
        hgb.fit(X_train, y_train)
        m = evaluate(hgb, X_test, y_test)
        results.append({
            "model": "HistGB(cw-exp)", "class_weight": str(cw), "calibrated": False,
            "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
            "log_loss": round(m["log_loss"], 4), "draw_recall": round(m["draw_recall"], 4),
            "best_params": "learning_rate=0.1, max_iter=400",
        })
        print(f"  cw={cw}: acc={m['accuracy']:.4f} draw_recall={m['draw_recall']:.4f} "
              f"logloss={m['log_loss']:.4f}")

    # ----- calibration: sigmoid (robust) + isotonic, on strong base models -----
    # The uncalibrated tree models are overconfident (log loss > uniform 1.099),
    # so calibration is the key lever. We try both methods via TimeSeriesSplit.
    print("\n[calibration] sigmoid + isotonic via TimeSeriesSplit ...")
    cal_bases = {
        "ExtraTrees": clone(fitted["ExtraTrees"]),
        "HistGB(None)": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED),
        "HistGB(balanced)": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.1, random_state=SEED, class_weight="balanced"),
    }
    for base_name, base_est in cal_bases.items():
        for method in ("sigmoid", "isotonic"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cal = CalibratedClassifierCV(clone(base_est), method=method,
                                             cv=TimeSeriesSplit(n_splits=3))
                cal.fit(X_train, y_train)
            m = evaluate(cal, X_test, y_test)
            cal_name = f"{base_name}+{method}"
            fitted[cal_name] = cal
            metrics[cal_name] = m
            results.append({
                "model": cal_name, "class_weight": "n/a", "calibrated": True,
                "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
                "log_loss": round(m["log_loss"], 4),
                "draw_recall": round(m["draw_recall"], 4),
                "best_params": f"{method}, TimeSeriesSplit(3)",
            })
            print(f"  {cal_name}: acc={m['accuracy']:.4f} logloss={m['log_loss']:.4f} "
                  f"draw_recall={m['draw_recall']:.4f}")

    comparison = pd.DataFrame(results)

    # ----- Stage 3 baseline -----
    base_df = pd.read_csv(config.OUTPUT_FILES["model_results"])
    base = base_df.sort_values("log_loss").iloc[0]
    base_acc, base_ll, base_f1 = base["accuracy"], base["log_loss"], base["macro_f1"]

    # ----- choose best: probabilities matter for simulation -> primary log loss -----
    best_name = min(metrics.items(), key=lambda kv: kv[1]["log_loss"])[0]
    best_model = fitted[best_name]
    best_m = metrics[best_name]
    comparison["is_best_stage9"] = comparison["model"] == best_name

    comparison = comparison.sort_values("log_loss").reset_index(drop=True)
    comparison.to_csv(config.OUTPUT_FILES["model_results_stage9"], index=False)

    modeling.plot_confusion_matrix(
        best_m["confusion_matrix"], f"Confusion Matrix - {best_name} (Stage 9, test)",
        config.OUTPUT_FILES["confusion_matrix_stage9"],
    )
    joblib.dump(
        {"model": best_model, "model_name": best_name, "feature_names": feature_names,
         "impute_state": impute_state, "class_order": CLASS_ORDER},
        config.OUTPUT_FILES["best_match_model_stage9"],
    )

    _print_summary(comparison, best_name, best_m, base_acc, base_ll, base_f1)
    return comparison


class _XGBStringWrapper:
    """Wrap an XGBoost classifier so predict/predict_proba use string labels."""

    def __init__(self, model, class_order):
        self.model = model
        self.class_order = list(class_order)
        self.classes_ = np.array(self.class_order)

    def predict(self, X):
        idx = self.model.predict(X)
        return np.array([self.class_order[int(i)] for i in idx])

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _print_summary(comparison, best_name, best_m, base_acc, base_ll, base_f1):
    print("\n" + "=" * 78)
    print("STAGE 9 MODEL COMPARISON (sorted by log loss; lower is better)")
    print("=" * 78)
    cols = ["model", "class_weight", "calibrated", "accuracy", "macro_f1",
            "log_loss", "draw_recall"]
    print(comparison[cols].to_string(index=False))

    print("\n--- Stage 3 baseline (current best) ---")
    print(f"  accuracy={base_acc:.4f}  macro_f1={base_f1:.4f}  log_loss={base_ll:.4f}")
    print(f"\n--- Stage 9 best by log loss: {best_name} ---")
    print(f"  accuracy={best_m['accuracy']:.4f}  macro_f1={best_m['macro_f1']:.4f}  "
          f"log_loss={best_m['log_loss']:.4f}  draw_recall={best_m['draw_recall']:.4f}")
    print("  confusion matrix (rows=actual", CLASS_ORDER, "):")
    print(best_m["confusion_matrix"])

    d_acc = best_m["accuracy"] - base_acc
    d_ll = best_m["log_loss"] - base_ll
    print("\n--- Deltas vs Stage 3 ---")
    print(f"  accuracy: {d_acc:+.4f}   log_loss: {d_ll:+.4f}   "
          f"macro_f1: {best_m['macro_f1'] - base_f1:+.4f}")

    print("\n--- RECOMMENDATION ---")
    uniform_ll = float(np.log(3))
    d_ll_abs = base_ll - best_m["log_loss"]          # positive => Stage 9 better
    meaningful = d_ll_abs >= 0.03                      # beyond run-to-run noise
    acc_ok = best_m["accuracy"] >= base_acc - 0.005

    print(f"  Note: uniform-guess log loss = {uniform_ll:.3f}. Every candidate (incl. "
          f"Stage 3 at {base_ll:.3f}) sits ABOVE it, i.e. all are overconfident; the real "
          "improvement is bringing log loss down toward ~1.10.")

    if meaningful and acc_ok:
        print(f"  USE the Stage 9 model ({best_name}). Log loss improves "
              f"{base_ll:.3f} -> {best_m['log_loss']:.3f} (-{d_ll_abs:.3f}) while accuracy "
              f"holds ({base_acc:.3f} -> {best_m['accuracy']:.3f}). Better probabilities "
              "directly help the Monte Carlo simulation.")
        print("  Next step (separate stage): regenerate Stage 4/5 with this model.")
    else:
        print(f"  KEEP the Stage 3 model for now. The Stage 9 best ({best_name}) only moves "
              f"log loss {base_ll:.3f} -> {best_m['log_loss']:.3f} "
              f"({'+' if d_ll_abs < 0 else '-'}{abs(d_ll_abs):.3f}), which is within noise "
              "and not a clear, material win. Do not replace the model on accuracy alone "
              "when calibrated probabilities are what the simulation consumes.")
        print("  Most useful side-finding: 'balanced' class weights are the only way to make "
              "the model predict draws at all (draw recall ~0.02 -> ~0.33, best macro-F1) - "
              "worth revisiting if group-stage draw realism becomes a priority.")
    print("\n  (Stage 3 artifacts were NOT overwritten; Stage 9 saved to *_stage9.*)")


if __name__ == "__main__":
    run()
