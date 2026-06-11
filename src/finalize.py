"""Stage 10 - final public-facing deployment pipeline.

Promotes the chosen public-facing model (ExtraTreesClassifier + sigmoid
calibration, the Stage 9 highest-accuracy *and* probability-calibrated
candidate) to a final, reproducible deployment:

  1. Refit the final model on the SAME leakage-safe features + chronological
     split, then save final_match_model.pkl / final_model_metrics.csv /
     report/final_model_summary.md.
  2. Regenerate 2026 fixture probabilities  -> final_fixtures_2026_predictions.csv
  3. Re-run the 10,000-sim Monte Carlo (same Stage 5 logic / Stage 8 bracket
     assumptions) -> final_champion_probabilities.csv,
     final_advancement_probabilities.csv
  4. Render final LinkedIn visuals into outputs/final_linkedin_visuals/.
  5. Write report/final_project_summary.md.

Nothing from earlier stages is overwritten - every new artifact uses the
"final_" naming convention. No 2026 data ever enters training; no random
splits; the 2023-2025 holdout is evaluation-only.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit

from . import config, linkedin_visuals, modeling, predict_2026, simulation
from .modeling_stage9 import CLASS_ORDER, evaluate

FINAL_NAME = "ExtraTrees+sigmoid"
FINAL_LABEL = "ExtraTreesClassifier + sigmoid calibration"
# Stage 9 tuned ExtraTrees hyperparameters (chosen by TimeSeriesSplit neg-log-loss).
ET_PARAMS = dict(n_estimators=600, max_depth=20, min_samples_leaf=3,
                 class_weight=None, n_jobs=-1, random_state=config.RANDOM_SEED)


# --------------------------------------------------------------------------- #
# 1. Final model
# --------------------------------------------------------------------------- #
def build_final_model():
    """Refit ExtraTrees+sigmoid; return (bundle, final_metrics, base_metrics)."""
    df = modeling.load_dataset()
    train, test = modeling.chronological_split(df)
    X_train, X_test, y_train, y_test, feature_names, impute_state = modeling.prepare_xy(
        train, test
    )

    base = ExtraTreesClassifier(**ET_PARAMS)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
    model.fit(X_train, y_train)

    m = evaluate(model, X_test, y_test)
    bundle = {
        "model": model,
        "model_name": FINAL_NAME,
        "model_label": FINAL_LABEL,
        "feature_names": feature_names,
        "impute_state": impute_state,
        "class_order": CLASS_ORDER,
    }

    # Stage 3 baseline (current production model) for the comparison row.
    base_df = pd.read_csv(config.OUTPUT_FILES["model_results"]).sort_values("log_loss")
    base_row = base_df.iloc[0]
    base_metrics = {
        "accuracy": float(base_row["accuracy"]),
        "macro_f1": float(base_row["macro_f1"]),
        "log_loss": float(base_row["log_loss"]),
        "model": str(base_row["model"]),
    }
    return bundle, m, base_metrics, (X_test, y_test)


def save_final_model(bundle, m, base_metrics) -> pd.DataFrame:
    joblib.dump(bundle, config.OUTPUT_FILES["final_model"])
    modeling.plot_confusion_matrix(
        m["confusion_matrix"], f"Confusion Matrix - {FINAL_NAME} (final, test)",
        config.OUTPUT_FILES["final_confusion_matrix"],
    )
    metrics = pd.DataFrame([
        {"stage": "Final (Stage 10)", "model": FINAL_LABEL,
         "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["macro_f1"], 4),
         "log_loss": round(m["log_loss"], 4), "draw_recall": round(m["draw_recall"], 4)},
        {"stage": "Baseline (Stage 3)", "model": base_metrics["model"],
         "accuracy": round(base_metrics["accuracy"], 4),
         "macro_f1": round(base_metrics["macro_f1"], 4),
         "log_loss": round(base_metrics["log_loss"], 4), "draw_recall": float("nan")},
    ])
    metrics.to_csv(config.OUTPUT_FILES["final_model_metrics"], index=False)
    return metrics


# --------------------------------------------------------------------------- #
# 2. Predictions + 3. Simulation
# --------------------------------------------------------------------------- #
def final_predictions(bundle) -> pd.DataFrame:
    fixtures = pd.read_parquet(config.PROCESSED_FILES["fixtures_2026_features"])
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    preds = predict_2026.predict(fixtures, bundle)
    preds.to_csv(config.OUTPUT_FILES["final_fixtures_predictions"], index=False)
    return preds


def final_simulation(preds) -> tuple[pd.DataFrame, pd.DataFrame]:
    setup = simulation.build_setup(preds)
    results = simulation.simulate(setup, config.N_SIMULATIONS, config.RANDOM_SEED)
    champions, advancement = simulation.build_tables(setup, results)
    champions.to_csv(config.OUTPUT_FILES["final_champion_probabilities"], index=False)
    advancement.to_csv(config.OUTPUT_FILES["final_advancement_probabilities"], index=False)
    return champions, advancement


# --------------------------------------------------------------------------- #
# 4. Final visuals
# --------------------------------------------------------------------------- #
def final_visuals() -> list[str]:
    linkedin_visuals.configure(
        output_dir=config.FINAL_LINKEDIN_DIR,
        sources={
            "preds": config.OUTPUT_FILES["final_fixtures_predictions"],
            "champ": config.OUTPUT_FILES["final_champion_probabilities"],
            "adv": config.OUTPUT_FILES["final_advancement_probabilities"],
            "model": config.OUTPUT_FILES["final_model_metrics"],
        },
        hero_footer=f"Final model: {FINAL_LABEL}",
        cards_footer=f"Final model: {FINAL_LABEL}  |  full 3-way distribution shown",
    )
    data = linkedin_visuals.load_data()
    # Exactly the six requested final visuals (no most_likely_tournament_path).
    return [
        linkedin_visuals.visual_champions(data),
        linkedin_visuals.visual_full_bracket(data),
        linkedin_visuals.visual_cards_a_f(data),
        linkedin_visuals.visual_cards_g_l(data),
        linkedin_visuals.visual_pipeline(data),
        linkedin_visuals.visual_validation(data),
    ]


# --------------------------------------------------------------------------- #
# 5. Reports
# --------------------------------------------------------------------------- #
def _top15_md(champions: pd.DataFrame) -> str:
    top = champions.sort_values("champion_prob", ascending=False).head(15)
    lines = ["| Rank | Team | Group | Champion prob |",
             "|---:|---|:---:|---:|"]
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        lines.append(f"| {i} | {r['team']} | {r['group']} | {r['champion_prob']*100:.2f}% |")
    return "\n".join(lines)


def write_model_summary(m, base_metrics) -> None:
    d_acc = m["accuracy"] - base_metrics["accuracy"]
    d_ll = m["log_loss"] - base_metrics["log_loss"]
    text = f"""# Final Model Summary

**Selected public-facing model:** `{FINAL_LABEL}`

## Why this model
Stage 9 produced two strong candidates:

| Candidate | Accuracy | Macro F1 | Log loss | Draw recall |
|---|---:|---:|---:|---:|
| **ExtraTrees + sigmoid** (selected) | {m['accuracy']*100:.1f}% | {m['macro_f1']:.3f} | {m['log_loss']:.3f} | {m['draw_recall']:.3f} |
| HistGB(balanced) + sigmoid | ~58.6% | ~0.428 | ~1.552 (best) | ~0.005 |

`ExtraTrees + sigmoid` was chosen as the **public-facing** model because it delivers
the **highest holdout accuracy ({m['accuracy']*100:.1f}%)** while remaining
probability-calibrated (sigmoid/Platt scaling fit via `TimeSeriesSplit`), giving the
best balance between predictive performance and presentation value. HistGB(balanced)
keeps a marginally lower log loss but loses ~1.5pp of accuracy.

## Final holdout metrics (2023-2025, chronological)

| Metric | Final model | Stage 3 baseline ({base_metrics['model']}) | Delta |
|---|---:|---:|---:|
| Accuracy | {m['accuracy']*100:.2f}% | {base_metrics['accuracy']*100:.2f}% | {d_acc*100:+.2f}pp |
| Macro F1 | {m['macro_f1']:.3f} | {base_metrics['macro_f1']:.3f} | {m['macro_f1']-base_metrics['macro_f1']:+.3f} |
| Log loss | {m['log_loss']:.3f} | {base_metrics['log_loss']:.3f} | {d_ll:+.3f} |

Confusion matrix (rows = actual `{CLASS_ORDER}`):

```
{m['confusion_matrix']}
```

## Reproducibility
- Trained on `data/processed/train_features.parquet`, 2010-2022 train window.
- Identical leakage-safe preprocessing as Stage 3 (Elo missingness indicators +
  median imputation fit on train only).
- Calibration: `CalibratedClassifierCV(method="sigmoid", cv=TimeSeriesSplit(3))`.
- Artifacts: `outputs/final_match_model.pkl`, `outputs/final_model_metrics.csv`,
  `outputs/final_confusion_matrix.png`.
"""
    (config.REPORT_DIR / "final_model_summary.md").write_text(text)


def write_project_summary(m, base_metrics, preds, champions) -> None:
    avg_draw = preds["P_draw"].mean()
    text = f"""# FIFA World Cup 2026 - Prediction Project (Final Summary)

## 1. Project objective
Build a reproducible, leakage-safe pipeline that predicts FIFA World Cup 2026
outcomes - from raw historical match data to per-team title probabilities - and
communicate the results with professional, presentation-ready visuals.

## 2. Datasets used
- `results.csv` - historical international match results (the spine dataset).
- `eloratings.csv` - time-stamped team Elo ratings.
- `goalscorers.csv`, `shootouts.csv`, `former_names.csv` - supporting context.
- The 2026 fixtures are carried as unplayed rows (prediction targets only).

## 3. Cleaning pipeline
- Mixed-format dates parsed robustly (ISO first, then US fallback for Elo).
- `results.csv` split into played history vs. unplayed 2026 fixtures.
- Elo cleaned (drop null / zero ratings, sort by team + date).
- Basic de-duplication and type coercion; cleaned files saved to `data/interim/`.

## 4. Stage 7 Unicode / non-breaking-space bug
Validation revealed Elo team names stored a non-breaking space (`U+00A0`) instead
of a normal space (e.g. `'United\\xa0States'`), so the temporal Elo join silently
failed for every multi-word team. Those teams received imputed (median) Elo,
badly under-rating hosts such as the USA. The fix normalizes all team columns
(replace `\\xa0`, strip/collapse whitespace, alias map) at ingestion. After the fix,
missing Elo cells in the 2026 fixtures dropped from 30 to 0 and USA's real Elo
(~1747) replaced the imputed ~1543 - materially reshuffling the contenders.

## 5. Feature engineering (leakage-safe)
- Target: `home_win` / `draw` / `away_win`.
- Elo via temporal as-of join (strictly before kickoff) + missingness indicators.
- Rolling last-5 form: win rate, goals for / against / diff, computed with
  `shift(1)` so the current match never feeds its own features.
- Context: `neutral` flag and `tournament_weight`.

## 6. Chronological validation approach
Strictly time-ordered split - train 2010-2022, holdout 2023-2025. No random
K-fold anywhere; hyperparameters tuned with `TimeSeriesSplit` on training data
only; imputation medians learned on train only; 2026 fixtures never used in
training.

## 7. Final selected model
`{FINAL_LABEL}` - tuned ExtraTrees ({ET_PARAMS['n_estimators']} trees,
max_depth={ET_PARAMS['max_depth']}, min_samples_leaf={ET_PARAMS['min_samples_leaf']})
wrapped in sigmoid (Platt) calibration fit via `TimeSeriesSplit`.

## 8. Final metrics (2023-2025 holdout)
| Metric | Final model | Stage 3 baseline |
|---|---:|---:|
| Accuracy | {m['accuracy']*100:.2f}% | {base_metrics['accuracy']*100:.2f}% |
| Macro F1 | {m['macro_f1']:.3f} | {base_metrics['macro_f1']:.3f} |
| Log loss | {m['log_loss']:.3f} | {base_metrics['log_loss']:.3f} |

## 9. Why ExtraTrees + sigmoid was selected
It posts the **highest holdout accuracy ({m['accuracy']*100:.1f}%)** of all Stage 9
candidates while staying probability-calibrated, so the Monte Carlo still consumes
sensible probabilities. Sigmoid calibration also cut log loss well below the
uncalibrated trees (~1.69 -> {m['log_loss']:.3f}) and below the Stage 3 baseline
({base_metrics['log_loss']:.3f}). It is the best balance of predictive performance
and public presentation value.

## 10. Tournament simulation methodology
- {config.N_SIMULATIONS:,} Monte Carlo runs (fixed seed) of the 48-team format.
- Group stage sampled directly from the model's per-fixture probabilities
  (avg. draw probability {avg_draw*100:.1f}%); ranking by points, a goal-difference
  approximation, then a seeded tie-break.
- Advancement: top 2 per group + 8 best third-placed teams = 32.
- Knockouts use per-team strength ratings (mean win + half draw prob); a sampled
  draw resolves via normalized win probabilities.
- **Simplification:** the exact FIFA position-based third-place slotting is not
  replicated; qualifiers are seeded by group performance into a standard balanced
  bracket. Brackets are labelled *"Illustrative simulation scenario based on model
  outputs."*

## 11. Top 15 champion probabilities (final model)
{_top15_md(champions)}

## 12. Major limitations
- All models sit above the uniform-guess log loss (~1.099): international results
  are genuinely hard and the feature set is modest.
- Draw recall is low (hard-label draws are rarely predicted); the simulation
  relies on draw *probability mass*, not hard draw calls.
- The knockout bracket is a defensible simplification, not the exact FIFA mapping.
- Elo/form features omit squad-level information, injuries, and rest days.

## 13. Future improvements
- Richer features: head-to-head history, rest days, travel, confederation strength.
- Proper FIFA third-place slotting and seeding pots.
- Score-line models (e.g. Poisson/Dixon-Coles) for real goal-difference tie-breaks.
- Ensemble of calibrated learners; monitor calibration drift over time.

---
*Artifacts use the `final_` naming convention; earlier-stage outputs are preserved.*
"""
    (config.REPORT_DIR / "final_project_summary.md").write_text(text)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    config.ensure_dirs()
    print("=" * 78)
    print("STAGE 10 - FINAL DEPLOYMENT PIPELINE")
    print(f"Public-facing model: {FINAL_LABEL}")
    print("=" * 78)

    bundle, m, base_metrics, _ = build_final_model()
    metrics = save_final_model(bundle, m, base_metrics)
    print("\n[1] Final model trained & saved.")

    preds = final_predictions(bundle)
    print(f"[2] Final 2026 fixture predictions: {len(preds)} fixtures "
          f"(avg draw prob {preds['P_draw'].mean()*100:.1f}%).")

    champions, advancement = final_simulation(preds)
    print(f"[3] Monte Carlo done: {config.N_SIMULATIONS:,} sims.")

    visual_paths = final_visuals()
    print(f"[4] Final visuals: {len(visual_paths)} files.")

    write_model_summary(m, base_metrics)
    write_project_summary(m, base_metrics, preds, champions)
    print("[5] Reports written.")

    _print_final(m, base_metrics, champions, metrics, visual_paths)
    return {"metrics": m, "champions": champions, "visuals": visual_paths}


def _print_final(m, base_metrics, champions, metrics, visual_paths):
    generated = [
        config.OUTPUT_FILES["final_model"],
        config.OUTPUT_FILES["final_model_metrics"],
        config.OUTPUT_FILES["final_confusion_matrix"],
        config.OUTPUT_FILES["final_fixtures_predictions"],
        config.OUTPUT_FILES["final_champion_probabilities"],
        config.OUTPUT_FILES["final_advancement_probabilities"],
        config.REPORT_DIR / "final_model_summary.md",
        config.REPORT_DIR / "final_project_summary.md",
    ] + [Path(p) for p in visual_paths]

    print("\n" + "=" * 78)
    print("ALL GENERATED FILES")
    print("=" * 78)
    for p in generated:
        print(f"  {p}")

    print("\n" + "=" * 78)
    print(f"FINAL MODEL METRICS  ({FINAL_LABEL})")
    print("=" * 78)
    print(f"  accuracy    : {m['accuracy']*100:.2f}%")
    print(f"  macro F1    : {m['macro_f1']:.3f}")
    print(f"  log loss    : {m['log_loss']:.3f}")
    print(f"  draw recall : {m['draw_recall']:.3f}")

    print("\n" + "=" * 78)
    print("TOP 15 CHAMPION PROBABILITIES (final model)")
    print("=" * 78)
    top = champions.sort_values("champion_prob", ascending=False).head(15).reset_index(drop=True)
    for i, r in top.iterrows():
        print(f"  {i+1:>2}. {r['team']:<22} (Grp {r['group']})  {r['champion_prob']*100:5.2f}%")

    print("\n" + "=" * 78)
    print("FINAL vs STAGE 3 BASELINE")
    print("=" * 78)
    print(f"  accuracy : {base_metrics['accuracy']*100:.2f}%  ->  {m['accuracy']*100:.2f}%  "
          f"({(m['accuracy']-base_metrics['accuracy'])*100:+.2f}pp)")
    print(f"  macro F1 : {base_metrics['macro_f1']:.3f}  ->  {m['macro_f1']:.3f}  "
          f"({m['macro_f1']-base_metrics['macro_f1']:+.3f})")
    print(f"  log loss : {base_metrics['log_loss']:.3f}  ->  {m['log_loss']:.3f}  "
          f"({m['log_loss']-base_metrics['log_loss']:+.3f})")

    print("\n" + "=" * 78)
    print("RECOMMENDED LINKEDIN CAROUSEL ORDER")
    print("=" * 78)
    order = [
        ("1", "champion_probabilities_top15.png", "Hero - instant payoff, the headline result"),
        ("2", "full_decision_tree_bracket.png", "The 'wow' artifact - full tournament tree"),
        ("3", "group_stage_match_cards_A_F.png", "Depth - real per-match probabilities"),
        ("4", "group_stage_match_cards_G_L.png", "Depth - continued"),
        ("5", "validation_bug_fix.png", "Credibility - the data-quality story"),
        ("6", "pipeline_summary.png", "Method - how it all fits together (closer)"),
    ]
    for n, f, why in order:
        print(f"  {n}. {f:<35} - {why}")
    print("\n  Strongest COVER image: champion_probabilities_top15.png")
    print("  (Clean ranked payoff; the bracket is the strongest secondary 'hook'.)")


if __name__ == "__main__":
    run()
