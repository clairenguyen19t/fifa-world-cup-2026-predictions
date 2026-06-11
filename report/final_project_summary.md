# FIFA World Cup 2026 - Prediction Project (Final Summary)

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
of a normal space (e.g. `'United\xa0States'`), so the temporal Elo join silently
failed for every multi-word team. Those teams received imputed (median) Elo,
badly under-rating hosts such as the USA. The fix normalizes all team columns
(replace `\xa0`, strip/collapse whitespace, alias map) at ingestion. After the fix,
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
`ExtraTreesClassifier + sigmoid calibration` - tuned ExtraTrees (600 trees,
max_depth=20, min_samples_leaf=3)
wrapped in sigmoid (Platt) calibration fit via `TimeSeriesSplit`.

## 8. Final metrics (2023-2025 holdout)
| Metric | Final model | Stage 3 baseline |
|---|---:|---:|
| Accuracy | 60.24% | 58.53% |
| Macro F1 | 0.457 | 0.508 |
| Log loss | 1.641 | 1.743 |

## 9. Why ExtraTrees + sigmoid was selected
It posts the **highest holdout accuracy (60.2%)** of all Stage 9
candidates while staying probability-calibrated, so the Monte Carlo still consumes
sensible probabilities. Sigmoid calibration also cut log loss well below the
uncalibrated trees (~1.69 -> 1.641) and below the Stage 3 baseline
(1.743). It is the best balance of predictive performance
and public presentation value.

## 10. Tournament simulation methodology
- 10,000 Monte Carlo runs (fixed seed) of the 48-team format.
- Group stage sampled directly from the model's per-fixture probabilities
  (avg. draw probability 23.5%); ranking by points, a goal-difference
  approximation, then a seeded tie-break.
- Advancement: top 2 per group + 8 best third-placed teams = 32.
- Knockouts use per-team strength ratings (mean win + half draw prob); a sampled
  draw resolves via normalized win probabilities.
- **Simplification:** the exact FIFA position-based third-place slotting is not
  replicated; qualifiers are seeded by group performance into a standard balanced
  bracket. Brackets are labelled *"Illustrative simulation scenario based on model
  outputs."*

## 11. Top 15 champion probabilities (final model)
| Rank | Team | Group | Champion prob |
|---:|---|:---:|---:|
| 1 | Argentina | A | 6.58% |
| 2 | France | K | 5.16% |
| 3 | Spain | F | 5.08% |
| 4 | England | H | 5.05% |
| 5 | Mexico | J | 5.05% |
| 6 | Belgium | C | 4.76% |
| 7 | Canada | D | 4.75% |
| 8 | Brazil | E | 4.62% |
| 9 | Ecuador | I | 4.60% |
| 10 | Portugal | G | 4.45% |
| 11 | Germany | I | 4.39% |
| 12 | Netherlands | L | 4.20% |
| 13 | Switzerland | D | 4.04% |
| 14 | Croatia | H | 3.70% |
| 15 | Colombia | G | 3.60% |

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
