# Stage 7 — Elo Team-Name Fix: Before/After Comparison

This report documents the fix for the Elo team-name bug found in Stage 6 and
compares the pipeline outputs before and after re-running Stages 1→5.

> The Stage 6 validation note (`report/model_validation_notes.md`) is left
> unchanged as the historical record of how the bug was discovered. This is a
> separate document.

## The bug (recap)

`eloratings.csv` stored multi-word team names using the **non-breaking space**
character `U+00A0` (e.g. `"United\xa0States"`, `"South\xa0Korea"`). The Stage-2
Elo as-of join matches team strings exactly, so **every multi-word team silently
failed to match** and was median-imputed (`home_elo ≈ 1543`). Two teams also had
genuine name differences (`Czechia`, `Democratic Republic of Congo`).

## The fix

In `src/ingest.py`, all team-name columns of every dataset are now normalized on
load via `normalize_name`:

1. replace `U+00A0` with a normal space,
2. collapse repeated whitespace and strip ends,
3. apply an alias map to a canonical spelling (`src/config.TEAM_ALIASES`):
   `Czechia→Czech Republic`, `Democratic Republic of Congo→DR Congo`,
   `Korea Republic→South Korea`, `Türkiye→Turkey`, `USA→United States`.

Applied to: `results` (home/away), `goalscorers` (home/away/team), `shootouts`
(home/away/winner/first_shooter), `eloratings` (team), `former_names`
(current/former). Stages 1→5 were then re-run end to end.

## 1. Missing Elo values in `fixtures_2026_features`

| | Home Elo missing | Away Elo missing | Distinct teams affected |
|---|---|---|---|
| **Before** | 13 | 17 | 10 |
| **After** | **0** | **0** | **0** |

Training-set Elo coverage also improved markedly: `home_elo` missing fell from
12,338 → 5,638 and `away_elo` from 12,306 → 5,952 (remaining gaps are genuinely
pre-Elo-era or rarely-rated historical sides, not name mismatches).

The ten previously-broken teams now resolve to real ratings (imputed median was
~1543):

| Team | Resolved Elo |
|---|---|
| South Korea | 1784 |
| United States | 1747 |
| Czech Republic | 1731 |
| DR Congo | 1616 |
| Saudi Arabia | 1612 |
| Ivory Coast | 1607 |
| New Zealand | 1586 |
| Bosnia and Herzegovina | 1571 |
| Cape Verde | 1560 |
| South Africa | 1531 |

## 2. Host champion probabilities

| Host | Before % | After % | Change |
|---|---|---|---|
| United States | 0.19 | **0.34** | +0.15 (≈ +80%) |
| Canada | 4.84 | 4.32 | −0.52 |
| Mexico | 4.18 | **4.86** | +0.68 |

The USA's Elo was corrected from the imputed 1543 to its real **1747**, and its
title probability roughly doubled (rank 40 → 36 of 48). It remains mid-tier: 1747
is only moderately above average in a 48-team field, and Group B (Turkey,
Australia, Paraguay) is competitive — so the *previously unreasonable* USA result
is now a defensible model output rather than a bug artifact. Canada dips slightly
because its group rivals (notably Switzerland) and knockout opponents are now
rated correctly; Mexico rises as its Group J rivals (South Korea, Czech Republic,
South Africa) are no longer over-rated by imputation.

## 3. Top-15 champion probabilities

| Rank | Before | % | After | % |
|---|---|---|---|---|
| 1 | Argentina | 6.32 | Argentina | 6.31 |
| 2 | England | 5.28 | **Spain** | 5.93 |
| 3 | France | 5.20 | **Germany** | 5.27 |
| 4 | Belgium | 4.92 | England | 5.23 |
| 5 | Canada | 4.84 | **Mexico** | 4.86 |
| 6 | Brazil | 4.80 | France | 4.57 |
| 7 | Switzerland | 4.70 | Brazil | 4.53 |
| 8 | Netherlands | 4.58 | Switzerland | 4.43 |
| 9 | Spain | 4.53 | Canada | 4.32 |
| 10 | Germany | 4.36 | Belgium | 4.32 |
| 11 | Mexico | 4.18 | Netherlands | 4.23 |
| 12 | Portugal | 4.16 | Portugal | 3.75 |
| 13 | Croatia | 3.82 | **Ecuador** | 3.71 |
| 14 | Colombia | 3.66 | Croatia | 3.67 |
| 15 | Japan | 3.10 | Colombia | 3.29 |

(Model also retrained: RandomForest remained best; test accuracy 0.563 → 0.585,
log loss 1.63 → 1.74.)

## Did the fix materially change results? — **Yes.**

- **Missing Elo eliminated** in the 2026 fixtures (30 cells → 0); the headline
  symptom is gone.
- **The top-15 reshuffled meaningfully**: Spain 9→2, Germany 10→3, Mexico 11→5,
  Ecuador entered the top 15, Japan dropped out. These shifts are driven both by
  the corrected ratings of previously-imputed teams *and* by their group rivals
  no longer being distorted.
- **The USA anomaly is resolved in cause** (real Elo restored) even though its
  probability stays modest, which is now reasonable rather than a bug.

The changes are material and corrective, not cosmetic.

## Teams still missing Elo in the 2026 fixtures

**None.** All 48 teams in `fixtures_2026_features` now have both `home_elo` and
`away_elo` populated from the Elo dataset.
