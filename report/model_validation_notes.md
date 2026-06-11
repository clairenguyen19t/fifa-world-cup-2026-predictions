# Stage 6 — Model Validation & Sanity Checks

Validation of the World Cup 2026 pipeline outputs (Stages 1–5). No model was
retrained and the simulation was not rebuilt; this is a read-only review of the
artifacts in `outputs/`. One **clear data bug** was found and is documented below
(flagged for a future fix, not applied in this stage).

Artifacts reviewed: `outputs/fixtures_2026_predictions.csv`,
`outputs/champion_probabilities.csv`, `outputs/advancement_probabilities.csv`,
`data/processed/fixtures_2026_features.parquet`, `outputs/best_match_model.pkl`,
`data/interim/eloratings_clean.parquet`.

---

## 1. Top-20 champion probabilities

| Rank | Team | Group | Champion % |
|---|---|---|---|
| 1 | Argentina | A | 6.32 |
| 2 | England | H | 5.28 |
| 3 | France | K | 5.20 |
| 4 | Belgium | C | 4.92 |
| 5 | Canada | D | 4.84 |
| 6 | Brazil | E | 4.80 |
| 7 | Switzerland | D | 4.70 |
| 8 | Netherlands | L | 4.58 |
| 9 | Spain | F | 4.53 |
| 10 | Germany | I | 4.36 |
| 11 | Mexico | J | 4.18 |
| 12 | Portugal | G | 4.16 |
| 13 | Croatia | H | 3.82 |
| 14 | Colombia | G | 3.66 |
| 15 | Japan | L | 3.10 |
| 16 | Ecuador | I | 3.01 |
| 17 | Turkey | B | 2.78 |
| 18 | Norway | K | 2.76 |
| 19 | Iran | C | 2.34 |
| 20 | Morocco | E | 2.12 |

The set of contenders is football-plausible (Argentina, England, France, Brazil,
Spain, Germany, Netherlands, Portugal all near the top). The distribution is
**flat** — the favorite sits at only ~6.3% — which is reasonable for a 48-team
field with a deliberately conservative 3-way model, but see §7.

## 2. Group-winner probabilities (all 12 groups)

| Group | Ordering (group-winner %) |
|---|---|
| A | Argentina 70, Austria 18, Algeria 9, Jordan 3 |
| B | Turkey 40, Australia 26, Paraguay 26, United States 8 |
| C | Belgium 59, Iran 26, Egypt 11, New Zealand 4 |
| D | Switzerland 47, Canada 43, Bosnia & H. 8, Qatar 2 |
| E | Brazil 57, Morocco 24, Scotland 15, Haiti 4 |
| F | Spain 57, Uruguay 18, Cape Verde 16, Saudi Arabia 9 |
| G | Portugal 44, Colombia 42, Uzbekistan 7, DR Congo 7 |
| H | England 57, Croatia 34, Panama 8, Ghana 1 |
| I | Germany 47, Ecuador 34, Ivory Coast 17, Curaçao 2 |
| J | Mexico 50, Czech Republic 24, South Korea 15, South Africa 11 |
| K | France 57, Norway 27, Senegal 13, Iraq 3 |
| L | Netherlands 53, Japan 34, Sweden 9, Tunisia 4 |

Within each group the favorite is sensible and orderings are monotone with team
quality. **Group B is the standout anomaly**: the USA (a host) is last at 8% —
see §3 and §6.

## 3. Host nations (Canada, USA, Mexico)

| Host | Group | Group-win % | Top-2 % | Reach QF % | Reach SF % | Champion % |
|---|---|---|---|---|---|---|
| Mexico | J | 49.8 | 78.5 | 27.4 | 14.9 | 4.18 |
| Canada | D | 42.9 | 81.6 | 29.6 | 16.3 | 4.84 |
| **United States** | B | **7.9** | **23.2** | **5.6** | **2.0** | **0.19** |

All three hosts are correctly given a genuine home advantage (`neutral = 0` in
their group matches, see §4), and Canada and Mexico land in the top tier as one
would expect for hosts. **The USA is the glaring outlier** — weakest of the three
hosts by an order of magnitude. This is **not** a host-advantage problem; it is a
direct consequence of the Elo bug in §6.

## 4. Neutral / home-advantage encoding for 2026 fixtures

- Fixture `neutral` flag: **63 neutral, 9 non-neutral**. The 9 non-neutral
  fixtures are exactly the three host nations' three group matches each — i.e.
  hosts get home advantage, all other group games are neutral. **This is correct.**
- Mean predicted probabilities:
  - Host (non-neutral) games: `P_home 0.414` vs `P_away 0.284` — a clear, correct
    home edge.
  - Neutral games: `P_home 0.372` vs `P_away 0.347` — a **small residual
    ~2.5pp bias** toward whichever team is arbitrarily listed as "home".

**Verdict:** neutral encoding is essentially correct. The minor residual home-team
bias on neutral games is a limitation (arbitrary home/away labelling still leaks a
little home-advantage signal the model learned historically), not a bug.

## 5. Does the simplified bracket give some teams an easier path?

- Correlation between team **strength rating** and **champion probability** is
  **0.955** — champion odds are almost entirely explained by team strength.
- No low-strength team receives an anomalously high champion probability. The
  largest positive gap between strength rank and champion rank is **Canada
  (+5)**, fully explained by its host advantage inflating group performance and
  therefore seeding — expected, not spurious.

**Verdict:** the simplified, performance-seeded balanced bracket is **not**
manufacturing easy paths. Its limitation is the opposite — by reseeding every
simulation it cannot reproduce the *fixed* real-world bracket's imbalances
("group of death" path effects). That is a modeling simplification, documented in
the README, not a bug.

## 6. Missing Elo in 2026 fixtures + imputation impact — **BUG FOUND**

Ten of the 48 teams have **no Elo** attached in the fixture features and were
median-imputed (`home_elo = 1543`, `away_elo = 1524`):

`United States, South Korea, South Africa, Czech Republic, Ivory Coast,
Cape Verde, DR Congo, Saudi Arabia, New Zealand, Bosnia and Herzegovina`.

**Root cause (clear bug):** the team names in `eloratings_clean` use the
**non-breaking space** character `U+00A0` instead of a normal space — e.g.
`"United\xa0States"`, `"South\xa0Korea"`. The Stage-2 Elo as-of join matches on
exact team strings, so **every multi-word team name silently fails to match** and
falls through to median imputation. (Two teams — `Czech Republic` → `Czechia`,
`DR Congo` → `Democratic Republic of Congo` — additionally have genuine name
differences.)

**Impact:** the affected teams are all pushed toward an average-or-below rating,
so they are systematically **under-rated**. The most visible symptom is the USA:
as a host it should be a strong side, but with an imputed mediocre Elo it collapses
to a 0.19% title chance and last in its group. This is an **implementation/data
bug**, not a model limitation.

**Fix (for a future version, not applied here):** normalize whitespace
(`str.replace("\u00a0", " ").str.strip()`) when loading `eloratings.csv` in
`src/ingest.py` / `src/clean.py`, and add a small alias map (`Czechia ↔ Czech
Republic`, `Democratic Republic of Congo ↔ DR Congo`, etc.). Then re-run
Stages 1→5. This was deliberately **not** done in Stage 6 per the instruction not
to retrain or rebuild the simulation.

## 7. Model vs. simple Elo-difference expectation

For the 46 fixtures where both teams have a real Elo:

- `corr(elo_diff, P_home_win − P_away_win) = 0.977`
- `corr(elo_win_expectancy, P_home_win) = 0.966`

The model is **highly consistent** with a pure Elo expectation, which is strong
evidence it learned the dominant signal correctly. Where they diverge, the model
is **more conservative** than Elo (e.g. Brazil vs Haiti: Elo 0.93 vs model 0.65;
Argentina vs Algeria: 0.90 vs 0.63), because ~28% of probability mass goes to
draws in the 3-way model and the random forest regresses extremes. This is
expected behaviour, not an error.

## 8. Obviously unreasonable outputs — bug vs. limitation

| Observation | Reasonable? | Cause |
|---|---|---|
| USA weakest host (0.19% title) | **No** | **BUG** — non-breaking-space Elo join failure (§6) |
| Other 9 multi-word teams under-rated | **No** | **BUG** — same as above |
| Flat champion distribution (top ~6%) | Borderline | Limitation — conservative 3-way model + draws |
| ~2.5pp home bias on neutral games | Minor | Limitation — arbitrary home/away labelling |
| Bracket can't reproduce fixed-draw path effects | Acceptable | Documented simplification |
| Favorites less dominant than Elo implies | Yes | Expected 3-way/draw behaviour |

---

## Summary

**What looks reasonable**
- Contender set, group-winner orderings, and host treatment for Canada and Mexico.
- Correct neutral/home-advantage flagging (hosts non-neutral, rest neutral).
- Very high agreement with Elo expectations (r ≈ 0.97), confirming the model
  learned the right primary signal.
- The simplified bracket is fair — champion odds track strength (r = 0.955) with
  no spurious easy paths.

**What looks suspicious**
- **The USA (and 9 other multi-word-named teams) are under-rated** due to a clear
  Elo team-name bug (non-breaking spaces). This is the single most important
  finding and the main driver of unreasonable outputs.
- A small residual home-advantage bias persists on neutral fixtures because of
  arbitrary home/away labelling.
- The champion distribution is quite flat; useful directionally but not yet
  sharp enough for confident point predictions.

**What should be improved in future versions**
1. **Fix the Elo team-name join** (strip `U+00A0`, add an alias map), then re-run
   the pipeline — highest priority, directly corrects ~10 teams including a host.
2. **Symmetrize neutral fixtures** — average predictions over both home/away
   orderings (or add an explicit `is_host` feature) to remove residual home bias.
3. **Improve discrimination** — add Elo as a continuous primary feature with
   better coverage, consider a calibrated gradient-boosted or Poisson scoreline
   model (Stage-0 plan), and calibrate probabilities (e.g. isotonic).
4. **More realistic knockout** — use a fixed bracket consistent with the official
   2026 slotting and compute knockout matchup probabilities from model features
   rather than derived strength ratings.
5. **Backtest** champion/advancement probabilities against historical tournaments
   to quantify calibration, not just match-level log loss.
