"""Stage 5 - Monte Carlo World Cup 2026 tournament simulation.

Consumes the Stage 4 fixture probabilities (``outputs/fixtures_2026_predictions.csv``)
and simulates the full 48-team tournament ``N_SIMULATIONS`` times to estimate, per
team, the probability of winning each group, advancing, reaching each knockout
round, and lifting the trophy.

Design (kept deliberately simple and transparent):
  * Groups are reconstructed from the fixture pairings (connected components).
  * Group stage: each of the 72 fixtures is sampled directly from its predicted
    [P_home_win, P_draw, P_away_win] - i.e. the model probabilities are used
    as-is.
  * Group ranking: points, then a goal-difference *approximation* (+1 win /
    0 draw / -1 loss accumulated), then a seeded random tie-breaker.
  * Advancement: top 2 per group (24) + 8 best third-placed teams = 32.
  * Knockout matchup probabilities are derived from per-team *strength ratings*
    (mean win + half draw probability over a team's group fixtures), because the
    model's fixture-level predictions only cover the group stage. A knockout
    match samples win/draw/loss; a sampled draw is resolved by the two teams'
    normalized win probabilities (Bradley-Terry on strengths).

SIMPLIFIED BRACKET LIMITATION: the exact FIFA 2026 position-based mapping of the
8 best third-placed teams into specific Round-of-32 slots is not replicated.
Instead, all 32 qualifiers are seeded by group-stage performance and placed into
a standard balanced single-elimination bracket. This is documented in the README.

No model retraining and no use of actual 2026 results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

POINTS_VEC = np.array(
    [config.POINTS_WIN, config.POINTS_DRAW, config.POINTS_LOSS], dtype=np.int64
)


# --------------------------------------------------------------------------- #
# Setup: teams, groups, fixtures, strengths
# --------------------------------------------------------------------------- #
def _connected_components(edges: list[tuple[str, str]]) -> list[set[str]]:
    """Return connected components (groups) from undirected team pairings."""
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen: set[str] = set()
    comps: list[set[str]] = []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(n for n in adj[x] if n not in seen)
        comps.append(comp)
    return comps


def build_setup(preds: pd.DataFrame) -> dict:
    """Build team indexing, group assignment, fixture arrays, and strengths.

    Returns a dict consumed by the simulator.
    """
    edges = list(zip(preds["home_team"], preds["away_team"]))
    comps = _connected_components(edges)

    # Prefer the official FIFA A-L letters: each group is labelled by the Pot 1
    # seed it contains. If every group resolves to a unique official letter we
    # order groups A..L by that letter; otherwise fall back to deterministic
    # alphabetical labelling (by each group's first team).
    seeds = config.OFFICIAL_GROUP_SEEDS

    def _official_label(comp: set[str]) -> str | None:
        for t in comp:
            if t in seeds:
                return seeds[t]
        return None

    labels = [_official_label(c) for c in comps]
    if len(set(labels)) == len(comps) and None not in labels:
        paired = sorted(zip(labels, comps), key=lambda x: x[0])
        group_labels = [lab for lab, _ in paired]
        comps_sorted = [c for _, c in paired]
    else:
        comps_sorted = sorted(comps, key=lambda c: sorted(c)[0])
        group_labels = [chr(ord("A") + i) for i in range(len(comps_sorted))]

    teams: list[str] = []
    team_group: dict[str, str] = {}
    for label, comp in zip(group_labels, comps_sorted):
        for t in sorted(comp):
            teams.append(t)
            team_group[t] = label

    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # Per-team strength from group fixtures: mean(P_win) + 0.5 * mean(P_draw).
    win_sum = np.zeros(n_teams)
    draw_sum = np.zeros(n_teams)
    cnt = np.zeros(n_teams)
    for _, r in preds.iterrows():
        h, a = team_idx[r.home_team], team_idx[r.away_team]
        win_sum[h] += r.P_home_win
        draw_sum[h] += r.P_draw
        cnt[h] += 1
        win_sum[a] += r.P_away_win
        draw_sum[a] += r.P_draw
        cnt[a] += 1
    strength = (win_sum / cnt) + 0.5 * (draw_sum / cnt)
    strength = np.clip(strength, 1e-6, None)

    # Fixture arrays for vectorized group-stage sampling.
    fx_home = preds["home_team"].map(team_idx).to_numpy()
    fx_away = preds["away_team"].map(team_idx).to_numpy()
    fx_probs = preds[["P_home_win", "P_draw", "P_away_win"]].to_numpy()

    # Map each team to its row position within its group (groups of 4).
    group_members: dict[str, list[int]] = {g: [] for g in group_labels}
    for t in teams:
        group_members[team_group[t]].append(team_idx[t])
    group_matrix = np.array([group_members[g] for g in group_labels])  # (12, 4)

    return {
        "teams": teams,
        "team_idx": team_idx,
        "team_group": team_group,
        "group_labels": group_labels,
        "strength": strength,
        "fx_home": fx_home,
        "fx_away": fx_away,
        "fx_probs": fx_probs,
        "group_matrix": group_matrix,
        "n_teams": n_teams,
    }


# --------------------------------------------------------------------------- #
# Bracket helper
# --------------------------------------------------------------------------- #
def standard_bracket_order(n: int) -> list[int]:
    """Standard balanced single-elimination seed positions for ``n`` slots.

    Ensures seed 1 and seed 2 can only meet in the final, etc. Returns a list of
    seed indices (0 = top seed) arranged so adjacent pairs are first-round games.
    """
    order = [0]
    while len(order) < n:
        m = len(order) * 2
        new = []
        for x in order:
            new.append(x)
            new.append(m - 1 - x)
        order = new
    return order


# --------------------------------------------------------------------------- #
# Vectorized simulation
# --------------------------------------------------------------------------- #
def _sample_outcomes(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Map uniform draws to outcome index 0/1/2 given a probability triple."""
    c1 = probs[0]
    c2 = probs[0] + probs[1]
    out = np.full(u.shape, 2, dtype=np.int8)
    out[u < c2] = 1
    out[u < c1] = 0
    return out


def simulate(setup: dict, n_sims: int, seed: int) -> dict:
    """Run ``n_sims`` vectorized tournament simulations; return count tables."""
    rng = np.random.default_rng(seed)
    n_teams = setup["n_teams"]
    strength = setup["strength"]
    group_matrix = setup["group_matrix"]  # (G, 4)
    n_groups = group_matrix.shape[0]

    points = np.zeros((n_sims, n_teams), dtype=np.int64)
    gd = np.zeros((n_sims, n_teams), dtype=np.int64)

    # --- Group stage: sample every fixture across all sims ---
    for f in range(len(setup["fx_home"])):
        h = setup["fx_home"][f]
        a = setup["fx_away"][f]
        u = rng.random(n_sims)
        out = _sample_outcomes(setup["fx_probs"][f], u)  # 0 home,1 draw,2 away
        home_win = out == 0
        draw = out == 1
        away_win = out == 2
        points[home_win, h] += config.POINTS_WIN
        points[away_win, h] += config.POINTS_LOSS
        points[draw, h] += config.POINTS_DRAW
        points[away_win, a] += config.POINTS_WIN
        points[home_win, a] += config.POINTS_LOSS
        points[draw, a] += config.POINTS_DRAW
        gd[home_win, h] += 1
        gd[away_win, h] -= 1
        gd[away_win, a] += 1
        gd[home_win, a] -= 1

    # --- Rank within each group (points, gd approx, random tie-break) ---
    # Composite key per team: large weight on points, then gd, then jitter.
    jitter = rng.random((n_sims, n_teams))
    key = points.astype(np.float64) * 1e6 + gd.astype(np.float64) * 1e3 + jitter

    group_winner = np.zeros((n_sims, n_groups), dtype=np.int64)
    runner_up = np.zeros((n_sims, n_groups), dtype=np.int64)
    third = np.zeros((n_sims, n_groups), dtype=np.int64)
    third_key = np.zeros((n_sims, n_groups), dtype=np.float64)
    seed_score = np.zeros((n_sims, n_teams), dtype=np.float64)
    top2_mask = np.zeros((n_sims, n_teams), dtype=bool)
    winner_mask = np.zeros((n_sims, n_teams), dtype=bool)

    rows = np.arange(n_sims)
    for g in range(n_groups):
        members = group_matrix[g]  # (4,)
        gkey = key[:, members]  # (n_sims, 4)
        order = np.argsort(-gkey, axis=1)  # best first
        ranked_teams = members[order]  # team ids per rank (n_sims, 4)
        group_winner[:, g] = ranked_teams[:, 0]
        runner_up[:, g] = ranked_teams[:, 1]
        third[:, g] = ranked_teams[:, 2]
        third_key[:, g] = np.take_along_axis(gkey, order, axis=1)[:, 2]
        winner_mask[rows, ranked_teams[:, 0]] = True
        top2_mask[rows, ranked_teams[:, 0]] = True
        top2_mask[rows, ranked_teams[:, 1]] = True
        # Seed score for qualified teams = their composite key.
        seed_score[rows, ranked_teams[:, 0]] = gkey[rows, order[:, 0]]
        seed_score[rows, ranked_teams[:, 1]] = gkey[rows, order[:, 1]]

    # --- Best 8 third-placed teams ---
    n_best = config.N_BEST_THIRDS
    third_order = np.argsort(-third_key, axis=1)  # rank groups' thirds per sim
    best_third_groups = third_order[:, :n_best]
    best_thirds = np.take_along_axis(third, best_third_groups, axis=1)  # (n_sims, 8)
    best_thirds_key = np.take_along_axis(third_key, best_third_groups, axis=1)
    seed_score[rows[:, None], best_thirds] = best_thirds_key

    # --- Assemble 32 qualifiers per sim ---
    qualifiers = np.concatenate([group_winner, runner_up, best_thirds], axis=1)  # (n_sims,32)

    # Seed within sim by seed_score (desc), then arrange into standard bracket.
    q_scores = np.take_along_axis(seed_score, qualifiers, axis=1)
    seed_order = np.argsort(-q_scores, axis=1)  # (n_sims, 32) seeds best->worst
    seeded = np.take_along_axis(qualifiers, seed_order, axis=1)  # team ids by seed
    bracket_pos = np.array(standard_bracket_order(32))
    bracket = seeded[:, bracket_pos]  # (n_sims, 32) team ids in bracket slots

    # --- Round-reach trackers ---
    reach = {
        "R32": np.zeros(n_teams, dtype=np.int64),
        "R16": np.zeros(n_teams, dtype=np.int64),
        "QF": np.zeros(n_teams, dtype=np.int64),
        "SF": np.zeros(n_teams, dtype=np.int64),
        "Final": np.zeros(n_teams, dtype=np.int64),
        "Champion": np.zeros(n_teams, dtype=np.int64),
    }

    def _count(team_ids: np.ndarray, bucket: str) -> None:
        flat = team_ids.reshape(-1)
        np.add.at(reach[bucket], flat, 1)

    _count(bracket, "R32")

    # --- Knockout rounds ---
    round_names = ["R16", "QF", "SF", "Final", "Champion"]
    current = bracket  # (n_sims, n_slots)
    for rname in round_names:
        n_slots = current.shape[1]
        left = current[:, 0:n_slots:2]
        right = current[:, 1:n_slots:2]
        sL = strength[left]
        sR = strength[right]
        p_left = sL / (sL + sR)  # advancement prob (draw resolves to same ratio)
        u = rng.random(left.shape)
        left_advances = u < p_left
        winners = np.where(left_advances, left, right)
        _count(winners, rname)
        current = winners

    return {
        "reach": reach,
        "winner_mask_counts": winner_mask.sum(axis=0),
        "top2_mask_counts": top2_mask.sum(axis=0),
        "n_sims": n_sims,
    }


# --------------------------------------------------------------------------- #
# Reporting & persistence
# --------------------------------------------------------------------------- #
def build_tables(setup: dict, results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the champion and advancement probability tables."""
    teams = setup["teams"]
    n = results["n_sims"]
    reach = results["reach"]

    df = pd.DataFrame(
        {
            "team": teams,
            "group": [setup["team_group"][t] for t in teams],
            "group_winner_prob": results["winner_mask_counts"] / n,
            "advance_top2_prob": results["top2_mask_counts"] / n,
            "reach_r32": reach["R32"] / n,
            "reach_r16": reach["R16"] / n,
            "reach_qf": reach["QF"] / n,
            "reach_sf": reach["SF"] / n,
            "reach_final": reach["Final"] / n,
            "champion_prob": reach["Champion"] / n,
        }
    )

    champions = (
        df[["team", "group", "champion_prob"]]
        .sort_values("champion_prob", ascending=False)
        .reset_index(drop=True)
    )
    advancement = df.sort_values(
        ["advance_top2_prob", "champion_prob"], ascending=False
    ).reset_index(drop=True)
    return champions, advancement


def print_report(champions: pd.DataFrame, advancement: pd.DataFrame) -> None:
    """Print the required summary views."""
    pd.set_option("display.width", 200)

    print("=" * 70)
    print("MONTE CARLO WORLD CUP 2026 SIMULATION")
    print("=" * 70)

    print("\n--- Top 15 champion probabilities ---")
    print(champions.head(15).assign(
        champion_prob=lambda d: (d["champion_prob"] * 100).round(2)
    ).to_string(index=False))

    print("\n--- Top 15 reach-final probabilities ---")
    fin = advancement.sort_values("reach_final", ascending=False).head(15)
    print(fin[["team", "group", "reach_final"]].assign(
        reach_final=lambda d: (d["reach_final"] * 100).round(2)
    ).to_string(index=False))

    print("\n--- Top 15 reach-semifinal probabilities ---")
    sf = advancement.sort_values("reach_sf", ascending=False).head(15)
    print(sf[["team", "group", "reach_sf"]].assign(
        reach_sf=lambda d: (d["reach_sf"] * 100).round(2)
    ).to_string(index=False))

    print("\n--- Most likely group winners (per group) ---")
    idx = advancement.groupby("group")["group_winner_prob"].idxmax()
    gw = advancement.loc[idx].sort_values("group")
    print(gw[["group", "team", "group_winner_prob"]].assign(
        group_winner_prob=lambda d: (d["group_winner_prob"] * 100).round(2)
    ).to_string(index=False))


def run(n_sims: int | None = None, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full Stage 5 simulation, save tables, and print the report."""
    config.ensure_dirs()
    n_sims = n_sims or config.N_SIMULATIONS
    seed = config.RANDOM_SEED if seed is None else seed

    preds = pd.read_csv(config.OUTPUT_FILES["fixtures_2026_predictions"])
    setup = build_setup(preds)
    print(f"teams: {setup['n_teams']} | groups: {len(setup['group_labels'])} | sims: {n_sims:,} | seed: {seed}\n")

    results = simulate(setup, n_sims, seed)
    champions, advancement = build_tables(setup, results)

    champions.to_csv(config.OUTPUT_FILES["champion_probabilities"], index=False)
    advancement.to_csv(config.OUTPUT_FILES["advancement_probabilities"], index=False)

    print_report(champions, advancement)
    print(f"\nSaved -> {config.OUTPUT_FILES['champion_probabilities']}")
    print(f"Saved -> {config.OUTPUT_FILES['advancement_probabilities']}")
    return champions, advancement


if __name__ == "__main__":
    run()
