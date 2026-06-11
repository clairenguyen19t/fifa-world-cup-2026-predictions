"""Stage 8 - LinkedIn-ready visualisations.

Generates five polished, dark-red "World Cup" themed infographics (six PNGs) at
300 DPI from the project's real outputs only:

  1. champion_probabilities_top15.png        (hero bar chart)
  2. group_stage_match_cards_A_F.png          (match prediction cards)
  3. group_stage_match_cards_G_L.png
  4. most_likely_tournament_path.png          (illustrative bracket path)
  5. pipeline_summary.png                     (project flow diagram)
  6. validation_bug_fix.png                   (Stage 7 bug-fix story)

Design is *inspired by* the Kaggle reference images (assets/reference/) but is an
original dark theme. All numbers come from real artifacts in ``outputs/`` and
``data/`` - nothing is hardcoded or invented. Matplotlib only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

from . import config, simulation

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
BG = "#0B0B0F"          # near-black background
PANEL = "#16161D"       # card / panel fill
PANEL_HI = "#1E1E27"    # lighter panel
RED = "#E1132B"         # primary World Cup red
RED_BRIGHT = "#FF3A47"  # accent / home win
RED_DARK = "#7C0E1A"    # header band
RED_DEEP = "#3A0710"    # deep gradient base
GOLD = "#F4C14E"        # champion / highlight
WHITE = "#F6F6F8"
MUTED = "#9C9CA8"
LINE = "#2C2C36"

# probability-pill colours (on-brand: red / slate / gold)
C_HOME = "#FF3A47"
C_DRAW = "#7E7E8C"
C_AWAY = "#F4C14E"

# --------------------------------------------------------------------------- #
# Overridable settings (defaults = Stage 8). Stage 10 calls configure() to
# point at the "final_" outputs, write to outputs/final_linkedin_visuals/, and
# relabel footers without touching the original Stage 8 artifacts.
# --------------------------------------------------------------------------- #
OUTPUT_DIR = config.LINKEDIN_DIR
HERO_FOOTER = "Model: Random Forest + Monte Carlo Simulation"
CARDS_FOOTER = "Source: project Random Forest model  |  full 3-way distribution shown"


def _default_sources() -> dict:
    return {
        "preds": config.OUTPUT_FILES["fixtures_2026_predictions"],
        "champ": config.OUTPUT_FILES["champion_probabilities"],
        "adv": config.OUTPUT_FILES["advancement_probabilities"],
        "model": config.OUTPUT_FILES["model_results"],
    }


SOURCES = _default_sources()


def configure(output_dir=None, sources=None, hero_footer=None,
              cards_footer=None) -> None:
    """Override output directory, data sources, and footer labels."""
    global OUTPUT_DIR, SOURCES, HERO_FOOTER, CARDS_FOOTER
    if output_dir is not None:
        OUTPUT_DIR = output_dir
    if sources is not None:
        SOURCES = sources
    if hero_footer is not None:
        HERO_FOOTER = hero_footer
    if cards_footer is not None:
        CARDS_FOOTER = cards_footer

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": WHITE,
    "axes.edgecolor": LINE,
})


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #
def _canvas(fw: float, fh: float):
    """Create a figure + full-bleed axis in data units (10 units per inch)."""
    fig = plt.figure(figsize=(fw, fh), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fw * 10)
    ax.set_ylim(0, fh * 10)
    ax.axis("off")
    ax.set_facecolor(BG)
    return fig, ax


def _vgrad(ax, x, y, w, h, c_bottom, c_top, z=0, alpha=1.0):
    """Vertical gradient rectangle (background flourish)."""
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    cmap = mcolors.LinearSegmentedColormap.from_list("g", [c_bottom, c_top])
    ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", cmap=cmap,
              origin="lower", zorder=z, alpha=alpha)


def _rrect(ax, x, y, w, h, fc, ec="none", lw=0.0, r=2.2, alpha=1.0, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
        mutation_aspect=1, zorder=z,
    )
    ax.add_patch(p)
    return p


def _seg(ax, x, y, w, h, fc, ec="none", lw=0.0, z=3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=z))


def _bg_flourish(ax, W, H):
    """Common deep-red glow at the top of every slide."""
    _vgrad(ax, 0, 0, W, H, BG, BG, z=0)
    _vgrad(ax, 0, H * 0.55, W, H * 0.45, BG, RED_DEEP, z=0, alpha=0.55)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data() -> dict:
    preds = pd.read_csv(SOURCES["preds"])
    champ = pd.read_csv(SOURCES["champ"])
    adv = pd.read_csv(SOURCES["adv"])
    model = pd.read_csv(SOURCES["model"])
    # team -> group, reconstructed consistently with the simulation
    setup = simulation.build_setup(preds)
    return {
        "preds": preds,
        "champ": champ,
        "adv": adv,
        "model": model,
        "setup": setup,
        "team_group": setup["team_group"],
        "strength": dict(zip(setup["teams"], setup["strength"])),
    }


def _save(fig, name: str) -> str:
    config.ensure_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, facecolor=BG)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# VISUAL 1 - Hero champion probabilities
# --------------------------------------------------------------------------- #
def visual_champions(data: dict) -> str:
    champ = data["champ"].sort_values("champion_prob", ascending=False).head(15)
    fw, fh = 9.0, 11.25
    fig, ax = _canvas(fw, fh)
    W, H = fw * 10, fh * 10
    _bg_flourish(ax, W, H)

    # Header
    ax.text(8, H - 13, "WORLD CUP 2026", color=GOLD, fontsize=14,
            fontweight="bold", ha="left", va="center")
    ax.text(8, H - 20.5, "Champion Probabilities", color=WHITE, fontsize=27,
            fontweight="bold", ha="left", va="center")
    ax.text(8, H - 27.5,
            "10,000 Monte Carlo simulations using historical match data,\n"
            "Elo ratings, and rolling-form features",
            color=MUTED, fontsize=11.5, ha="left", va="top", linespacing=1.4)

    # Bars
    teams = champ["team"].tolist()
    vals = (champ["champion_prob"] * 100).tolist()
    n = len(teams)
    top = H - 36
    bottom = 12
    row_h = (top - bottom) / n
    bar_h = row_h * 0.6
    x0 = 30
    bar_max = 48
    vmax = max(vals)

    for i, (t, v) in enumerate(zip(teams, vals)):
        yc = top - (i + 0.5) * row_h
        is_top = i == 0
        # rank
        ax.text(6, yc, f"{i+1:>2}", color=GOLD if is_top else MUTED,
                fontsize=12 if is_top else 10.5,
                fontweight="bold", ha="left", va="center")
        # team name
        ax.text(11.5, yc, t, color=WHITE, fontsize=11.5,
                fontweight="bold" if is_top else "normal", ha="left", va="center")
        # track
        _rrect(ax, x0, yc - bar_h / 2, bar_max, bar_h, fc=PANEL_HI, r=bar_h / 2, z=2)
        # bar
        bw = max(bar_max * v / vmax, bar_h)
        ax.add_patch(FancyBboxPatch(
            (x0, yc - bar_h / 2), bw, bar_h,
            boxstyle=f"round,pad=0,rounding_size={bar_h/2}",
            facecolor=GOLD if is_top else RED, edgecolor="none",
            mutation_aspect=1, zorder=3))
        # value label
        ax.text(x0 + bw + 1.5, yc, f"{v:.1f}%",
                color=GOLD if is_top else WHITE, fontsize=11,
                fontweight="bold", ha="left", va="center")

    # divider + footer
    ax.plot([8, W - 8], [8.5, 8.5], color=LINE, lw=1, zorder=2)
    ax.text(8, 5, HERO_FOOTER,
            color=MUTED, fontsize=10.5, ha="left", va="center")
    ax.text(W - 8, 5, "Probabilities sum across 48 teams",
            color=MUTED, fontsize=9, ha="right", va="center", style="italic")
    return _save(fig, "champion_probabilities_top15.png")


# --------------------------------------------------------------------------- #
# VISUAL 2 - Match prediction cards
# --------------------------------------------------------------------------- #
_LABEL_FS = 7.3  # value-line font size (kept identical to the prior theme)


def _place_seg_labels(ax, bx, bw, ly, segs):
    """Place H/D/A labels centred on each segment's midpoint along ``[bx, bx+bw]``.

    Each label is drawn ``ha="center", va="center"`` at its segment midpoint so it
    aligns with the colour above it. Where a segment is too narrow for the label,
    a left-to-right (then right-to-left) sweep nudges the label just outside the
    segment with a consistent gap, keeping labels from overlapping and clamped to
    the bar bounds. Pure positioning - colours/sizes/theme are unchanged.

    ``segs`` is a list of ``(text, color, frac)`` in draw order (home, draw, away).
    """
    # Approximate glyph width in data units (axis is 10 units / inch).
    char_w = _LABEL_FS * 0.6 / 72 * 10
    gap = 0.7

    items = []
    start = bx
    for text, color, frac in segs:
        sw = bw * frac
        mid = start + sw / 2.0
        half = max(len(text) * char_w / 2.0, 0.1)
        items.append([text, color, mid, half])
        start += sw

    # Left-to-right: ensure each label clears the previous one.
    for i in range(1, len(items)):
        min_x = items[i - 1][2] + items[i - 1][3] + gap + items[i][3]
        if items[i][2] < min_x:
            items[i][2] = min_x

    # Right clamp + right-to-left back-off so the rightmost stays in bounds.
    right_edge = bx + bw
    if items[-1][2] + items[-1][3] > right_edge:
        items[-1][2] = right_edge - items[-1][3]
        for i in range(len(items) - 2, -1, -1):
            max_x = items[i + 1][2] - items[i + 1][3] - gap - items[i][3]
            if items[i][2] > max_x:
                items[i][2] = max_x

    # Left clamp for the first label.
    if items[0][2] - items[0][3] < bx:
        items[0][2] = bx + items[0][3]

    for text, color, mid, _half in items:
        ax.text(mid, ly, text, color=color, fontsize=_LABEL_FS,
                fontweight="bold", ha="center", va="center")


def _team_label(ax, x, y, name, favored, side):
    """Draw a team name (always white/light) with a small gold star if it is the
    model's favoured side. ``side`` = 'home' (left-aligned) or 'away'
    (right-aligned). Names are never coloured gold so the gold/away-win bar stays
    unambiguous; the star is the only gold winner cue.
    """
    fs = 8.2
    char_w = fs * 0.6 / 72 * 10
    star = "\u2605"
    if side == "home":
        ax.text(x, y, name, color=WHITE, fontsize=fs, ha="left", va="center")
        if favored:
            ax.text(x + len(name) * char_w + 0.7, y, star, color=GOLD,
                    fontsize=fs - 1.4, ha="left", va="center")
    else:
        ax.text(x, y, name, color=WHITE, fontsize=fs, ha="right", va="center")
        if favored:
            ax.text(x - len(name) * char_w - 0.7, y, star, color=GOLD,
                    fontsize=fs - 1.4, ha="right", va="center")


def _match_cell(ax, x, y, w, h, row):
    """Draw one match: two teams (winner flagged with a gold star), a 3-way
    probability pill, and a centred value line. Generous horizontal padding
    keeps names off the card edges and clear of the bar."""
    ph, pd_, pa = row["P_home_win"], row["P_draw"], row["P_away_win"]
    outcome = row["predicted_outcome"]
    home_hi = outcome == "home_win"
    away_hi = outcome == "away_win"
    draw_hi = outcome == "draw"

    pad = 1.8

    # team names above the bar (both white; favoured side flagged with a star)
    name_y = y + h - 1.3
    _team_label(ax, x + pad, name_y, _disp(row["home_team"], 16), home_hi, "home")
    _team_label(ax, x + w - pad, name_y, _disp(row["away_team"], 16), away_hi, "away")

    # probability pill (segmented), inset by the same horizontal padding
    bx, bw = x + pad, w - 2 * pad
    by, bh = y + h - 3.5, 1.4
    _rrect(ax, bx, by, bw, bh, fc=PANEL, r=bh / 2, z=3)
    segs = [(ph, C_HOME, home_hi), (pd_, C_DRAW, draw_hi), (pa, C_AWAY, away_hi)]
    cx = bx
    for frac, col, hi in segs:
        sw = bw * frac
        _seg(ax, cx, by, sw, bh, fc=col, z=4)
        cx += sw
    # white outline on most-likely segment
    cx = bx
    for frac, col, hi in segs:
        sw = bw * frac
        if hi:
            ax.add_patch(Rectangle((cx, by), sw, bh, facecolor="none",
                                   edgecolor=WHITE, linewidth=1.1, zorder=5))
        cx += sw

    # value line - each H/D/A label is centred (va="center") on its own
    # segment's midpoint instead of fixed card offsets, so the label sits under
    # the matching colour. Narrow segments push their label out with consistent
    # padding so labels never overlap.
    vy = y + 0.7
    _place_seg_labels(ax, bx, bw, vy, [
        (f"H {ph*100:.0f}%", C_HOME, ph),
        (f"D {pd_*100:.0f}%", MUTED, pd_),
        (f"A {pa*100:.0f}%", C_AWAY, pa),
    ])


def _short(name: str, n: int) -> str:
    return name if len(name) <= n else name[: n - 1] + "\u2026"


# Display-only aliases for long team names (labels only - the underlying data,
# probabilities and simulation keep their canonical names untouched).
DISPLAY_ALIASES = {
    "Bosnia and Herzegovina": "Bosnia & Herz.",
    "Czech Republic": "Czech Rep.",
    "Ivory Coast": "C\u00f4te d\u2019Ivoire",
    "United States": "USA",
    "Democratic Republic of Congo": "DR Congo",
    "DR Congo": "DR Congo",
    "South Korea": "South Korea",
    "Saudi Arabia": "Saudi Arabia",
    "Cape Verde": "Cape Verde",
}


def _disp(name: str, n: int) -> str:
    """Map a team name to its clean display alias, then truncate as a fallback."""
    return _short(DISPLAY_ALIASES.get(name, name), n)


def _cards_image(data: dict, groups: list[str], filename: str, subtitle: str) -> str:
    preds = data["preds"].copy()
    preds["date"] = pd.to_datetime(preds["date"])
    tg = data["team_group"]
    preds["group"] = preds["home_team"].map(tg)

    fw, fh = 15.0, 11.6
    fig, ax = _canvas(fw, fh)
    W, H = fw * 10, fh * 10
    _bg_flourish(ax, W, H)

    # header band
    _rrect(ax, 6, H - 20, W - 12, 14, fc=RED_DARK, r=2.5, z=2)
    ax.text(W / 2, H - 10.5, "Group-Stage Match Predictions", color=WHITE,
            fontsize=21, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, H - 16.5, subtitle, color=GOLD, fontsize=11,
            ha="center", va="center")
    # legend
    lx = 9
    for lab, col in [("Home win", C_HOME), ("Draw", C_DRAW), ("Away win", C_AWAY)]:
        _seg(ax, lx, H - 25.2, 2.2, 1.6, fc=col, z=4)
        ax.text(lx + 3, H - 24.4, lab, color=MUTED, fontsize=9, ha="left", va="center")
        lx += len(lab) * 1.5 + 12
    # star = favoured side (winner cue; keeps gold off the team names)
    ax.text(lx, H - 24.4, "\u2605", color=GOLD, fontsize=9, ha="left", va="center")
    ax.text(lx + 2.6, H - 24.4, "Favored (model pick)", color=MUTED, fontsize=9,
            ha="left", va="center")

    # 3 columns x 2 rows of group panels
    n_cols, n_rows = 3, 2
    margin_x, gap_x = 6, 3
    top_area = H - 28
    bottom_area = 5
    gap_y = 3
    panel_w = (W - 2 * margin_x - (n_cols - 1) * gap_x) / n_cols
    panel_h = (top_area - bottom_area - (n_rows - 1) * gap_y) / n_rows

    for idx, g in enumerate(groups):
        c = idx % n_cols
        r = idx // n_cols
        px = margin_x + c * (panel_w + gap_x)
        py = top_area - (r + 1) * panel_h - r * gap_y
        _rrect(ax, px, py, panel_w, panel_h, fc=PANEL, ec=LINE, lw=1, r=2.5, z=2)
        # group title
        _rrect(ax, px, py + panel_h - 5, panel_w, 5, fc=RED, r=2.5, z=3)
        ax.text(px + panel_w / 2, py + panel_h - 2.5, f"GROUP {g}", color=WHITE,
                fontsize=11, fontweight="bold", ha="center", va="center")

        gmatches = preds[preds["group"] == g].sort_values("date").reset_index(drop=True)
        cell_area = panel_h - 5.5
        cell_h = cell_area / len(gmatches)
        for j, (_, row) in enumerate(gmatches.iterrows()):
            cy = py + panel_h - 5.5 - (j + 1) * cell_h
            _match_cell(ax, px + 0.8, cy + 0.3, panel_w - 1.6, cell_h - 0.6, row)

    ax.text(W - 6, 2.4, CARDS_FOOTER,
            color=MUTED, fontsize=8.5, ha="right", va="center", style="italic")
    return _save(fig, filename)


def visual_cards_a_f(data: dict) -> str:
    return _cards_image(data, list("ABCDEF"), "group_stage_match_cards_A_F.png",
                        "Groups A - F  |  predicted outcome probabilities")


def visual_cards_g_l(data: dict) -> str:
    return _cards_image(data, list("GHIJKL"), "group_stage_match_cards_G_L.png",
                        "Groups G - L  |  predicted outcome probabilities")


# --------------------------------------------------------------------------- #
# VISUAL 3 - Most likely tournament path (illustrative bracket)
# --------------------------------------------------------------------------- #
def _seed_bracket(data: dict):
    """Return (bracket, strength) where bracket is 32 teams in slot order.

    Qualifier selection and seeding follow group-placement logic first
    (winners > runners-up > thirds), with champion/Elo strength only as the
    secondary ordering within a tier.
    """
    adv = data["adv"]
    strength = data["strength"]

    groups = sorted(adv["group"].unique())
    winners, runners, third_candidates = [], [], []
    for g in groups:
        sub = adv[adv["group"] == g].sort_values("advance_top2_prob", ascending=False)
        top2 = sub.head(2)["team"].tolist()
        gw = adv[adv["team"].isin(top2)].sort_values("group_winner_prob", ascending=False)
        w = gw.iloc[0]["team"]
        ru = [t for t in top2 if t != w][0]
        winners.append(w)
        runners.append(ru)
        third_candidates.append(sub.iloc[2]["team"])

    thirds = (
        adv[adv["team"].isin(third_candidates)]
        .sort_values("reach_r32", ascending=False)
        .head(config.N_BEST_THIRDS)["team"].tolist()
    )

    winners.sort(key=lambda t: -strength[t])
    runners.sort(key=lambda t: -strength[t])
    thirds.sort(key=lambda t: -strength[t])
    seed_pool = winners + runners + thirds  # seed 1..32

    bracket = [seed_pool[i] for i in simulation.standard_bracket_order(32)]
    return bracket, strength


def _play_rounds(bracket: list, strength: dict):
    """Deterministically resolve the bracket; return list of rounds of matches.

    Each match is ``(team_a, team_b, winner, win_prob)``; winner is the higher
    strength side and ``win_prob = sW / (sA + sB)`` (Elo-based).
    """
    cur = bracket
    all_rounds = []
    while len(cur) > 1:
        matches, nxt = [], []
        for i in range(0, len(cur), 2):
            a, b = cur[i], cur[i + 1]
            sa, sb = strength[a], strength[b]
            w = a if sa >= sb else b
            p = max(sa, sb) / (sa + sb)
            matches.append((a, b, w, p))
            nxt.append(w)
        all_rounds.append(matches)
        cur = nxt
    return all_rounds, cur[0]


def build_bracket_path(data: dict) -> dict:
    """Construct a defensible illustrative knockout path (champion's road)."""
    bracket, strength = _seed_bracket(data)
    all_rounds, champion = _play_rounds(bracket, strength)
    round_names = ["Round of 32", "Round of 16", "Quarterfinals",
                   "Semifinals", "Final"]
    path = []
    for matches in all_rounds:
        for (a, b, w, p) in matches:
            if champion in (a, b):
                path.append((a, b, w, p))
                break
    return {"round_names": round_names, "path": path, "champion": champion,
            "tg": data["team_group"]}


def visual_bracket(data: dict) -> str:
    bp = build_bracket_path(data)
    rounds, path, champ, tg = bp["round_names"], bp["path"], bp["champion"], bp["tg"]

    fw, fh = 16.0, 9.6
    fig, ax = _canvas(fw, fh)
    W, H = fw * 10, fh * 10
    _bg_flourish(ax, W, H)

    # header
    _rrect(ax, 6, H - 18, W - 12, 12, fc=RED_DARK, r=2.5, z=2)
    ax.text(W / 2, H - 9.3, f"Road to the Final  -  {champ}", color=WHITE,
            fontsize=21, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, H - 14.3,
            "Illustrative simulation scenario based on model outputs",
            color=GOLD, fontsize=11.5, ha="center", va="center", style="italic")

    n = len(path)
    margin = 6
    gap = 3
    card_w = (W - 2 * margin - (n - 1) * gap) / n
    card_h = 30
    cy = (H - 18) / 2 - card_h / 2 + 4

    for i, ((a, b, w, p), rname) in enumerate(zip(path, rounds)):
        cx = margin + i * (card_w + gap)
        # round label
        _rrect(ax, cx, cy + card_h + 3, card_w, 5, fc=RED, r=2, z=3)
        ax.text(cx + card_w / 2, cy + card_h + 5.5, rname.upper(), color=WHITE,
                fontsize=9.5, fontweight="bold", ha="center", va="center")
        # card
        is_final = i == n - 1
        _rrect(ax, cx, cy, card_w, card_h, fc=PANEL,
               ec=GOLD if is_final else LINE, lw=1.6 if is_final else 1, r=2.5, z=2)

        # two team rows
        for k, team in enumerate((a, b)):
            ty = cy + card_h - 6 - k * 6.5
            win = team == w
            _rrect(ax, cx + 1.5, ty - 2.4, card_w - 3, 5, fc=PANEL_HI,
                   ec=GOLD if win else "none", lw=1.2 if win else 0, r=1.6, z=3)
            ax.text(cx + 3, ty, _disp(team, 16),
                    color=GOLD if win else WHITE, fontsize=9.5,
                    fontweight="bold" if win else "normal", ha="left", va="center")
            if win:
                ax.text(cx + card_w - 3, ty, "\u2713", color=GOLD, fontsize=10,
                        fontweight="bold", ha="right", va="center")

        # "vs"
        ax.text(cx + card_w / 2, cy + card_h - 9.4, "vs", color=MUTED,
                fontsize=7.5, ha="center", va="center")
        # winner + prob
        ax.plot([cx + 2, cx + card_w - 2], [cy + 8.5, cy + 8.5], color=LINE, lw=0.8, zorder=3)
        ax.text(cx + card_w / 2, cy + 5.6, "WINNER", color=MUTED, fontsize=7,
                fontweight="bold", ha="center", va="center")
        ax.text(cx + card_w / 2, cy + 2.6, f"{w}  -  {p*100:.0f}%", color=GOLD,
                fontsize=9.5, fontweight="bold", ha="center", va="center")

        # chevron between cards
        if i < n - 1:
            ax.text(cx + card_w + gap / 2, cy + card_h / 2, "\u203A", color=RED_BRIGHT,
                    fontsize=20, fontweight="bold", ha="center", va="center", zorder=6)

    ax.text(W / 2, 4,
            "Qualifiers: top 2 per group (advance_top2_prob) + 8 best third-placed "
            "(reach_r32).  Match winner by Elo-based strength.",
            color=MUTED, fontsize=8.5, ha="center", va="center", style="italic")
    return _save(fig, "most_likely_tournament_path.png")


# --------------------------------------------------------------------------- #
# VISUAL 3b - Full decision-tree bracket (32 teams)
# --------------------------------------------------------------------------- #
def _bracket_card(ax, x, y, w, h, a, b, winner, prob, highlight=False, big=False):
    """Draw one bracket match card with both teams; winner highlighted."""
    _rrect(ax, x, y, w, h, fc=PANEL,
           ec=GOLD if highlight else LINE, lw=1.6 if highlight else 0.9,
           r=1.6, z=4)
    fs = 9.0 if big else 7.6
    pad = w * 0.06
    for k, team in enumerate((a, b)):
        ty = y + h * (0.70 if k == 0 else 0.30)
        win = team == winner
        if win:
            _rrect(ax, x + pad * 0.5, ty - h * 0.18, w - pad, h * 0.36,
                   fc="#2A1B12", ec=GOLD, lw=0.8, r=1.0, z=5)
        ax.text(x + pad, ty, _disp(team, 15),
                color=GOLD if win else MUTED, fontsize=fs,
                fontweight="bold" if win else "normal",
                ha="left", va="center", zorder=6)
        if win:
            ax.text(x + w - pad, ty, f"{prob*100:.0f}%", color=GOLD,
                    fontsize=fs - 0.8, fontweight="bold", ha="right",
                    va="center", zorder=6)


def _connect(ax, x0, y0, x1, y1, midx, col=LINE, lw=1.0):
    """Bracket-style elbow connector: feeder -> vertical riser -> next card."""
    ax.plot([x0, midx], [y0, y0], color=col, lw=lw, zorder=2)
    ax.plot([midx, midx], [y0, y1], color=col, lw=lw, zorder=2)
    ax.plot([midx, x1], [y1, y1], color=col, lw=lw, zorder=2)


def visual_full_bracket(data: dict) -> str:
    bracket, strength = _seed_bracket(data)
    all_rounds, champion = _play_rounds(bracket, strength)
    champ_prob = float(
        data["champ"].loc[data["champ"]["team"] == champion, "champion_prob"].iloc[0]
    )

    fw, fh = 20.0, 11.5
    fig, ax = _canvas(fw, fh)
    W, H = fw * 10, fh * 10
    _bg_flourish(ax, W, H)

    # header
    _rrect(ax, 5, H - 16, W - 10, 11, fc=RED_DARK, r=2.5, z=2)
    ax.text(W / 2, H - 8.0, "FIFA World Cup 2026  -  Full Decision-Tree Bracket",
            color=WHITE, fontsize=20, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, H - 13.2,
            "Illustrative full decision-tree scenario based on model outputs",
            color=GOLD, fontsize=11, ha="center", va="center", style="italic")

    # geometry
    top, bottom = H - 22, 9
    card_w = 21.0
    margin = 4.0
    step = (W - 2 * margin - card_w) / 8.0
    xs = [margin + c * step for c in range(9)]   # 9 column left edges

    def col_ys(n):
        s = (top - bottom) / n
        return [top - (i + 0.5) * s for i in range(n)]

    def merge_ys(ys):
        return [(ys[2 * i] + ys[2 * i + 1]) / 2 for i in range(len(ys) // 2)]

    # left/right y-ladders
    L = [col_ys(8)]
    for _ in range(3):
        L.append(merge_ys(L[-1]))           # R16(4), QF(2), SF(1)
    R = [list(L[0])]
    for _ in range(3):
        R.append(merge_ys(R[-1]))
    final_y = (L[3][0] + R[3][0]) / 2

    ch = 7.4   # card height
    # round-to-column mapping (left then right)
    left_cols = [0, 1, 2, 3]
    right_cols = [8, 7, 6, 5]
    round_names = ["ROUND OF 32", "ROUND OF 16", "QUARTERFINALS", "SEMIFINALS"]

    # split each round's matches into left and right halves
    def halves(matches):
        h = len(matches) // 2
        return matches[:h], matches[h:]

    # round labels across the top
    label_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    label_txt = ["R32", "R16", "QF", "SF", "FINAL", "SF", "QF", "R16", "R32"]
    for c, t in zip(label_cols, label_txt):
        _rrect(ax, xs[c], top + 2.5, card_w, 4.5, fc=RED, r=1.6, z=3)
        ax.text(xs[c] + card_w / 2, top + 4.75, t, color=WHITE, fontsize=9,
                fontweight="bold", ha="center", va="center", zorder=4)

    # draw the four knockout rounds on both sides + connectors
    for r in range(4):
        lmatches, rmatches = halves(all_rounds[r])
        lys, rys = L[r], R[r]
        lc, rc = left_cols[r], right_cols[r]
        for i, (m, y) in enumerate(zip(lmatches, lys)):
            _bracket_card(ax, xs[lc], y - ch / 2, card_w, ch, *m)
        for i, (m, y) in enumerate(zip(rmatches, rys)):
            _bracket_card(ax, xs[rc], y - ch / 2, card_w, ch, *m)
        # connectors to next round (toward centre)
        if r < 3:
            nlc, nrc = left_cols[r + 1], right_cols[r + 1]
            nlys, nrys = L[r + 1], R[r + 1]
            for i in range(len(nlys)):
                midx = (xs[lc] + card_w + xs[nlc]) / 2
                _connect(ax, xs[lc] + card_w, lys[2 * i], xs[nlc], nlys[i], midx)
                _connect(ax, xs[lc] + card_w, lys[2 * i + 1], xs[nlc], nlys[i], midx)
            for i in range(len(nrys)):
                midx = (xs[rc] + xs[nrc] + card_w) / 2
                _connect(ax, xs[rc], rys[2 * i], xs[nrc] + card_w, nrys[i], midx)
                _connect(ax, xs[rc], rys[2 * i + 1], xs[nrc] + card_w, nrys[i], midx)

    # SF -> Final connectors
    fcx = xs[4]
    midxl = (xs[3] + card_w + fcx) / 2
    _connect(ax, xs[3] + card_w, L[3][0], fcx, final_y, midxl)
    midxr = (xs[5] + fcx + card_w) / 2
    _connect(ax, xs[5], R[3][0], fcx + card_w, final_y, midxr)

    # Final card
    final_match = all_rounds[4][0]
    _bracket_card(ax, xs[4], final_y - ch / 2, card_w, ch, *final_match,
                  highlight=True)

    # Champion banner above the final
    cby = final_y + 16
    _connect(ax, fcx + card_w / 2, final_y + ch / 2, fcx + card_w / 2, cby - 5, fcx + card_w / 2)
    ax.plot([fcx + card_w / 2, fcx + card_w / 2], [final_y + ch / 2, cby - 5.5],
            color=GOLD, lw=1.2, zorder=2)
    _rrect(ax, xs[4] - 1, cby - 5.5, card_w + 2, 11, fc=RED, ec=GOLD, lw=1.8, r=2.2, z=5)
    ax.text(fcx + card_w / 2, cby + 2.6, "\u2605 CHAMPION", color=GOLD, fontsize=10,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(fcx + card_w / 2, cby - 1.0, _disp(champion, 18), color=WHITE,
            fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(fcx + card_w / 2, cby - 4.0, f"title odds {champ_prob*100:.1f}%",
            color=GOLD, fontsize=8, ha="center", va="center", zorder=6)

    # footer
    ax.text(W / 2, 3.5,
            "Qualifiers: top 2 per group (advance_top2_prob) + 8 best third-placed "
            "(reach_r32).  Seeding by group placement; match winner by Elo-based "
            "strength.  %% = modelled win probability.",
            color=MUTED, fontsize=8.5, ha="center", va="center", style="italic")
    return _save(fig, "full_decision_tree_bracket.png")


# --------------------------------------------------------------------------- #
# VISUAL 4 - Pipeline summary
# --------------------------------------------------------------------------- #
def _pipeline_stats(data: dict) -> dict:
    stats = {}
    try:
        stats["matches"] = len(pd.read_parquet(config.INTERIM_FILES["results_played"]))
    except Exception:
        stats["matches"] = None
    try:
        stats["elo"] = len(pd.read_parquet(config.INTERIM_FILES["eloratings_clean"]))
    except Exception:
        stats["elo"] = None
    stats["features"] = len(config.MODEL_FEATURES)
    best = data["model"].sort_values("log_loss").iloc[0]
    stats["model_name"] = best["model"]
    stats["acc"] = best["accuracy"]
    stats["fixtures"] = len(data["preds"])
    stats["sims"] = config.N_SIMULATIONS
    top = data["champ"].sort_values("champion_prob", ascending=False).iloc[0]
    stats["top_team"] = top["team"]
    stats["top_prob"] = top["champion_prob"] * 100
    return stats


def visual_pipeline(data: dict) -> str:
    s = _pipeline_stats(data)
    fw, fh = 9.0, 11.25
    fig, ax = _canvas(fw, fh)
    W, H = fw * 10, fh * 10
    _bg_flourish(ax, W, H)

    ax.text(W / 2, H - 11, "From Raw Data to Champion Odds", color=WHITE,
            fontsize=22, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, H - 17, "World Cup 2026 prediction pipeline", color=GOLD,
            fontsize=12, ha="center", va="center")

    # two inputs -> merge -> chain
    cw, ch = 58, 8.2
    cx = (W - cw) / 2

    def node(y, title, sub, fc=PANEL, ec=LINE, tcol=WHITE, w=cw, x=cx, big=False):
        _rrect(ax, x, y, w, ch, fc=fc, ec=ec, lw=1.3, r=2.2, z=3)
        ax.text(x + w / 2, y + ch * 0.62, title, color=tcol,
                fontsize=12 if big else 11, fontweight="bold",
                ha="center", va="center")
        if sub:
            ax.text(x + w / 2, y + ch * 0.24, sub, color=MUTED, fontsize=8.8,
                    ha="center", va="center")

    def arrow(y0, y1, x=W / 2):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=RED_BRIGHT, lw=2.2),
                    zorder=2)

    # input row (two side-by-side)
    iw = 27
    top_y = H - 30
    ix1 = cx
    ix2 = cx + cw - iw
    _rrect(ax, ix1, top_y, iw, ch, fc=PANEL, ec=RED, lw=1.3, r=2.2, z=3)
    ax.text(ix1 + iw / 2, top_y + ch * 0.62, "Match results", color=WHITE,
            fontsize=10.5, fontweight="bold", ha="center", va="center")
    ax.text(ix1 + iw / 2, top_y + ch * 0.24,
            f"{s['matches']:,} games" if s["matches"] else "historical",
            color=MUTED, fontsize=8.8, ha="center", va="center")
    _rrect(ax, ix2, top_y, iw, ch, fc=PANEL, ec=RED, lw=1.3, r=2.2, z=3)
    ax.text(ix2 + iw / 2, top_y + ch * 0.62, "Elo ratings", color=WHITE,
            fontsize=10.5, fontweight="bold", ha="center", va="center")
    ax.text(ix2 + iw / 2, top_y + ch * 0.24,
            f"{s['elo']:,} rows" if s["elo"] else "team strength",
            color=MUTED, fontsize=8.8, ha="center", va="center")

    chain = [
        ("Cleaning & team normalization", "fix Unicode names, split fixtures"),
        ("Feature engineering", f"{s['features']} leakage-safe features"),
        (f"{s['model_name']} match model", f"test accuracy {s['acc']*100:.1f}%"),
        ("2026 match probabilities", f"{s['fixtures']} group-stage fixtures"),
        ("Monte Carlo simulation", f"{s['sims']:,} tournament runs"),
    ]
    gap = (top_y - 16) / (len(chain) + 1)
    # arrows from inputs merge point
    merge_y = top_y - gap + ch
    arrow(top_y, merge_y, x=ix1 + iw / 2)
    arrow(top_y, merge_y, x=ix2 + iw / 2)

    y = top_y
    for i, (title, sub) in enumerate(chain):
        ny = y - gap
        node(ny, title, sub)
        arrow(y if i else merge_y, ny + ch) if i else arrow(merge_y, ny + ch)
        y = ny

    # final highlight node
    fy = y - gap
    _rrect(ax, cx, fy, cw, ch + 1.5, fc=RED, ec=GOLD, lw=1.6, r=2.4, z=3)
    ax.text(W / 2, fy + (ch + 1.5) * 0.62, "Champion probabilities", color=WHITE,
            fontsize=12, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, fy + (ch + 1.5) * 0.24,
            f"{s['top_team']} leads at {s['top_prob']:.1f}%", color=GOLD,
            fontsize=9.5, fontweight="bold", ha="center", va="center")
    arrow(y, fy + ch + 1.5)

    ax.text(W / 2, 4, "Reproducible: clean -> features -> model -> predict -> simulate",
            color=MUTED, fontsize=9, ha="center", va="center", style="italic")
    return _save(fig, "pipeline_summary.png")


# --------------------------------------------------------------------------- #
# VISUAL 5 - Validation / bug-fix story
# --------------------------------------------------------------------------- #
def visual_validation(data: dict) -> str:
    fw, fh = 9.0, 11.25
    fig, ax = _canvas(fw, fh)
    W, H = fw * 10, fh * 10
    _bg_flourish(ax, W, H)

    ax.text(W / 2, H - 11, "Why Data Validation Matters", color=WHITE,
            fontsize=23, fontweight="bold", ha="center", va="center")
    ax.text(W / 2, H - 17.5,
            "A single hidden Unicode character distorted the whole simulation",
            color=GOLD, fontsize=11, ha="center", va="center")

    # The bug panel
    by, bh = 70, 22
    _rrect(ax, 7, by, W - 14, bh, fc=PANEL, ec=RED, lw=1.3, r=2.5, z=3)
    ax.text(11, by + bh - 3.5, "THE BUG", color=RED_BRIGHT, fontsize=12,
            fontweight="bold", ha="left", va="center")
    ax.text(11, by + bh - 7.5,
            "Elo team names stored a non-breaking space (U+00A0) instead of a\n"
            "normal space, so the Elo join silently failed for every multi-word team.",
            color=WHITE, fontsize=10.2, ha="left", va="top", linespacing=1.5)
    # code-style chips (kept within the 7..83 panel bounds)
    chip_y = by + 2.4
    ax.text(11, chip_y + 2, "Expected", color=MUTED, fontsize=8.5, ha="left",
            va="center", zorder=6)
    _rrect(ax, 25, chip_y, 22, 4, fc="#101016", ec="#2FAF6B", lw=1, r=1.2, z=4)
    ax.text(26.5, chip_y + 2, "'United States'", color="#7CD992", fontsize=8.5,
            ha="left", va="center", family="DejaVu Sans Mono", zorder=6)
    ax.text(50, chip_y + 2, "Stored", color=MUTED, fontsize=8.5, ha="left",
            va="center", zorder=6)
    _rrect(ax, 61, chip_y, 21, 4, fc="#101016", ec=RED, lw=1, r=1.2, z=4)
    ax.text(62.3, chip_y + 2, "'United\\xa0States'", color=RED_BRIGHT, fontsize=8.5,
            ha="left", va="center", family="DejaVu Sans Mono", zorder=6)

    # before / after comparison cards
    cy, card_h = 24, 40
    cw = (W - 14 - 4) / 2
    metrics = [
        ("BEFORE FIX", RED, WHITE, [
            ("30", "Missing Elo cells in 2026 fixtures"),
            ("1543", "USA Elo - imputed median"),
            ("0.19%", "USA title odds - unrealistic for a host"),
        ]),
        ("AFTER FIX", "#2FAF6B", "#0B0B0F", [
            ("0", "Missing Elo cells - fully covered"),
            ("1747", "USA Elo - real rating"),
            ("0.34%", "USA title odds - defensible"),
        ]),
    ]
    for i, (label, col, hdr_txt, rows) in enumerate(metrics):
        x = 7 + i * (cw + 4)
        _rrect(ax, x, cy, cw, card_h, fc=PANEL, ec=col, lw=1.5, r=2.5, z=3)
        _rrect(ax, x, cy + card_h - 5.5, cw, 5.5, fc=col, r=2.5, z=4)
        ax.text(x + cw / 2, cy + card_h - 2.7, label, color=hdr_txt,
                fontsize=12, fontweight="bold", ha="center", va="center")
        for j, (val, sub) in enumerate(rows):
            ry = cy + card_h - 13 - j * 9.5
            ax.text(x + cw / 2, ry + 2.3, val, color=col if i else WHITE,
                    fontsize=17, fontweight="bold", ha="center", va="center")
            ax.text(x + cw / 2, ry - 2.2, sub, color=MUTED,
                    fontsize=8.2, ha="center", va="center")

    # takeaway
    ax.text(W / 2, 13,
            "Fixing one invisible character corrected ~10 teams\nand reshuffled the top contenders.",
            color=WHITE, fontsize=10.5, ha="center", va="center", linespacing=1.5)
    ax.text(W / 2, 5, "Stage 7 validation  |  numbers from project before/after re-run",
            color=MUTED, fontsize=8.5, ha="center", va="center", style="italic")
    return _save(fig, "validation_bug_fix.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_all() -> list[str]:
    data = load_data()
    paths = [
        visual_champions(data),
        visual_cards_a_f(data),
        visual_cards_g_l(data),
        visual_bracket(data),
        visual_full_bracket(data),
        visual_pipeline(data),
        visual_validation(data),
    ]
    print("Generated LinkedIn visuals:")
    for p in paths:
        print("  -", p)
    return paths


if __name__ == "__main__":
    generate_all()
