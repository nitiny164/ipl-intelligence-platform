"""
module_5_explorer.py — Data Explorer analytics layer.

All functions are framework-independent (no Streamlit imports).
They accept pre-loaded DataFrames and filter parameters, and return
DataFrames ready for display / download.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FILTER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_match_filters(
    matches: pd.DataFrame,
    seasons: list[int] | None = None,
    team_ids: list[int] | None = None,
    venues: list[str] | None = None,
) -> pd.DataFrame:
    """Return filtered matches DataFrame."""
    m = matches.copy()
    if seasons:
        m = m[m["season"].isin(seasons)]
    if team_ids:
        m = m[m["team1"].isin(team_ids) | m["team2"].isin(team_ids)]
    if venues:
        m = m[m["venue"].isin(venues)]
    return m


def apply_delivery_filters(
    deliveries: pd.DataFrame,
    match_ids: pd.Index | list | None = None,
    phases: list[str] | None = None,
    innings: list[int] | None = None,
) -> pd.DataFrame:
    """Return filtered deliveries DataFrame."""
    d = deliveries.copy()
    if match_ids is not None:
        d = d[d["match_id"].isin(match_ids)]
    if phases:
        d = d[d["over_phase"].astype(str).isin(phases)]
    if innings:
        d = d[d["innings"].isin(innings)]
    return d


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — MATCH SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def match_summary_view(
    matches: pd.DataFrame,
    id2name: dict,
) -> pd.DataFrame:
    """
    Human-readable match summary table.
    Returns one row per match with team names, result, margin, venue, season.
    """
    m = matches.copy()

    def _tid(x):
        try:
            return id2name.get(int(x), str(x))
        except (ValueError, TypeError):
            return str(x)

    m["Team 1"]  = m["team1"].map(_tid)
    m["Team 2"]  = m["team2"].map(_tid)
    m["Winner"]  = m["match_winner"].map(_tid)
    m["Toss"]    = m["toss_winner"].map(_tid)

    # Margin string
    def _margin(row):
        if pd.notna(row.get("win_by_runs")) and row["win_by_runs"] > 0:
            return f"{int(row['win_by_runs'])} runs"
        if pd.notna(row.get("win_by_wickets")) and row["win_by_wickets"] > 0:
            return f"{int(row['win_by_wickets'])} wkts"
        return "N/R / Tie"

    m["Margin"]  = m.apply(_margin, axis=1)
    m["1st Inn Score"] = m["first_innings_total"].fillna(0).astype(int)

    cols = ["match_id","season","date","venue","Team 1","Team 2",
            "Winner","Margin","1st Inn Score","toss_decision"]
    avail = [c for c in cols if c in m.columns]
    return m[avail].rename(columns={"match_id":"Match ID","season":"Season",
                                    "date":"Date","venue":"Venue",
                                    "toss_decision":"Toss Decision"})


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — BATTER SCORECARD
# ─────────────────────────────────────────────────────────────────────────────

def batter_scorecard(
    deliveries: pd.DataFrame,
    players: pd.DataFrame,
    min_balls: int = 50,
) -> pd.DataFrame:
    """
    Aggregated batter stats over the filtered delivery set.
    Returns: batter, innings, runs, balls, SR, avg, 4s, 6s, boundary%, 50s, 100s, dot%.
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()

    agg = (
        d.groupby("batter_id")
        .agg(
            innings=("match_id", "nunique"),
            runs=("batter_runs", "sum"),
            balls=("batter_runs", "count"),
            fours=(    "batter_runs", lambda x: (x == 4).sum()),
            sixes=(    "batter_runs", lambda x: (x == 6).sum()),
            dots=(     "batter_runs", lambda x: (x == 0).sum()),
            dismissals=("is_wicket", "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["balls"] >= min_balls]
    agg["SR"]          = (agg["runs"] / agg["balls"] * 100).round(1)
    agg["dot_pct"]     = (agg["dots"] / agg["balls"] * 100).round(1)
    agg["avg"]         = (agg["runs"] / agg["dismissals"].replace(0, np.nan)).round(1)
    agg["boundary_pct"]= ((agg["fours"] + agg["sixes"]) / agg["balls"] * 100).round(1)

    # Per-innings totals for 50s and 100s
    inn_scores = (
        d.groupby(["batter_id", "match_id"])["batter_runs"]
        .sum()
        .reset_index(name="inn_score")
    )
    milestones = (
        inn_scores.groupby("batter_id")
        .agg(
            fifties= ("inn_score", lambda x: ((x >= 50) & (x < 100)).sum()),
            hundreds=("inn_score", lambda x: (x >= 100).sum()),
        )
        .reset_index()
    )
    agg = agg.merge(milestones, on="batter_id", how="left")
    agg["fifties"]  = agg["fifties"].fillna(0).astype(int)
    agg["hundreds"] = agg["hundreds"].fillna(0).astype(int)

    pid2name = dict(zip(players["player_id"], players["player_name"]))
    agg["Batter"] = agg["batter_id"].map(lambda x: pid2name.get(x, str(x)))

    return (
        agg[["Batter","innings","runs","balls","SR","avg","fours","sixes",
             "boundary_pct","fifties","hundreds","dot_pct"]]
        .rename(columns={"innings":"Innings","runs":"Runs","balls":"Balls",
                         "SR":"Strike Rate","avg":"Average","fours":"4s","sixes":"6s",
                         "boundary_pct":"Boundary %","fifties":"50s",
                         "hundreds":"100s","dot_pct":"Dot %"})
        .sort_values("Runs", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — BOWLER SCORECARD
# ─────────────────────────────────────────────────────────────────────────────

def bowler_scorecard(
    deliveries: pd.DataFrame,
    players: pd.DataFrame,
    min_balls: int = 60,
) -> pd.DataFrame:
    """
    Aggregated bowler stats over the filtered delivery set.
    Returns: bowler, overs, runs conceded, wickets, economy, SR, avg.
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()

    agg = (
        d.groupby("bowler_id")
        .agg(
            balls=("total_runs", "count"),
            runs_conceded=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            dots=("total_runs", lambda x: (x == 0).sum()),
        )
        .reset_index()
    )
    agg = agg[agg["balls"] >= min_balls]
    agg["overs"]   = (agg["balls"] // 6 + (agg["balls"] % 6) / 10).round(1)
    agg["economy"] = (agg["runs_conceded"] / agg["balls"] * 6).round(2)
    agg["bowl_SR"] = (agg["balls"] / agg["wickets"].replace(0, np.nan)).round(1)
    agg["bowl_avg"]= (agg["runs_conceded"] / agg["wickets"].replace(0, np.nan)).round(1)
    agg["dot_pct"] = (agg["dots"] / agg["balls"] * 100).round(1)

    pid2name = dict(zip(players["player_id"], players["player_name"]))
    agg["Bowler"] = agg["bowler_id"].map(lambda x: pid2name.get(x, str(x)))

    return (
        agg[["Bowler","overs","runs_conceded","wickets","economy","bowl_SR","bowl_avg","dot_pct"]]
        .rename(columns={"overs":"Overs","runs_conceded":"Runs","wickets":"Wickets",
                         "economy":"Economy","bowl_SR":"Bowl SR","bowl_avg":"Average",
                         "dot_pct":"Dot %"})
        .sort_values("Wickets", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — VENUE SCORECARD
# ─────────────────────────────────────────────────────────────────────────────

def venue_scorecard(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Per-venue aggregated stats from filtered matches.
    """
    completed = matches[matches["is_completed"]].copy()
    completed["chase_won"] = completed["win_by_wickets"].notna() & (completed["win_by_wickets"] > 0)

    v = (
        completed.groupby("venue")
        .agg(
            matches=("match_id", "count"),
            avg_1st_inn=("first_innings_total", "mean"),
            avg_target=("target", "mean"),
            chase_wins=("chase_won", "sum"),
        )
        .reset_index()
    )
    v["chase_pct"] = (v["chase_wins"] / v["matches"] * 100).round(1)
    v["avg_1st_inn"] = v["avg_1st_inn"].round(1)
    v["avg_target"]  = v["avg_target"].round(1)

    return (
        v.rename(columns={"venue":"Venue","matches":"Matches",
                          "avg_1st_inn":"Avg 1st Inn","avg_target":"Avg Target",
                          "chase_pct":"Chase Win %"})
        [["Venue","Matches","Avg 1st Inn","Avg Target","Chase Win %"]]
        .sort_values("Matches", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 5 — PHASE SCORING BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def phase_breakdown(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Per-phase (powerplay/middle/death) scoring stats across filtered deliveries.
    """
    d = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    phase_order = {"powerplay": 0, "middle": 1, "death": 2}

    agg = (
        d.groupby("over_phase")
        .agg(
            balls=("batter_runs", "count"),
            runs=("batter_runs", "sum"),
            boundaries=("batter_runs", lambda x: ((x == 4) | (x == 6)).sum()),
            sixes=("batter_runs", lambda x: (x == 6).sum()),
            dots=("batter_runs", lambda x: (x == 0).sum()),
            wickets=("is_wicket", "sum"),
        )
        .reset_index()
    )
    agg["run_rate"]    = (agg["runs"] / agg["balls"] * 6).round(2)
    agg["boundary_pct"]= (agg["boundaries"] / agg["balls"] * 100).round(1)
    agg["dot_pct"]     = (agg["dots"] / agg["balls"] * 100).round(1)
    agg["wkt_per_over"]= (agg["wickets"] / agg["balls"] * 6).round(3)
    agg["sort_key"]    = agg["over_phase"].map(phase_order).fillna(9)

    return (
        agg.sort_values("sort_key")
        [["over_phase","balls","runs","run_rate","boundary_pct","dot_pct","wickets","wkt_per_over"]]
        .rename(columns={
            "over_phase":"Phase","balls":"Balls","runs":"Runs",
            "run_rate":"Run Rate","boundary_pct":"Boundary %",
            "dot_pct":"Dot %","wickets":"Wickets","wkt_per_over":"Wkts/Over"
        })
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTITY COMPARISON (2-3 batters or bowlers)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PLAYER RECORDS ENGINE — auto-surface superlatives (all-time, full dataset)
# ─────────────────────────────────────────────────────────────────────────────

def get_player_records(
    player_id: int,
    deliveries: pd.DataFrame,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    mode: str = "batter",   # "batter" or "bowler"
    top_n: int = 3,         # rank threshold to qualify as a record
) -> list[dict]:
    """
    Auto-detect notable records for a player using the full dataset.
    Returns a list of dicts: {icon, label, value, colour}
    Empty list = player holds no top-N records worth showing.
    """
    d   = deliveries[~deliveries["is_super_over"] & deliveries["is_legal_ball"]].copy()
    pid2name = dict(zip(players["player_id"], players["player_name"]))
    records  = []

    def _rank(series: pd.Series, pid: int, ascending=False) -> int:
        ranked = series.rank(method="min", ascending=ascending)
        return int(ranked.get(pid, 9999))

    if mode == "batter":
        # ── All-time runs rank ────────────────────────────────────────────
        all_runs = d.groupby("batter_id")["batter_runs"].sum()
        rk = _rank(all_runs, player_id)
        if rk <= top_n:
            records.append({
                "icon": "military_tech", "colour": "#F57F17",
                "label": f"#{rk} all-time run scorer in IPL history",
                "value": f"{int(all_runs.get(player_id, 0)):,} runs",
            })

        # ── All-time sixes ────────────────────────────────────────────────
        all_sixes = d.groupby("batter_id").apply(lambda x: (x["batter_runs"] == 6).sum())
        rk = _rank(all_sixes, player_id)
        if rk <= top_n:
            records.append({
                "icon": "rocket_launch", "colour": "#C62828",
                "label": f"#{rk} six-hitter of all time in IPL",
                "value": f"{int(all_sixes.get(player_id, 0))} sixes",
            })

        # ── All-time fours ────────────────────────────────────────────────
        all_fours = d.groupby("batter_id").apply(lambda x: (x["batter_runs"] == 4).sum())
        rk = _rank(all_fours, player_id)
        if rk <= top_n:
            records.append({
                "icon": "bolt", "colour": "#1565C0",
                "label": f"#{rk} boundary hitter (4s) in IPL history",
                "value": f"{int(all_fours.get(player_id, 0))} fours",
            })

        # ── Highest individual innings ────────────────────────────────────
        inn_scores = d.groupby(["match_id", "batter_id"])["batter_runs"].sum()
        player_inn = inn_scores.xs(player_id, level="batter_id") if player_id in inn_scores.index.get_level_values("batter_id") else pd.Series([], dtype=float)
        if len(player_inn) > 0:
            hs = int(player_inn.max())
            all_hs = inn_scores.groupby("batter_id").max()
            rk = _rank(all_hs, player_id)
            if rk <= top_n and hs >= 80:
                records.append({
                    "icon": "emoji_events", "colour": "#2E7D32",
                    "label": f"#{rk} highest individual innings in IPL history",
                    "value": f"{hs} runs",
                })

        # ── Man of the Match count ────────────────────────────────────────
        mom_counts = matches["player_of_match"].value_counts()
        mom_pid    = mom_counts.get(player_id, 0)
        if mom_pid > 0:
            rk = int((mom_counts > mom_pid).sum()) + 1
            if rk <= top_n:
                records.append({
                    "icon": "star", "colour": "#E65100",
                    "label": f"#{rk} most Player of the Match awards in IPL",
                    "value": f"{int(mom_pid)} awards",
                })

        # ── Death-over specialist (most runs in death overs) ───────────────
        death_runs = d[d["over_phase"] == "death"].groupby("batter_id")["batter_runs"].sum()
        rk = _rank(death_runs, player_id)
        if rk <= top_n:
            records.append({
                "icon": "local_fire_department", "colour": "#C62828",
                "label": f"#{rk} death-overs run scorer in IPL",
                "value": f"{int(death_runs.get(player_id, 0))} runs (ov 16–20)",
            })

        # ── Powerplay aggressor (best SR in PP, min 200 balls) ────────────
        pp = d[d["over_phase"] == "powerplay"].groupby("batter_id").agg(
            r=("batter_runs", "sum"), b=("batter_runs", "count")
        )
        pp = pp[pp["b"] >= 200]
        pp["sr"] = pp["r"] / pp["b"] * 100
        if player_id in pp.index:
            rk = _rank(pp["sr"], player_id)
            if rk <= top_n:
                records.append({
                    "icon": "speed", "colour": "#4527A0",
                    "label": f"#{rk} best powerplay strike rate in IPL (min 200 balls)",
                    "value": f"{pp.loc[player_id, 'sr']:.1f} SR",
                })

        # ── Consistency: most 50+ innings ─────────────────────────────────
        fifty_plus = inn_scores.groupby("batter_id").apply(lambda x: (x >= 50).sum())
        rk = _rank(fifty_plus, player_id)
        if rk <= top_n:
            records.append({
                "icon": "workspace_premium", "colour": "#00695C",
                "label": f"#{rk} most 50+ scores in IPL history",
                "value": f"{int(fifty_plus.get(player_id, 0))} innings",
            })

    else:  # bowler
        # ── All-time wickets rank ─────────────────────────────────────────
        all_wkts = d.groupby("bowler_id")["is_wicket"].sum()
        rk = _rank(all_wkts, player_id)
        if rk <= top_n:
            records.append({
                "icon": "military_tech", "colour": "#F57F17",
                "label": f"#{rk} all-time wicket taker in IPL history",
                "value": f"{int(all_wkts.get(player_id, 0))} wickets",
            })

        # ── Best bowling figures in a single innings ───────────────────────
        bowl_inn = d.groupby(["match_id", "innings", "bowler_id"]).agg(
            w=("is_wicket", "sum"), r=("total_runs", "sum")
        ).reset_index()
        pbowl = bowl_inn[bowl_inn["bowler_id"] == player_id]
        if len(pbowl) > 0:
            best_row = pbowl.sort_values(["w", "r"], ascending=[False, True]).iloc[0]
            bw, br   = int(best_row["w"]), int(best_row["r"])
            # Rank vs all players' best innings figures
            all_best = bowl_inn.sort_values(["w", "r"], ascending=[False, True]).groupby("bowler_id").first()
            all_best_wkts = all_best["w"]
            rk = _rank(all_best_wkts, player_id)
            if rk <= top_n and bw >= 4:
                records.append({
                    "icon": "emoji_events", "colour": "#C62828",
                    "label": f"#{rk} best single-innings bowling figures in IPL",
                    "value": f"{bw}/{br}",
                })

        # ── Most dot balls bowled ─────────────────────────────────────────
        dots = d.groupby("bowler_id").apply(lambda x: (x["total_runs"] == 0).sum())
        rk   = _rank(dots, player_id)
        if rk <= top_n:
            records.append({
                "icon": "block", "colour": "#1565C0",
                "label": f"#{rk} most dot balls bowled in IPL history",
                "value": f"{int(dots.get(player_id, 0))} dots",
            })

        # ── Powerplay specialist (most PP wickets) ─────────────────────────
        pp_wkts = d[d["over_phase"] == "powerplay"].groupby("bowler_id")["is_wicket"].sum()
        rk = _rank(pp_wkts, player_id)
        if rk <= top_n:
            records.append({
                "icon": "local_fire_department", "colour": "#E65100",
                "label": f"#{rk} powerplay wicket taker in IPL",
                "value": f"{int(pp_wkts.get(player_id, 0))} wickets (ov 1–6)",
            })

        # ── Death-over economy (min 300 balls) ───────────────────────────
        death_b = d[d["over_phase"] == "death"].groupby("bowler_id").agg(
            r=("total_runs", "sum"), b=("total_runs", "count")
        )
        death_b = death_b[death_b["b"] >= 300]
        death_b["eco"] = death_b["r"] / death_b["b"] * 6
        if player_id in death_b.index:
            rk = _rank(death_b["eco"], player_id, ascending=True)  # lower is better
            if rk <= top_n:
                records.append({
                    "icon": "shield", "colour": "#2E7D32",
                    "label": f"#{rk} best death-over economy in IPL (min 300 balls)",
                    "value": f"{death_b.loc[player_id, 'eco']:.2f} ER",
                })

        # ── Hat-tricks ────────────────────────────────────────────────────
        wk = d[d["is_wicket"] == True].copy()
        wk = wk[wk["wicket_kind"].astype(str).str.lower() != "run out"]
        wk = wk.sort_values(["match_id", "innings", "bowler_id", "legal_balls_bowled"])
        wk["prev1"] = wk.groupby(["match_id", "innings", "bowler_id"])["legal_balls_bowled"].shift(1)
        wk["prev2"] = wk.groupby(["match_id", "innings", "bowler_id"])["legal_balls_bowled"].shift(2)
        wk["is_ht"] = (wk["legal_balls_bowled"] - wk["prev1"] == 1) & (wk["prev1"] - wk["prev2"] == 1)
        ht_count = int(wk[wk["bowler_id"] == player_id]["is_ht"].sum())
        if ht_count >= 1:
            records.append({
                "icon": "whatshot", "colour": "#C62828",
                "label": f"Hat-trick{'s' if ht_count > 1 else ''} in IPL",
                "value": f"{ht_count}x hat-trick",
            })

        # ── Man of the Match count ────────────────────────────────────────
        mom_counts = matches["player_of_match"].value_counts()
        mom_pid    = mom_counts.get(player_id, 0)
        if mom_pid > 0:
            rk = int((mom_counts > mom_pid).sum()) + 1
            if rk <= top_n:
                records.append({
                    "icon": "star", "colour": "#E65100",
                    "label": f"#{rk} most Player of the Match awards in IPL",
                    "value": f"{int(mom_pid)} awards",
                })

    return records


def compare_bowlers(
    deliveries: pd.DataFrame,
    bowler_ids: list[int],
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side phase-wise comparison of 2–3 bowlers."""
    d = deliveries[
        deliveries["bowler_id"].isin(bowler_ids) &
        ~deliveries["is_super_over"] &
        deliveries["is_legal_ball"]
    ].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    pid2name = dict(zip(players["player_id"], players["player_name"]))

    rows = []
    for bid in bowler_ids:
        bd = d[d["bowler_id"] == bid]
        for phase in ["powerplay", "middle", "death"]:
            ph = bd[bd["over_phase"] == phase]
            if len(ph) == 0:
                continue
            wickets = int(ph["is_wicket"].sum())
            balls   = len(ph)
            runs    = int(ph["total_runs"].sum())
            rows.append({
                "Bowler":    pid2name.get(bid, str(bid)),
                "Phase":     phase,
                "Balls":     balls,
                "Wickets":   wickets,
                "Runs":      runs,
                "Economy":   round(runs / balls * 6, 2) if balls else 0,
                "Bowl SR":   round(balls / wickets, 1) if wickets else None,
                "Dot %":     round((ph["total_runs"] == 0).sum() / balls * 100, 1) if balls else 0,
            })
    return pd.DataFrame(rows)


def compare_batters(
    deliveries: pd.DataFrame,
    batter_ids: list[int],
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side phase-wise comparison of 2–3 batters."""
    d = deliveries[
        deliveries["batter_id"].isin(batter_ids) &
        ~deliveries["is_super_over"] &
        deliveries["is_legal_ball"]
    ].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    pid2name = dict(zip(players["player_id"], players["player_name"]))

    rows = []
    for bid in batter_ids:
        bd = d[d["batter_id"] == bid]
        for phase in ["powerplay", "middle", "death"]:
            ph = bd[bd["over_phase"] == phase]
            if len(ph) == 0:
                continue
            rows.append({
                "Batter": pid2name.get(bid, str(bid)),
                "Phase": phase,
                "Balls": len(ph),
                "Runs": ph["batter_runs"].sum(),
                "SR": round(ph["batter_runs"].sum() / len(ph) * 100, 1),
                "4s": (ph["batter_runs"] == 4).sum(),
                "6s": (ph["batter_runs"] == 6).sum(),
                "Dot %": round((ph["batter_runs"] == 0).sum() / len(ph) * 100, 1),
            })
    return pd.DataFrame(rows)
