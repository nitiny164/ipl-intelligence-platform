"""
module_2b_h2h.py — Team vs Team head-to-head analytics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def h2h_summary(matches: pd.DataFrame, team_a: int, team_b: int) -> dict:
    mask = (
        matches["is_completed"] &
        (
            ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
            ((matches["team1"] == team_b) & (matches["team2"] == team_a))
        )
    )
    df = matches[mask].sort_values("match_date")
    if df.empty:
        return {}

    total  = len(df)
    a_wins = int((df["match_winner"] == team_a).sum())
    b_wins = int((df["match_winner"] == team_b).sum())
    no_res = total - a_wins - b_wins

    last5        = df.tail(5)["match_winner"].tolist()
    last5_labels = ["A" if w == team_a else ("B" if w == team_b else "NR") for w in last5]

    by_runs    = df[(df["win_by_runs"] > 0) & df["win_by_runs"].notna()]
    by_wickets = df[(df["win_by_wickets"] > 0) & df["win_by_wickets"].notna()]

    def _biggest(sub_df, col, winner_id):
        rows = sub_df[sub_df["match_winner"] == winner_id]
        return int(rows[col].max()) if not rows.empty else None

    return {
        "total":          total,
        "a_wins":         a_wins,
        "b_wins":         b_wins,
        "no_result":      no_res,
        "a_win_pct":      round(a_wins / total * 100, 1),
        "b_win_pct":      round(b_wins / total * 100, 1),
        "last5":          last5_labels,
        "a_biggest_runs": _biggest(by_runs,    "win_by_runs",    team_a),
        "b_biggest_runs": _biggest(by_runs,    "win_by_runs",    team_b),
        "a_biggest_wkts": _biggest(by_wickets, "win_by_wickets", team_a),
        "b_biggest_wkts": _biggest(by_wickets, "win_by_wickets", team_b),
        "first_meeting":  str(df["match_date"].min())[:10],
        "last_meeting":   str(df["match_date"].max())[:10],
    }


def h2h_season_trend(matches: pd.DataFrame, team_a: int, team_b: int) -> pd.DataFrame:
    mask = (
        matches["is_completed"] &
        (
            ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
            ((matches["team1"] == team_b) & (matches["team2"] == team_a))
        )
    )
    df = matches[mask].sort_values("match_date").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    df["a_win"]                  = (df["match_winner"] == team_a).astype(int)
    df["cumulative_a_wins"]      = df["a_win"].cumsum()
    df["match_num"]              = range(1, len(df) + 1)
    df["cumulative_a_win_pct"]   = (df["cumulative_a_wins"] / df["match_num"] * 100).round(1)
    df["cumulative_b_win_pct"]   = (100 - df["cumulative_a_win_pct"]).round(1)
    df["winner_flag"]            = df["match_winner"].apply(
        lambda w: "A" if w == team_a else ("B" if w == team_b else "NR")
    )
    return df[["match_date", "season", "match_num", "cumulative_a_win_pct",
               "cumulative_b_win_pct", "winner_flag", "venue",
               "win_by_runs", "win_by_wickets"]].copy()


def h2h_match_history(matches: pd.DataFrame, players: pd.DataFrame,
                      team_a: int, team_b: int, id2name: dict) -> pd.DataFrame:
    mask = (
        ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
        ((matches["team1"] == team_b) & (matches["team2"] == team_a))
    )
    df = matches[mask].sort_values("match_date", ascending=False).copy()

    pid2name = dict(zip(players["player_id"], players["player_name"]))

    df["Winner"]          = df["match_winner"].map(id2name)
    df["Margin"]          = df.apply(
        lambda r: (f"{int(r['win_by_runs'])} runs"
                   if pd.notna(r["win_by_runs"]) and r["win_by_runs"] > 0
                   else (f"{int(r['win_by_wickets'])} wkts"
                         if pd.notna(r["win_by_wickets"]) and r["win_by_wickets"] > 0
                         else "N/R")),
        axis=1,
    )
    df["Player of Match"] = df["player_of_match"].map(pid2name)
    df["Season"]          = df["season"].astype(int)
    df["Date"]            = pd.to_datetime(df["match_date"]).dt.strftime("%d %b %Y")

    return df[["Date", "Season", "venue", "Winner", "Margin",
               "Player of Match"]].rename(columns={"venue": "Venue"}).reset_index(drop=True)


def h2h_phase_battle(deliveries: pd.DataFrame, matches: pd.DataFrame,
                     team_a: int, team_b: int) -> dict:
    match_ids = matches[
        matches["is_completed"] &
        (
            ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
            ((matches["team1"] == team_b) & (matches["team2"] == team_a))
        )
    ]["match_id"]

    d = deliveries[
        deliveries["match_id"].isin(match_ids) &
        ~deliveries["is_super_over"] &
        deliveries["is_legal_ball"]
    ].copy()

    phase_order = ["powerplay", "middle", "death"]
    results = {}
    for team_id, key in [(team_a, "team_a"), (team_b, "team_b")]:
        batting = d[d["team_batting"] == team_id]
        if batting.empty:
            results[key] = pd.DataFrame()
            continue

        grp = batting.groupby("over_phase").agg(
            runs       = ("batter_runs", "sum"),
            balls      = ("batter_runs", "count"),
            boundaries = ("batter_runs", lambda x: ((x == 4) | (x == 6)).sum()),
            sixes      = ("batter_runs", lambda x: (x == 6).sum()),
            dots       = ("batter_runs", lambda x: (x == 0).sum()),
            wickets    = ("is_wicket",   "sum"),
        ).reindex(phase_order).reset_index()

        grp["run_rate"]      = (grp["runs"]       / grp["balls"]  * 6).round(2)
        grp["boundary_pct"]  = (grp["boundaries"] / grp["balls"]  * 100).round(1)
        grp["six_pct"]       = (grp["sixes"]      / grp["balls"]  * 100).round(1)
        grp["dot_pct"]       = (grp["dots"]       / grp["balls"]  * 100).round(1)
        grp["wkts_per_over"] = (grp["wickets"]    / grp["balls"]  * 6).round(3)
        results[key] = grp[["over_phase", "run_rate", "boundary_pct",
                             "six_pct", "dot_pct", "wkts_per_over"]]

    return results


def h2h_pom_leaders(matches: pd.DataFrame, players: pd.DataFrame,
                    team_a: int, team_b: int, top_n: int = 8) -> pd.DataFrame:
    mask = (
        matches["is_completed"] &
        (
            ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
            ((matches["team1"] == team_b) & (matches["team2"] == team_a))
        )
    )
    df = matches[mask].dropna(subset=["player_of_match"]).copy()
    pid2name = dict(zip(players["player_id"], players["player_name"]))

    counts = (
        df["player_of_match"].astype(int)
        .value_counts().head(top_n)
        .reset_index()
    )
    counts.columns = ["player_id", "Awards"]
    counts["Player"] = counts["player_id"].map(pid2name).fillna("Unknown")
    return counts[["Player", "Awards"]].reset_index(drop=True)


def h2h_venue_split(matches: pd.DataFrame, team_a: int, team_b: int,
                    id2name: dict) -> pd.DataFrame:
    mask = (
        matches["is_completed"] &
        (
            ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
            ((matches["team1"] == team_b) & (matches["team2"] == team_a))
        )
    )
    df = matches[mask].copy()
    if df.empty:
        return pd.DataFrame()

    grp = df.groupby("venue").agg(
        Matches = ("match_id",      "count"),
        a_wins  = ("match_winner",  lambda x: (x == team_a).sum()),
        b_wins  = ("match_winner",  lambda x: (x == team_b).sum()),
    ).reset_index()

    name_a = id2name.get(team_a, str(team_a))
    name_b = id2name.get(team_b, str(team_b))
    grp[name_a]         = grp["a_wins"]
    grp[name_b]         = grp["b_wins"]
    grp[f"{name_a} Win%"] = (grp["a_wins"] / grp["Matches"] * 100).round(1)
    return (grp[["venue", "Matches", name_a, name_b, f"{name_a} Win%"]]
            .rename(columns={"venue": "Venue"})
            .sort_values("Matches", ascending=False)
            .reset_index(drop=True))


def h2h_toss_impact(matches: pd.DataFrame, team_a: int, team_b: int) -> dict:
    mask = (
        matches["is_completed"] &
        (
            ((matches["team1"] == team_a) & (matches["team2"] == team_b)) |
            ((matches["team1"] == team_b) & (matches["team2"] == team_a))
        )
    )
    df = matches[mask].copy()
    if df.empty:
        return {}

    toss_winner_wins = df[df["toss_winner"] == df["match_winner"]]
    return {
        "total":                len(df),
        "toss_winner_wins":     len(toss_winner_wins),
        "toss_winner_win_pct":  round(len(toss_winner_wins) / len(df) * 100, 1),
        "chase_wins":           int((df["win_by_wickets"] > 0).sum()),
        "bat_first_wins":       int((df["win_by_runs"] > 0).sum()),
    }
