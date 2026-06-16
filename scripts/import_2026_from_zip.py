"""
Import 2026 IPL matches from Cricsheet JSON zip into our raw CSV files,
then re-run the foundation pipeline to rebuild Parquet files.

Usage:  python -m scripts.import_2026_from_zip
"""
import sys, json, zipfile
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
ZIP_PATH = ROOT / "archive (2).zip"

# ── Load reference tables ────────────────────────────────────────────────────
teams_df   = pd.read_csv(RAW / "teams_data.csv")
aliases_df = pd.read_csv(RAW / "team_aliases.csv")
players_df = pd.read_csv(RAW / "players-data-updated.csv")

# Build team name → team_id map
team_name_to_id: dict[str, int] = {}
for _, row in teams_df.iterrows():
    team_name_to_id[row["team_name"].strip()] = int(row["team_id"])
for _, row in aliases_df.iterrows():
    team_name_to_id[row["alias_name"].strip()] = int(row["team_id"])

# Build player name → {player_id, bat_style, bowl_style}
player_lookup: dict[str, dict] = {}
for _, row in players_df.iterrows():
    player_lookup[str(row["player_name"]).strip()] = {
        "player_id": int(row["player_id"]),
        "bat_style":  str(row["bat_style"])  if pd.notna(row["bat_style"])  else "",
        "bowl_style": str(row["bowl_style"]) if pd.notna(row["bowl_style"]) else "",
    }

def resolve_team(name: str) -> int | None:
    return team_name_to_id.get(name.strip())

def resolve_player(name: str) -> dict:
    return player_lookup.get(name.strip(), {
        "player_id": None, "bat_style": "", "bowl_style": ""
    })

# ── Load existing raw CSVs ───────────────────────────────────────────────────
matches_csv  = pd.read_csv(RAW / "ipl_matches_data.csv")
balls_csv    = pd.read_csv(RAW / "ball_by_ball_data.csv")

existing_dates = set(
    matches_csv[matches_csv["season_id"] == 2026]["match_date"].tolist()
)
max_match_id = int(matches_csv["match_id"].max())
print(f"Existing 2026 dates: {len(existing_dates)}")
print(f"Max existing match_id: {max_match_id}")

# ── Parse zip ───────────────────────────────────────────────────────────────
zf = zipfile.ZipFile(ZIP_PATH)

new_match_rows = []
new_ball_rows  = []

current_id = max_match_id + 1

files_2026 = []
for fname in zf.namelist():
    try:
        raw = zf.read(fname)
        if not raw.strip(): continue
        data = json.loads(raw)
        if str(data["info"].get("season")) == "2026":
            date = data["info"]["dates"][0]
            if date not in existing_dates:
                files_2026.append((date, data))
    except Exception as e:
        print(f"  SKIP {fname}: {e}")

files_2026.sort(key=lambda x: x[0])
print(f"New matches to import: {len(files_2026)}")

for date, data in files_2026:
    info    = data["info"]
    innings = data["innings"]

    teams   = info["teams"]           # [team1, team2]
    t1_id   = resolve_team(teams[0])
    t2_id   = resolve_team(teams[1])
    if t1_id is None or t2_id is None:
        print(f"  WARNING: unresolved team in {date} — {teams}")
        continue

    toss_winner_name = info["toss"]["winner"]
    toss_winner_id   = resolve_team(toss_winner_name)
    toss_decision    = info["toss"]["decision"]  # 'field' or 'bat'

    outcome = info.get("outcome", {})
    winner_name = outcome.get("winner")
    winner_id   = resolve_team(winner_name) if winner_name else None
    win_by      = outcome.get("by", {})
    win_by_runs    = float(win_by.get("runs", 0)) or np.nan
    win_by_wickets = float(win_by.get("wickets", 0)) or np.nan
    result = "win" if winner_name else ("no result" if "method" in outcome else "tie")

    pom_names = info.get("player_of_match", [])
    pom_id    = resolve_player(pom_names[0])["player_id"] if pom_names else np.nan

    match_id = current_id
    current_id += 1

    event = info.get("event", {})

    new_match_rows.append({
        "match_id":       match_id,
        "season_id":      2026,
        "balls_per_over": info.get("balls_per_over", 6),
        "city":           info.get("city", ""),
        "match_date":     date,
        "event_name":     event.get("name", "Indian Premier League"),
        "match_number":   event.get("match_number", np.nan),
        "gender":         info.get("gender", "male"),
        "match_type":     info.get("match_type", "T20"),
        "format":         info.get("match_type", "T20"),
        "overs":          info.get("overs", 20),
        "season":         2026,
        "team_type":      info.get("team_type", "club"),
        "venue":          info.get("venue", ""),
        "toss_winner":    toss_winner_id,
        "team1":          t1_id,
        "team2":          t2_id,
        "toss_decision":  toss_decision,
        "match_winner":   winner_id,
        "win_by_runs":    win_by_runs,
        "win_by_wickets": win_by_wickets,
        "result":         result,
        "player_of_match": pom_id,
    })

    # ── Ball-by-ball ────────────────────────────────────────────────────────
    inn_number = 0
    for inn in innings:
        is_super_over = bool(inn.get("super_over", False))
        if not is_super_over:
            inn_number += 1

        batting_team  = inn["team"]
        batting_id    = resolve_team(batting_team)
        bowling_id    = t2_id if batting_id == t1_id else t1_id

        for over_obj in inn["overs"]:
            ov = int(over_obj["over"])    # 0-indexed
            ball_num = 0
            for delivery in over_obj["deliveries"]:
                batter      = delivery["batter"]
                bowler      = delivery["bowler"]
                non_striker = delivery["non_striker"]
                runs        = delivery["runs"]
                extras_d    = delivery.get("extras", {})
                wickets_d   = delivery.get("wickets", [])

                batter_info  = resolve_player(batter)
                bowler_info  = resolve_player(bowler)

                is_wide   = "wides"   in extras_d
                is_noball = "noballs" in extras_d
                is_legbye = "legbyes" in extras_d
                is_bye    = "byes"    in extras_d
                is_penalty= "penalty" in extras_d
                is_wicket = len(wickets_d) > 0

                player_out = wickets_d[0]["player_out"] if is_wicket else np.nan
                wicket_kind= wickets_d[0]["kind"]       if is_wicket else np.nan
                fielders   = (
                    ",".join(f.get("name","") for f in wickets_d[0].get("fielders",[]))
                    if is_wicket else np.nan
                )

                new_ball_rows.append({
                    "season_id":      2026,
                    "match_id":       match_id,
                    "batter":         batter,
                    "bowler":         bowler,
                    "non_striker":    non_striker,
                    "team_batting":   batting_id,
                    "team_bowling":   bowling_id,
                    "over_number":    ov,
                    "ball_number":    ball_num,
                    "batter_runs":    int(runs["batter"]),
                    "extras":         int(runs["extras"]),
                    "total_runs":     int(runs["total"]),
                    "batsman_type":   batter_info["bat_style"],
                    "bowler_type":    bowler_info["bowl_style"],
                    "player_out":     player_out,
                    "fielders_involved": fielders,
                    "is_wicket":      is_wicket,
                    "is_wide_ball":   is_wide,
                    "is_no_ball":     is_noball,
                    "is_leg_bye":     is_legbye,
                    "is_bye":         is_bye,
                    "is_penalty":     is_penalty,
                    "wide_ball_runs":  int(extras_d.get("wides",   0)),
                    "no_ball_runs":    int(extras_d.get("noballs",  0)),
                    "leg_bye_runs":    int(extras_d.get("legbyes",  0)),
                    "bye_runs":        int(extras_d.get("byes",     0)),
                    "penalty_runs":    int(extras_d.get("penalty",  0)),
                    "wicket_kind":     wicket_kind,
                    "is_super_over":   is_super_over,
                    "innings":         (inn_number if not is_super_over
                                        else inn_number + 10),  # flag super overs
                })
                if not is_wide:   # ball counter only on legal + noball
                    ball_num += 1

print(f"\nNew match rows:  {len(new_match_rows)}")
print(f"New ball rows:   {len(new_ball_rows)}")

if not new_match_rows:
    print("Nothing to add — exiting.")
    sys.exit(0)

# ── Append to CSVs ───────────────────────────────────────────────────────────
new_m_df = pd.DataFrame(new_match_rows)
new_b_df = pd.DataFrame(new_ball_rows)

# Ensure column order matches original
new_m_df = new_m_df[matches_csv.columns]
new_b_df = new_b_df[balls_csv.columns]

updated_matches = pd.concat([matches_csv, new_m_df], ignore_index=True)
updated_balls   = pd.concat([balls_csv,   new_b_df], ignore_index=True)

updated_matches.to_csv(RAW / "ipl_matches_data.csv",  index=False)
updated_balls.to_csv(  RAW / "ball_by_ball_data.csv", index=False)

print(f"\nMatches CSV now has {len(updated_matches)} rows")
print(f"Balls CSV now has   {len(updated_balls):,} rows")

# Verification
m2026 = updated_matches[updated_matches["season_id"]==2026]
print(f"2026 matches: {len(m2026)}  (was {len(existing_dates)})")
print(f"Date range:   {m2026['match_date'].min()} to {m2026['match_date'].max()}")

# ── Re-run pipeline ──────────────────────────────────────────────────────────
print("\nRe-running Module 0 pipeline...")
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "src.module_0_foundation"],
    cwd=str(ROOT), capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    print("\nPipeline failed — CSVs are already updated, re-run manually with:")
    print("  python -m src.module_0_foundation")
else:
    print("\nPipeline complete. Parquet files rebuilt.")
