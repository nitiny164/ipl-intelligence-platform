# Data Quality Report — IPL Intelligence Platform
  Generated: 2026-06-14 20:32

  ## Source Files

  | File | Rows |
  |---|---|
  | ipl_matches_data.csv | 1,243 |
  | ball_by_ball_data.csv | 295,732 |
  | players-data-updated.csv | 799 |
  | teams_data.csv | 16 |
  | team_aliases.csv | 46 |

  ## Dataset Statistics

  - **Matches:** 1,243
  - **Deliveries:** 295,732
  - **Players:** 799
  - **Seasons:** 19 (2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)
  - **Season 2020/21 → 2020:** normalised with `is_uae_season = True` flag

  ## Match Results

  | Result | Count |
  |---|---|
  | Completed (win) | 1218 |
  | Tie (super-over decided) | 16 |
  | Tie (no super-over result) | 0 |
  | No result | 9 |

  ## Cleaning Decisions

  ### Season normalisation
  The string `"2020/21"` is mapped to integer `2020`. A boolean column `is_uae_season`
  is added so any analysis can filter or annotate the UAE-hosted season separately.

  ### Super-over deliveries
  175 super-over deliveries exist in the raw data.
  They are **retained** in `deliveries.parquet` with `is_super_over = True` and
  **excluded** by all win-probability model queries and run-rate aggregations via
  filter `is_super_over == False`. Tie outcomes are resolved to the super-over winner
  in `matches.parquet` via `match_winner`.

  ### No-result matches
  9 no-result matches are retained with `is_no_result = True`
  and excluded from win/loss analysis. They are included in match counts and venue aggregations.

  ### Player name resolution
  The ball-by-ball file uses player **names**; the matches file uses player **IDs**.
  A `player_name → player_id` bridge was built from the player master.

  **Unresolved names (12):**
- `Akshat Raghuwanshi`
- `BJ Dwarshuis`
- `N Sindhu`
- `R Smaran`
- `RS Ghosh`
- `Raghu Sharma`
- `Ravi Singh`
- `SR Dubey`
- `SS Mishra`
- `Salil Arora`
- `T Dahiya`
- `T Vijay`

  ### Franchise lineage
  Two analytical views are maintained:
  - **`team_id` view:** season-accurate analysis using the exact team identity per season.
  - **`lineage_id` view:** historical trend analysis treating rebrands as one continuous franchise.
    - Deccan Chargers (id 1068) → Sunrisers Hyderabad (id 2)
    - Rising Pune Supergiant (id 4) → Rising Pune Supergiants (id 3604) *(spelling normalisation)*
    - All other teams: `lineage_id = team_id`
    - Gujarat Lions (id 5) and Gujarat Titans (id 615) remain **separate** (different franchises).

  ## Enriched Columns Added

  ### deliveries.parquet (beyond raw columns)
  | Column | Description |
  |---|---|
  | `over_phase` | `powerplay` (overs 1–6) / `middle` (7–15) / `death` (16–20) |
  | `over_number_1idx` | 1-based over number (human-readable) |
  | `running_score` | Cumulative runs *before* this ball (exclusive) |
  | `running_wickets` | Cumulative wickets *before* this ball (exclusive) |
  | `legal_balls_bowled` | Non-wide, non-no-ball deliveries bowled so far in innings |
  | `first_innings_total` | Total runs scored in innings 1 of this match |
  | `target` | `first_innings_total + 1` (innings 2 only) |
  | `balls_remaining` | `120 − legal_balls_bowled` (innings 2 only) |
  | `runs_remaining` | `target − running_score` (innings 2 only) |
  | `required_run_rate` | `runs_remaining / (balls_remaining / 6)` (innings 2 only) |
  | `current_run_rate` | `running_score / (legal_balls_bowled / 6)` (innings 2 only) |
  | `run_rate_gap` | `current_run_rate − required_run_rate` |
  | `batter_id`, `bowler_id`, `non_striker_id` | Resolved player IDs |

  ### matches.parquet (beyond raw columns)
  | Column | Description |
  |---|---|
  | `is_completed` | True if result == "win" |
  | `is_tie` | True if result == "tie" |
  | `is_no_result` | True if result == "no result" |
  | `is_super_over_decided` | True if tie was resolved via super over |
  | `is_uae_season` | True if season == 2020 (UAE-hosted) |
  | `first_innings_total` | Joined from deliveries |
  | `target` | `first_innings_total + 1` |

  ## Assumptions & Limitations

  - **No auction/salary data:** all recommendations are framed around performance and strategy, not player valuation.
  - **No injury records:** player career gaps are visible but not explainable.
  - **No DRS/review data:** umpire decisions are taken as ground truth.
  - **Win probability model** is trained on completed matches only; DLS-adjusted matches are not explicitly handled.
  - **Over phase boundaries** are fixed (PP: 1–6, Middle: 7–15, Death: 16–20) and do not account for strategic timeouts or fielding restrictions in the first two overs.

  ## Processed Files

  | File | Key Contents |
  |---|---|
  | `matches.parquet` | Cleaned + enriched match records |
  | `deliveries.parquet` | Cleaned + enriched ball-by-ball records |
  | `players.parquet` | Player master with resolved IDs |
  | `teams.parquet` | Canonical teams with lineage mapping |
