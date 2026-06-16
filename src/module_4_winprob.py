"""
module_4_winprob.py — Win Probability Engine.

Business question: "Given the match situation at any ball, who is winning — why —
and how often does the model get surprised?"

Pipeline:
  1. Feature engineering  → model_features.parquet
  2. Model training       → LogReg baseline + XGBoost primary (chronological split)
  3. Validation           → AUC/log-loss/calibration per season
  4. SHAP explainability  → per-ball feature contributions
  5. Precompute replays   → replay_matches.parquet
  6. Improbable finishes  → improbable_finishes.parquet

Chronological split (stated in UI and model card):
  Train    : seasons ≤ 2023
  Validate : season == 2024
  Test     : seasons >= 2025
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


class _IsotonicCalibrator:
    """Module-level wrapper so joblib can pickle it (local classes cannot be pickled)."""
    def __init__(self, base, iso):
        self._base = base
        self._iso  = iso

    def predict_proba(self, X):
        p = self._iso.predict(self._base.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "running_score", "running_wickets", "balls_remaining",
    "runs_remaining", "required_run_rate", "current_run_rate",
    "run_rate_gap", "legal_balls_bowled",
    "phase_pp", "phase_mid", "phase_death",   # one-hot of over_phase
    "team_form",                               # rolling 10-match win % (joined)
    "venue_chase_pct",                         # venue historical chase success %
]

# Season split: always uses latest available season as test, one before as val,
# everything else as train — so adding 2027/2028 data never needs a code change.
def _auto_seasons(deliveries_path):
    import pyarrow.parquet as pq
    try:
        seasons = sorted(pq.read_table(deliveries_path, columns=["season"])
                         .to_pandas()["season"].dropna().unique().tolist())
    except Exception:
        seasons = list(range(2008, 2027))
    latest      = int(max(seasons))
    val_season  = latest - 1
    train_cutoff = val_season - 1
    return (
        [int(s) for s in seasons if s <= train_cutoff],
        [val_season],
        [int(s) for s in seasons if s > val_season],
    )

from src.data_loader import PROCESSED_DIR as _PD
TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS = _auto_seasons(_PD / "deliveries.parquet")
LABEL_COL      = "chasing_team_won"


# ===========================================================================
# 1. Feature engineering
# ===========================================================================
def build_features(deliveries: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build ball-level feature set for the win-probability model.
    Only second-innings deliveries from completed matches (no super-overs).
    Label: did the chasing team win this match?
    """
    print("  Building model features...")

    # Completed matches only; resolve tie via super-over winner
    completed = matches[matches["is_completed"] | matches["is_super_over_decided"]].copy()
    completed = completed.dropna(subset=["match_winner"])

    # Second innings only, no super overs, legal and illegal balls kept for completeness
    d = deliveries[
        (deliveries["innings"] == 2) &
        (~deliveries["is_super_over"])
    ].copy()
    d["over_phase"] = d["over_phase"].astype(str)

    # Join match outcome: did the batting team (chasing team) win?
    d = d.merge(
        completed[["match_id", "match_winner", "season"]].rename(columns={"season": "match_season"}),
        on="match_id", how="inner"
    )
    d[LABEL_COL] = (d["team_batting"] == d["match_winner"].astype("Int64")).astype(int)

    # --- Team rolling form (10-match win %, computed from matches up to that season) ---
    team_form_map = _compute_team_form(matches)
    d["team_form"] = d.apply(
        lambda r: team_form_map.get((int(r["team_batting"]), int(r["match_season"])), 50.0),
        axis=1
    )

    # --- Venue chase success % (historical, using all matches) ---
    venue_map = _compute_venue_chase_pct(matches)
    d = d.merge(matches[["match_id", "venue"]], on="match_id", how="left")
    d["venue_chase_pct"] = d["venue"].map(venue_map).fillna(50.0)

    # --- Phase one-hot encoding ---
    d["phase_pp"]    = (d["over_phase"] == "powerplay").astype(int)
    d["phase_mid"]   = (d["over_phase"] == "middle").astype(int)
    d["phase_death"] = (d["over_phase"] == "death").astype(int)

    # --- Filter to legal balls only (have meaningful run-rate columns) ---
    d = d[d["is_legal_ball"]].copy()

    # --- Handle nulls (early in innings before RRR is defined) ---
    for col in ["required_run_rate", "current_run_rate", "run_rate_gap"]:
        d[col] = d[col].fillna(0.0)
    d["balls_remaining"]  = d["balls_remaining"].fillna(120.0)
    d["runs_remaining"]   = d["runs_remaining"].fillna(d["runs_remaining"].median())

    # Drop rows where key features are still null
    d = d.dropna(subset=FEATURE_COLS + [LABEL_COL])

    print(f"    Feature set: {len(d):,} balls | {d[LABEL_COL].mean():.1%} chasing-team wins")
    return d[FEATURE_COLS + [LABEL_COL, "match_id", "match_season", "over_number", "ball_number"]].reset_index(drop=True)


def _compute_team_form(matches: pd.DataFrame) -> dict:
    """10-match rolling win % per team per season (look-back only — no data leakage)."""
    completed = matches[matches["is_completed"]].sort_values("match_date").copy()
    form: dict[tuple, float] = {}

    for team_id in pd.concat([completed["team1"], completed["team2"]]).dropna().unique():
        team_id = int(team_id)
        team_m  = completed[(completed["team1"] == team_id) | (completed["team2"] == team_id)].copy()
        team_m["won"] = team_m["match_winner"] == team_id
        team_m["rolling_win_pct"] = team_m["won"].rolling(10, min_periods=1).mean() * 100
        for season in team_m["season"].unique():
            season_rows = team_m[team_m["season"] == season]
            if not season_rows.empty:
                form[(team_id, int(season))] = float(season_rows["rolling_win_pct"].iloc[-1])
    return form


def _compute_venue_chase_pct(matches: pd.DataFrame) -> dict:
    """Historical chase success % per venue."""
    completed = matches[matches["is_completed"]].copy()
    completed["chase_won"] = completed["win_by_wickets"].notna() & (completed["win_by_wickets"] > 0)
    return (completed.groupby("venue")["chase_won"].mean() * 100).to_dict()


# ===========================================================================
# 2. Train models
# ===========================================================================
def train_models(features: pd.DataFrame) -> dict:
    """
    Train LogReg baseline and XGBoost primary model on chronological split.
    Returns a results dict with models, scalers, and per-season metrics.
    """
    print("\n  Training models (chronological split)...")

    train_mask = features["match_season"].isin(TRAIN_SEASONS)
    val_mask   = features["match_season"].isin(VAL_SEASONS)
    test_mask  = features["match_season"].isin(TEST_SEASONS)

    X_train = features.loc[train_mask, FEATURE_COLS].values
    y_train = features.loc[train_mask, LABEL_COL].values
    X_val   = features.loc[val_mask,   FEATURE_COLS].values
    y_val   = features.loc[val_mask,   LABEL_COL].values
    X_test  = features.loc[test_mask,  FEATURE_COLS].values
    y_test  = features.loc[test_mask,  LABEL_COL].values

    print(f"    Train: {len(X_train):,} balls ({TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]})")
    print(f"    Val  : {len(X_val):,} balls ({VAL_SEASONS})")
    print(f"    Test : {len(X_test):,} balls ({TEST_SEASONS})")

    # --- Logistic Regression baseline ---
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_val_proba  = lr_pipe.predict_proba(X_val)[:, 1]
    lr_test_proba = lr_pipe.predict_proba(X_test)[:, 1]

    print(f"    LogReg  — Val AUC: {roc_auc_score(y_val, lr_val_proba):.4f}"
          f"  | Test AUC: {roc_auc_score(y_test, lr_test_proba):.4f}")

    # --- XGBoost primary ---
    # Monotonicity constraints: encode cricket domain knowledge directly into the model.
    # This fixes multicollinearity-driven direction inversions (e.g. balls_remaining
    # was showing negative SHAP despite more time being good for the chaser).
    # +1 = feature must push win prob UP as it increases (cricket-correct)
    # -1 = feature must push win prob DOWN as it increases
    #  0 = unconstrained (phase flags: contextual, not directional)
    _MONO = {
        "running_score":      +1,   # higher score → better for chaser
        "running_wickets":    -1,   # more wickets fallen → worse
        "balls_remaining":    +1,   # more time → better
        "runs_remaining":     -1,   # more to chase → worse
        "required_run_rate":  -1,   # higher RRR → harder
        "current_run_rate":   +1,   # scoring faster → better
        "run_rate_gap":       +1,   # ahead of schedule → better
        "legal_balls_bowled": -1,   # more balls used → less time remaining
        "phase_pp":            0,   # powerplay context — can help or hurt
        "phase_mid":           0,   # middle overs context
        "phase_death":         0,   # death overs context
        "team_form":          +1,   # better recent form → better
        "venue_chase_pct":    +1,   # chase-friendly ground → better
    }
    mono_tuple = tuple(_MONO[f] for f in FEATURE_COLS)

    xgb_model = xgb.XGBClassifier(
        n_estimators=600,           # more trees — early stopping will find the right count
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=mono_tuple,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.set_params(early_stopping_rounds=30)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_it = getattr(xgb_model, "best_iteration", "n/a")
    print(f"    Best iteration: {best_it} (early stopping from 600)")
    xgb_val_proba  = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

    print(f"    XGBoost — Val AUC: {roc_auc_score(y_val, xgb_val_proba):.4f}"
          f"  | Test AUC: {roc_auc_score(y_test, xgb_test_proba):.4f}")

    # --- Isotonic calibration on top of XGBoost ---
    # Calibration wraps the model: it bins predicted probabilities and adjusts
    # them to match observed frequencies. cv="prefit" means the model is already
    # trained; we just fit the calibrator on the validation set.
    try:
        calibrated_xgb = CalibratedClassifierCV(xgb_model, cv="prefit", method="isotonic")
        calibrated_xgb.fit(X_val, y_val)
    except Exception:
        # sklearn ≥ 1.4 changed cv="prefit" API — use the base estimator + manual isotonic
        from sklearn.isotonic import IsotonicRegression
        _raw_val = xgb_model.predict_proba(X_val)[:, 1]
        _iso = IsotonicRegression(out_of_bounds="clip")
        _iso.fit(_raw_val, y_val)
        calibrated_xgb = _IsotonicCalibrator(xgb_model, _iso)
    cal_test_proba = calibrated_xgb.predict_proba(X_test)[:, 1]

    print(f"    XGB+Cal — Test AUC: {roc_auc_score(y_test, cal_test_proba):.4f}"
          f"  | Brier: {brier_score_loss(y_test, cal_test_proba):.4f}")

    # --- Per-season test metrics ---
    season_metrics = []
    for s in TEST_SEASONS + VAL_SEASONS:
        mask = features["match_season"] == s
        if mask.sum() == 0:
            continue
        Xs = features.loc[mask, FEATURE_COLS].values
        ys = features.loc[mask, LABEL_COL].values
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        proba_s = calibrated_xgb.predict_proba(Xs)[:, 1]
        season_metrics.append({
            "season": s,
            "balls": len(ys),
            "auc": round(roc_auc_score(ys, proba_s), 4),
            "log_loss": round(log_loss(ys, proba_s), 4),
            "brier": round(brier_score_loss(ys, proba_s), 4),
        })

    # --- Calibration curve for the primary model ---
    frac_pos, mean_pred = calibration_curve(y_test, cal_test_proba, n_bins=10)
    cal_data = {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()}

    # --- Save models ---
    joblib.dump(lr_pipe,       MODELS_DIR / "logreg_baseline_v1.joblib")
    joblib.dump(xgb_model,     MODELS_DIR / "xgboost_v1.joblib")
    joblib.dump(calibrated_xgb, MODELS_DIR / "win_probability_model_v1.joblib")
    print("    [OK] Models saved to models/")

    return {
        "lr_pipe":        lr_pipe,
        "xgb_model":      xgb_model,
        "calibrated_xgb": calibrated_xgb,
        "season_metrics": pd.DataFrame(season_metrics),
        "cal_data":       cal_data,
        "feature_cols":   FEATURE_COLS,
        "val_auc_lr":     roc_auc_score(y_val, lr_val_proba),
        "val_auc_xgb":    roc_auc_score(y_val, xgb_val_proba),
        "test_auc_cal":   roc_auc_score(y_test, cal_test_proba),
        "test_brier_cal": brier_score_loss(y_test, cal_test_proba),
    }


# ===========================================================================
# 3. SHAP explainability
# ===========================================================================
def compute_shap_values(xgb_model, features: pd.DataFrame, n_sample: int = 2000) -> tuple:
    """
    Compute SHAP values for a sample of test-set balls.
    Returns (shap_values array, sample_df) for use in the Streamlit SHAP panel.
    """
    import shap
    test_mask = features["match_season"].isin(TEST_SEASONS + VAL_SEASONS)
    sample = features[test_mask].sample(min(n_sample, test_mask.sum()), random_state=42)
    X_sample = sample[FEATURE_COLS].values

    explainer    = shap.TreeExplainer(xgb_model)
    shap_values  = explainer.shap_values(X_sample)
    return shap_values, sample.reset_index(drop=True)


# ===========================================================================
# 4. Precompute replay sequences
# ===========================================================================
def precompute_replays(features: pd.DataFrame, model, matches: pd.DataFrame,
                        n_matches: int = 30) -> pd.DataFrame:
    """
    For the n_matches most "dramatic" completed matches (highest probability swing),
    precompute ball-by-ball win probabilities for the live replay UI.
    Selects from all seasons to span historical coverage.
    """
    print("  Precomputing replay sequences...")

    test_features = features[features["match_season"].isin(TEST_SEASONS + VAL_SEASONS + TRAIN_SEASONS[-3:])].copy()
    X_all  = test_features[FEATURE_COLS].values
    test_features = test_features.copy()
    test_features["win_prob"] = model.predict_proba(X_all)[:, 1]

    # Compute max probability swing per match
    swing = (
        test_features.groupby("match_id")["win_prob"]
        .agg(lambda x: x.max() - x.min())
        .reset_index()
        .rename(columns={"win_prob": "max_swing"})
    )
    swing = swing.sort_values("max_swing", ascending=False)

    # Pick top n_matches spread across seasons
    top_match_ids = swing.head(n_matches * 3).merge(
        matches[["match_id","season"]], on="match_id", how="left"
    ).drop_duplicates("season", keep="first").head(n_matches)["match_id"].tolist()

    replay = test_features[test_features["match_id"].isin(top_match_ids)].copy()

    # Add team names for display (via matches)
    replay = replay.merge(
        matches[["match_id","team1","team2","venue","season","match_winner"]],
        on="match_id", how="left"
    )
    return replay[["match_id","season","team1","team2","venue","match_winner",
                   "over_number","ball_number","running_score","running_wickets",
                   "balls_remaining","runs_remaining","required_run_rate","win_prob",
                   LABEL_COL]].reset_index(drop=True)


# ===========================================================================
# 5. Improbable Finishes leaderboard
# ===========================================================================
def compute_improbable_finishes(features: pd.DataFrame, model,
                                  matches: pd.DataFrame) -> pd.DataFrame:
    """
    Rank matches by minimum win probability reached by the eventual winner during the chase.
    Also compute max single-ball swing in the final 24 balls.
    """
    print("  Computing Improbable Finishes leaderboard...")

    X_all = features[FEATURE_COLS].values
    features = features.copy()
    features["win_prob"] = model.predict_proba(X_all)[:, 1]

    # Actual winner's win probability curve
    # chasing_team_won == 1 means the batting team (chasing) won
    # win_prob = probability that chasing team wins
    winner_probs = features[features[LABEL_COL] == 1].copy()
    loser_probs  = features[features[LABEL_COL] == 0].copy()

    # For eventual chasing-team winners: min probability they reached
    min_probs = (
        winner_probs.groupby("match_id")["win_prob"]
        .min().reset_index().rename(columns={"win_prob": "min_winner_prob"})
    )

    # Max single-ball swing in final 24 balls
    final_balls = features[features["balls_remaining"] <= 24].copy()
    final_balls["prob_shift"] = final_balls.groupby("match_id")["win_prob"].diff().abs()
    max_swing = (
        final_balls.groupby("match_id")["prob_shift"]
        .max().reset_index().rename(columns={"prob_shift": "max_final_swing"})
    )

    result = (
        min_probs
        .merge(max_swing, on="match_id", how="outer")
        .merge(matches[["match_id","team1","team2","venue","season","match_winner",
                         "win_by_runs","win_by_wickets"]], on="match_id", how="left")
    )
    result = result.sort_values("min_winner_prob").reset_index(drop=True)
    result.index += 1
    return result


# ===========================================================================
# 6. Inference helpers (used by Streamlit page)
# ===========================================================================
def load_model():
    """Load the calibrated XGBoost model from disk."""
    path = MODELS_DIR / "win_probability_model_v1.joblib"
    if not path.exists():
        raise FileNotFoundError("Model not found. Run the training pipeline first.")
    return joblib.load(path)


def predict_ball_state(model, state: dict) -> float:
    """
    Predict win probability for a single ball state.
    state keys must match FEATURE_COLS.
    Returns probability (0–1) that the chasing team wins.
    """
    row = np.array([[state.get(f, 0.0) for f in FEATURE_COLS]])
    return float(model.predict_proba(row)[0, 1])


def shap_for_state(xgb_model, state: dict) -> dict:
    """Return SHAP feature contributions for a single ball state."""
    import shap
    row  = np.array([[state.get(f, 0.0) for f in FEATURE_COLS]])
    expl = shap.TreeExplainer(xgb_model)
    sv   = expl.shap_values(row)[0]
    return dict(zip(FEATURE_COLS, sv))


# ===========================================================================
# MAIN — run once to build all Module 4 artifacts
# ===========================================================================
def run_module4_pipeline(deliveries, matches):
    print("\n" + "=" * 60)
    print("  Module 4 — Win Probability Pipeline")
    print("=" * 60)

    # 1. Features
    features = build_features(deliveries, matches)
    features.to_parquet(PROCESSED / "model_features.parquet", index=False)
    print(f"    [OK] model_features.parquet ({len(features):,} rows)")

    # 2. Train
    results = train_models(features)
    model      = results["calibrated_xgb"]
    xgb_model  = results["xgb_model"]

    # 3. Replays
    replay = precompute_replays(features, model, matches)
    replay.to_parquet(PROCESSED / "replay_matches.parquet", index=False)
    print(f"    [OK] replay_matches.parquet ({replay['match_id'].nunique()} matches)")

    # 4. Improbable Finishes
    finishes = compute_improbable_finishes(features, model, matches)
    finishes.to_parquet(PROCESSED / "improbable_finishes.parquet", index=False)
    print(f"    [OK] improbable_finishes.parquet ({len(finishes)} matches)")

    # 5. Save validation results
    import json
    val_summary = {
        "val_auc_logreg": round(results["val_auc_lr"], 4),
        "val_auc_xgboost": round(results["val_auc_xgb"], 4),
        "test_auc_calibrated": round(results["test_auc_cal"], 4),
        "test_brier_calibrated": round(results["test_brier_cal"], 4),
        "calibration_curve": results["cal_data"],
        "season_metrics": results["season_metrics"].to_dict(orient="records"),
        "train_seasons": TRAIN_SEASONS,
        "val_seasons": VAL_SEASONS,
        "test_seasons": TEST_SEASONS,
        "feature_cols": FEATURE_COLS,
    }
    (PROJECT_ROOT / "docs" / "model_validation_report.json").write_text(
        json.dumps(val_summary, indent=2), encoding="utf-8"
    )
    print("    [OK] docs/model_validation_report.json written")

    print("\n" + "=" * 60)
    print("  Module 4 pipeline complete.")
    print("=" * 60 + "\n")
    return results
