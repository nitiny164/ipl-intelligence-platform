"""
Module 2 — Team War Room
Business question: For any team, what is their competitive identity?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data_loader import load_matches, load_deliveries, load_teams, processed_exists
from app.style import inject_styles, page_header, CHART_COLORS
from src.module_2_teams import (
    head_to_head_matrix, home_away_record, team_phase_profile,
    rolling_form, season_standings, toss_recommender,
    team_season_record, collapse_propensity, id_to_name,
)

st.set_page_config(page_title="Team War Room | IPL Intelligence", layout="wide")
inject_styles()

if not processed_exists():
    st.error("Run `python -m src.module_0_foundation` first.")
    st.stop()

m  = load_matches()
d  = load_deliveries()
t  = load_teams()

id2name = id_to_name(t)
name2id = {v: k for k, v in id2name.items()}
active_team_ids = set(m[m["is_completed"]]["team1"].dropna().astype(int).tolist() +
                      m[m["is_completed"]]["team2"].dropna().astype(int).tolist())
team_names_sorted = sorted([id2name[tid] for tid in active_team_ids if tid in id2name])

page_header("shield", "Team War Room",
            "Competitive identity: where does each team dominate, where do they leak?")

_HINT = "color:#9E9E9E;font-size:0.72rem;font-style:italic;margin:2px 0 12px"

# ── 1. Head-to-Head Matrix ──────────────────────────────────────────────────
st.header("1. Head-to-Head Win Matrix")
st.markdown(
    '<p style="' + _HINT + '">How to read: Each cell = how often the team on the left (row) beat the team '
    'at the top (column). Green = strong record, Red = weaker record. 65% means the row team won 65 out '
    'of 100 matches against that column team. The diagonal is always empty (team can\'t play itself). '
    'Use the season slider to zoom into a specific era.</p>',
    unsafe_allow_html=True,
)

season_range = st.slider("Season range", 2008, 2026, (2008, 2026), key="h2h_seasons")
st.markdown(
    '<p style="' + _HINT + '">Slider: drag the left handle to start from a later season (e.g. 2016 for '
    'modern-era only). Drag the right handle to exclude recent seasons.</p>',
    unsafe_allow_html=True,
)
pivot_pct, pivot_matches = head_to_head_matrix(m, t, season_range[0], season_range[1])

shared_teams = [n for n in team_names_sorted if n in pivot_pct.index and n in pivot_pct.columns]
pivot_disp = pivot_pct.loc[shared_teams, shared_teams]

fig_h2h = go.Figure(data=go.Heatmap(
    z=pivot_disp.values,
    x=list(pivot_disp.columns),
    y=list(pivot_disp.index),
    colorscale="RdYlGn",
    zmid=50,
    zmin=0, zmax=100,
    text=pivot_disp.values.round(0),
    texttemplate="%{text}%",
    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Win%: %{z:.1f}%<extra></extra>",
))
fig_h2h.update_layout(height=600, title="Win % (row team vs column team — higher = row team dominates)",
                       xaxis=dict(tickangle=-45), template="plotly_white")
st.plotly_chart(fig_h2h, use_container_width=True)

st.markdown("---")

# ── 2. Team deep-dive selector ───────────────────────────────────────────────
st.header("2. Team Deep Dive")
selected_team_name = st.selectbox(
    "Select Team", team_names_sorted,
    index=team_names_sorted.index("Mumbai Indians") if "Mumbai Indians" in team_names_sorted else 0
)
selected_team_id = name2id[selected_team_name]

col_season_l, col_season_r = st.columns(2)
with col_season_l:
    season_min_dd = st.selectbox("From season", sorted(m["season"].unique()), index=0, key="dd_min")
with col_season_r:
    season_max_dd = st.selectbox("To season", sorted(m["season"].unique()), index=len(m["season"].unique())-1, key="dd_max")

st.markdown(
    '<p style="' + _HINT + '">Season range above applies to all charts below — Rolling Form, Phase Strength, and Collapse.</p>',
    unsafe_allow_html=True,
)

# ── Rolling form ─────────────────────────────────────────────────────────────
st.subheader(f"Rolling Form — {selected_team_name}")
st.markdown(
    '<p style="' + _HINT + '">How to read: This chart shows how the team\'s form evolved match by match '
    'across their full history. The Y-axis = win percentage over the last N matches ending on that date '
    '(not the season). A rising line = the team was on a winning streak. A dip = they lost several in a row. '
    'Hover to see the exact date and opponent. The dotted grey line at 50% is the break-even mark — '
    'above it means the team won more than they lost in that rolling window.</p>',
    unsafe_allow_html=True,
)

form_window = st.slider("Rolling window (number of matches)", 5, 20, 10)
st.markdown(
    '<p style="' + _HINT + '">Slider: choose how many past matches to average. '
    'Window = 5 = very reactive (short streaks visible). Window = 15 = smoother, shows longer trends. '
    'Tip: set to 5 to see hot/cold streaks; set to 15 to see season-long momentum.</p>',
    unsafe_allow_html=True,
)

form_df = rolling_form(m, t, selected_team_id, window=form_window,
                       season_min=season_min_dd, season_max=season_max_dd)
fig_form = go.Figure()
fig_form.add_trace(go.Scatter(
    x=form_df["match_date"], y=form_df["rolling_win_pct"],
    mode="lines", fill="tozeroy", fillcolor="rgba(21,101,192,0.12)",
    line=dict(color="#1565C0", width=2),
    hovertemplate=f"%{{x|%d %b %Y}}<br>Win % (last {form_window} matches): %{{y:.1f}}%<br>vs %{{customdata}}<extra></extra>",
    customdata=form_df["opponent"],
))
fig_form.add_hline(y=50, line_dash="dot", line_color="#9E9E9E",
                   annotation_text="50% (break-even)", annotation_font_color="#9E9E9E")
fig_form.update_layout(
    xaxis_title="Match Date",
    yaxis_title=f"Rolling {form_window}-Match Win %",
    height=340, template="plotly_white",
    yaxis=dict(range=[0, 100]),
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig_form, use_container_width=True)

st.markdown("---")

# ── Phase-wise strength bar chart ─────────────────────────────────────────────
st.subheader(f"Phase-Wise Strength — {selected_team_name}")
st.markdown(
    '<p style="' + _HINT + '">How to read: Each group of bars = one match phase (Powerplay / Middle / Death). '
    'Within each phase, 3 metrics are shown: Run Rate, Boundary %, and Dot Ball %. '
    '<b style="color:#1565C0">Blue bars = Batting</b> · <b style="color:#C62828">Red bars = Bowling</b>. '
    'The score is normalised — <b>50 = exactly league average</b>. '
    'Above 50 = better than league average in that metric. Below 50 = weaker. '
    'For Dot Ball %: higher bowling score = more dot balls conceded by opposition (good for the team). '
    'Lower batting score = team faces more dot balls (bad for batting).</p>',
    unsafe_allow_html=True,
)

phase = team_phase_profile(d, t, selected_team_id, season_min_dd, season_max_dd)
phase_order = ["powerplay", "middle", "death"]

bat_df  = phase["batting"].set_index("over_phase")
bowl_df = phase["bowling"].set_index("over_phase")

_lg_raw = d[(~d["is_super_over"]) & d["is_legal_ball"] & (d["season_id"].between(season_min_dd, season_max_dd))].copy()
_lg_raw["is_boundary"] = _lg_raw["batter_runs"].isin([4, 6])
_lg_raw["is_dot"]      = _lg_raw["batter_runs"] == 0
league_bat = (
    _lg_raw.groupby("over_phase")
    .agg(runs=("batter_runs","sum"), balls=("batter_runs","count"),
         boundaries=("is_boundary","sum"), dots=("is_dot","sum"))
    .assign(
        run_rate     = lambda x: x.runs / x.balls * 6,
        boundary_pct = lambda x: x.boundaries / x.balls * 100,
        dot_pct      = lambda x: x.dots / x.balls * 100,
    )
)

def norm_score(value, league_avg, higher_is_better=True):
    if league_avg == 0:
        return 50
    ratio = value / league_avg
    score = ratio * 50 if higher_is_better else (2 - ratio) * 50
    return max(0, min(100, score))

_PHASE_FULL  = {"powerplay": "Powerplay (Overs 1–6)", "middle": "Middle (Overs 7–15)", "death": "Death (Overs 16–20)"}
_METRIC_FULL = {"run_rate": "Run Rate", "boundary_pct": "Boundary %", "dot_pct": "Dot Ball %"}
_METRIC_HELP = {
    "run_rate":     "Runs per over — higher is better for batting",
    "boundary_pct": "% of balls hit to boundary — higher is better for batting",
    "dot_pct":      "% of balls with no run — lower is better for batting, higher is better for bowling",
}

rows = []
for ph in phase_order:
    for metric, higher_bat in [("run_rate", True), ("boundary_pct", True), ("dot_pct", False)]:
        bat_val  = bat_df.loc[ph, metric]  if ph in bat_df.index  else np.nan
        bowl_val = bowl_df.loc[ph, metric] if ph in bowl_df.index else np.nan
        lg_val   = league_bat.loc[ph, metric] if ph in league_bat.index else 1
        bat_score  = round(norm_score(bat_val,  lg_val,  higher_bat),  1) if pd.notna(bat_val)  else 50.0
        bowl_score = round(norm_score(bowl_val, lg_val, not higher_bat), 1) if pd.notna(bowl_val) else 50.0
        suffix = "%" if "pct" in metric else " rpo"
        rows.append({
            "Phase":         _PHASE_FULL[ph],
            "Metric":        _METRIC_FULL[metric],
            "Batting Score": bat_score,
            "Bowling Score": bowl_score,
            "bat_raw":       round(float(bat_val),  2) if pd.notna(bat_val)  else None,
            "bowl_raw":      round(float(bowl_val), 2) if pd.notna(bowl_val) else None,
            "league_avg":    round(float(lg_val),   2) if pd.notna(lg_val)   else None,
            "suffix":        suffix,
            "Help":          _METRIC_HELP[metric],
        })

score_df = pd.DataFrame(rows)

# Build grouped bar chart — one subplot column per phase
from plotly.subplots import make_subplots

phase_labels = [_PHASE_FULL[p] for p in phase_order]
_PHASE_SUBPLOT = {
    "powerplay": "Powerplay  (Overs 1–6)",
    "middle":    "Middle  (Overs 7–15)",
    "death":     "Death  (Overs 16–20)",
}
fig_phase = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"<b>{_PHASE_SUBPLOT[p]}</b>" for p in phase_order],
    shared_yaxes=True,
    horizontal_spacing=0.06,
)

metrics_in_order = ["Run Rate", "Boundary %", "Dot Ball %"]
bat_color  = "#1565C0"
bowl_color = "#C62828"
show_legend = True

for col_idx, ph in enumerate(phase_order, start=1):
    ph_data = score_df[score_df["Phase"] == _PHASE_FULL[ph]]

    def _get(metric, col):
        row = ph_data[ph_data["Metric"] == metric]
        return row[col].iloc[0] if not row.empty else None

    bat_vals   = [_get(m, "Batting Score") or 50 for m in metrics_in_order]
    bowl_vals  = [_get(m, "Bowling Score") or 50 for m in metrics_in_order]
    bat_raws   = [_get(m, "bat_raw") for m in metrics_in_order]
    bowl_raws  = [_get(m, "bowl_raw") for m in metrics_in_order]
    lg_avgs    = [_get(m, "league_avg") for m in metrics_in_order]
    suffixes   = [_get(m, "suffix") or "" for m in metrics_in_order]

    bat_hover  = [
        f"<b>{m}</b><br>Score: {s:.0f} / 100<br>Team: {r}{sx}<br>League avg: {lg}{sx}"
        for m, s, r, lg, sx in zip(metrics_in_order, bat_vals, bat_raws, lg_avgs, suffixes)
    ]
    bowl_hover = [
        f"<b>{m}</b><br>Score: {s:.0f} / 100<br>Team: {r}{sx}<br>League avg: {lg}{sx}"
        for m, s, r, lg, sx in zip(metrics_in_order, bowl_vals, bowl_raws, lg_avgs, suffixes)
    ]

    fig_phase.add_trace(go.Bar(
        name="Batting", x=metrics_in_order, y=bat_vals,
        marker_color=bat_color, showlegend=show_legend,
        text=[f"{v:.0f}" for v in bat_vals],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=12, family="Arial Black"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=bat_hover,
        width=0.38,
    ), row=1, col=col_idx)

    fig_phase.add_trace(go.Bar(
        name="Bowling", x=metrics_in_order, y=bowl_vals,
        marker_color=bowl_color, showlegend=show_legend,
        text=[f"{v:.0f}" for v in bowl_vals],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=12, family="Arial Black"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=bowl_hover,
        width=0.38,
    ), row=1, col=col_idx)

    show_legend = False  # only first subplot shows legend

    # Add average reference line per panel
    fig_phase.add_hline(y=50, line_dash="dot", line_color="#9E9E9E", line_width=1.5,
                        annotation_text="avg" if col_idx == 1 else "",
                        annotation_font_color="#9E9E9E", annotation_font_size=10,
                        row=1, col=col_idx)

fig_phase.update_layout(
    barmode="group",
    height=460,
    template="plotly_white",
    yaxis=dict(
        range=[0, 110], title="Score  (50 = league average)",
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["0", "25", "50 ← avg", "75", "100"],
    ),
    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                font=dict(size=13)),
    margin=dict(t=60, b=80, l=10, r=10),
)
# Consistent y-axis range on panels 2 & 3
for i in range(2, 4):
    fig_phase.update_yaxes(range=[0, 110], row=1, col=i)
# Clean subplot title font
fig_phase.update_annotations(font=dict(size=13))

st.plotly_chart(fig_phase, use_container_width=True)

# Show actual raw numbers in an expander
with st.expander("View actual phase numbers (raw values vs league average)"):
    raw_rows = []
    for ph in phase_order:
        b  = bat_df.loc[ph]   if ph in bat_df.index   else None
        w  = bowl_df.loc[ph]  if ph in bowl_df.index  else None
        lg = league_bat.loc[ph] if ph in league_bat.index else None
        raw_rows.append({
            "Phase":            ph.capitalize(),
            "Bat RR":           round(b["run_rate"],     2) if b  is not None else "-",
            "Lg RR":            round(lg["run_rate"],    2) if lg is not None else "-",
            "Bat Bdry%":        round(b["boundary_pct"], 1) if b  is not None else "-",
            "Lg Bdry%":         round(lg["boundary_pct"],1) if lg is not None else "-",
            "Bat Dot%":         round(b["dot_pct"],      1) if b  is not None else "-",
            "Lg Dot%":          round(lg["dot_pct"],     1) if lg is not None else "-",
            "Bowl RR":          round(w["run_rate"],     2) if w  is not None else "-",
            "Bowl Bdry%":       round(w["boundary_pct"], 1) if w  is not None else "-",
            "Bowl Dot%":        round(w["dot_pct"],      1) if w  is not None else "-",
        })
    st.markdown('<p style="' + _HINT + '">Lg = League average across all teams in the selected season range. '
                'Score = how the team compares: 50 = exactly at league average, above 50 = better, below = weaker.</p>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ── 3. Home vs Away ──────────────────────────────────────────────────────────
st.header("3. Home vs Away Win Rates — All Teams")
st.markdown(
    '<p style="' + _HINT + '">How to read: Each bubble = one team. X-axis = away win %, Y-axis = home win %. '
    'Teams above the diagonal dotted line = perform better at home than away (home advantage). '
    'Teams below = equally good or better away. Bubble size = number of home matches played. '
    'Colour = home win % (green = strong home record).</p>',
    unsafe_allow_html=True,
)

ha = home_away_record(m, t)
ha_team = ha[ha["team"] == selected_team_name]
if not ha_team.empty:
    r = ha_team.iloc[0]
    adv = r["home_win_pct"] - r["away_win_pct"]
    adv_color  = "#2E7D32" if adv > 0 else ("#C62828" if adv < 0 else "#546E7A")
    adv_icon   = "Home dominant" if adv > 2 else ("Away dominant" if adv < -2 else "Balanced")
    home_color = "#1565C0"
    away_color = "#C62828"

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:16px">
      <div style="background:#EEF4FF;border:1px solid #BBDEFB;border-left:5px solid {home_color};
                  border-radius:10px;padding:16px 20px">
        <p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.07em;color:{home_color}">Home Win %</p>
        <p style="margin:0;font-size:2rem;font-weight:800;color:#1A1A2E">{r['home_win_pct']:.1f}%</p>
        <p style="margin:4px 0 0;font-size:0.75rem;color:#546E7A">{int(r['home_matches'])} home matches played</p>
      </div>
      <div style="background:#FFEBEE;border:1px solid #FFCDD2;border-left:5px solid {away_color};
                  border-radius:10px;padding:16px 20px">
        <p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.07em;color:{away_color}">Away Win %</p>
        <p style="margin:0;font-size:2rem;font-weight:800;color:#1A1A2E">{r['away_win_pct']:.1f}%</p>
        <p style="margin:4px 0 0;font-size:0.75rem;color:#546E7A">{int(r['away_matches'])} away matches played</p>
      </div>
      <div style="background:#FFFFFF;border:1px solid #E3E8EF;border-left:5px solid {adv_color};
                  border-radius:10px;padding:16px 20px">
        <p style="margin:0 0 4px;font-size:0.7rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.07em;color:{adv_color}">Home Advantage</p>
        <p style="margin:0;font-size:2rem;font-weight:800;color:{adv_color}">{adv:+.1f}%</p>
        <p style="margin:4px 0 0;font-size:0.75rem;color:#546E7A">{adv_icon}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Zoom axes to data range with 8% padding
_x_min = max(0,  ha["away_win_pct"].min() - 8)
_x_max = min(100, ha["away_win_pct"].max() + 8)
_y_min = max(0,  ha["home_win_pct"].min() - 8)
_y_max = min(100, ha["home_win_pct"].max() + 8)
_pad   = max(_x_max - _x_min, _y_max - _y_min)  # square viewport

fig_ha = go.Figure()

# Diagonal "no advantage" line clipped to data range
_diag = max(_x_min, _y_min), min(_x_max, _y_max)
fig_ha.add_shape(type="line",
    x0=_diag[0], y0=_diag[0], x1=_diag[1], y1=_diag[1],
    line=dict(dash="dot", color="#BDBDBD", width=1.5))
fig_ha.add_annotation(
    x=(_diag[0]+_diag[1])/2 + 3, y=(_diag[0]+_diag[1])/2 - 3,
    text="No home advantage", showarrow=False,
    font=dict(size=10, color="#BDBDBD"), textangle=-45,
)

# Shade "home advantage" region lightly
fig_ha.add_shape(type="rect",
    x0=_x_min, y0=(_x_min+_y_min)/2, x1=_x_max, y1=_y_max,
    fillcolor="rgba(21,101,192,0.04)", line_width=0, layer="below")

# Highlight selected team differently
ha["is_selected"] = ha["team"] == selected_team_name

for _, row in ha.iterrows():
    is_sel = row["team"] == selected_team_name
    fig_ha.add_trace(go.Scatter(
        x=[row["away_win_pct"]], y=[row["home_win_pct"]],
        mode="markers+text",
        text=[row["team"]],
        textposition="top center",
        textfont=dict(
            size=12 if is_sel else 10,
            color="#1565C0" if is_sel else "#37474F",
            family="Arial Black" if is_sel else "Arial",
        ),
        marker=dict(
            size=max(12, min(32, row["home_matches"] * 0.55)),
            color=row["home_win_pct"],
            colorscale="RdYlGn",
            cmin=30, cmax=75,
            line=dict(width=2.5 if is_sel else 1,
                      color="#1565C0" if is_sel else "white"),
            opacity=1.0 if is_sel else 0.80,
        ),
        hovertemplate=(
            f"<b>{row['team']}</b><br>"
            f"Home win %: {row['home_win_pct']:.1f}% ({int(row['home_matches'])} matches)<br>"
            f"Away win %: {row['away_win_pct']:.1f}% ({int(row['away_matches'])} matches)<br>"
            f"Home advantage: {row['home_win_pct']-row['away_win_pct']:+.1f}%<extra></extra>"
        ),
        showlegend=False,
    ))

fig_ha.update_layout(
    xaxis=dict(title="Away Win %", range=[_x_min, _x_max],
               ticksuffix="%", gridcolor="#F0F0F0"),
    yaxis=dict(title="Home Win %", range=[_y_min, _y_max],
               ticksuffix="%", gridcolor="#F0F0F0"),
    height=520, template="plotly_white",
    margin=dict(t=20, b=50, l=60, r=20),
    plot_bgcolor="#FAFBFD",
)
st.plotly_chart(fig_ha, use_container_width=True)

st.markdown("---")

# ── 4. Season Standings ───────────────────────────────────────────────────────
st.header("4. Season Points Table Reconstruction")
st.markdown(
    '<p style="' + _HINT + '">How to read: Reconstructed points table for the selected season based on '
    'match outcomes (2 points per win). NRR = Net Run Rate, used as tie-breaker. '
    'Top 4 teams in the league stage qualify for playoffs.</p>',
    unsafe_allow_html=True,
)

selected_season = st.selectbox("Select Season", sorted(m["season"].unique(), reverse=True), key="standings_season")
standings = season_standings(m, t, selected_season)

# ── Determine playoff positions for this season ───────────────────────────────
def get_playoff_positions(matches_df, teams_df, season):
    """
    Returns dict: team_name -> finish label
    IPL playoff format:
      Qualifier 1 (Q1):  #1 vs #2 league → winner → Final; loser → Q2
      Eliminator  (E):   #3 vs #4 league → winner → Q2;   loser = 4th
      Qualifier 2 (Q2):  Q1-loser vs E-winner → winner → Final; loser = 3rd
      Final:             Q1-winner vs Q2-winner → Champion
    We detect playoff matches as the last 4 completed matches of the season.
    """
    _id2name = dict(zip(teams_df["team_id"], teams_df["team_name"]))

    season_m = matches_df[
        (matches_df["season"] == season) & (matches_df["is_completed"])
    ].sort_values("match_date")

    if len(season_m) < 16:   # too short to have playoffs
        return {}

    playoff_matches = season_m.tail(4).reset_index(drop=True)
    if len(playoff_matches) < 4:
        return {}

    def teams_of(row):
        return _id2name.get(row["team1"]), _id2name.get(row["team2"])
    def winner_of(row):
        return _id2name.get(row["match_winner"])
    def loser_of(row):
        t1, t2 = teams_of(row)
        w = winner_of(row)
        return t2 if w == t1 else t1

    q1_row  = playoff_matches.iloc[0]   # earliest playoff
    e_row   = playoff_matches.iloc[1]   # eliminator
    q2_row  = playoff_matches.iloc[2]   # qualifier 2
    fin_row = playoff_matches.iloc[3]   # final

    return {
        winner_of(fin_row): "Champion",
        loser_of(fin_row):  "Runner-up",
        loser_of(q2_row):   "3rd Place",
        loser_of(e_row):    "4th Place",
    }

playoff_pos = get_playoff_positions(m, t, selected_season)

# ── Build display DataFrame ───────────────────────────────────────────────────
standings_disp = standings.reset_index(drop=True).copy()
standings_disp.insert(0, "Rank", range(1, len(standings_disp) + 1))
standings_disp = standings_disp.rename(columns={
    "played": "Played", "won": "Won", "lost": "Lost",
    "tied": "Tied", "nr": "NR", "points": "Points", "nrr": "NRR",
})

# Add finish column
_FINISH_BADGE = {
    "Champion":   "Champion",
    "Runner-up":  "Runner-up",
    "3rd Place":  "3rd Place",
    "4th Place":  "4th Place",
}
standings_disp["Playoff Finish"] = standings_disp["Team"].map(
    lambda name: playoff_pos.get(name, "Did not qualify" if standings_disp[standings_disp["Team"]==name].index[0] >= 4 else "Qualified")
)

max_pts = int(standings_disp["Points"].max()) if "Points" in standings_disp.columns else 28

st.dataframe(
    standings_disp,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank":           st.column_config.NumberColumn("Rank",      format="%d",   width="small"),
        "Team":           st.column_config.TextColumn("Team",                        width="medium"),
        "Played":         st.column_config.NumberColumn("Played",    format="%d",   width="small"),
        "Won":            st.column_config.NumberColumn("Won",       format="%d",   width="small"),
        "Lost":           st.column_config.NumberColumn("Lost",      format="%d",   width="small"),
        "Tied":           st.column_config.NumberColumn("Tied",      format="%d",   width="small"),
        "NR":             st.column_config.NumberColumn("No Result", format="%d",   width="small"),
        "Points":         st.column_config.ProgressColumn(
                              "Points", min_value=0, max_value=max_pts,
                              format="%d pts", width="medium"),
        "NRR":            st.column_config.NumberColumn("NRR",       format="%.3f", width="small"),
        "Playoff Finish": st.column_config.TextColumn("Playoff Finish",              width="medium"),
    },
)

# Show playoff bracket below table if playoffs happened
if playoff_pos:
    champion  = next((t for t,f in playoff_pos.items() if f=="Champion"),  "")
    runner_up = next((t for t,f in playoff_pos.items() if f=="Runner-up"), "")
    third     = next((t for t,f in playoff_pos.items() if f=="3rd Place"), "")
    fourth    = next((t for t,f in playoff_pos.items() if f=="4th Place"), "")

    st.markdown(f"""
    <div style="background:#F8F9FC;border:1px solid #E3E8EF;border-radius:10px;padding:16px 20px;margin-top:10px">
      <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;
                  color:#546E7A;margin-bottom:12px">Playoff Bracket — {selected_season}</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;text-align:center">
        <div style="background:#fff;border:1px solid #E3E8EF;border-radius:8px;padding:10px">
          <div style="font-size:0.65rem;color:#9E9E9E;margin-bottom:4px">CHAMPION</div>
          <div style="font-size:0.85rem;font-weight:700;color:#F57F17">{champion}</div>
          <div style="font-size:0.7rem;color:#9E9E9E;margin-top:2px">Won Final</div>
        </div>
        <div style="background:#fff;border:1px solid #E3E8EF;border-radius:8px;padding:10px">
          <div style="font-size:0.65rem;color:#9E9E9E;margin-bottom:4px">RUNNER-UP</div>
          <div style="font-size:0.85rem;font-weight:700;color:#546E7A">{runner_up}</div>
          <div style="font-size:0.7rem;color:#9E9E9E;margin-top:2px">Lost Final</div>
        </div>
        <div style="background:#fff;border:1px solid #E3E8EF;border-radius:8px;padding:10px">
          <div style="font-size:0.65rem;color:#9E9E9E;margin-bottom:4px">3RD PLACE</div>
          <div style="font-size:0.85rem;font-weight:700;color:#546E7A">{third}</div>
          <div style="font-size:0.7rem;color:#9E9E9E;margin-top:2px">Lost Qualifier 2</div>
        </div>
        <div style="background:#fff;border:1px solid #E3E8EF;border-radius:8px;padding:10px">
          <div style="font-size:0.65rem;color:#9E9E9E;margin-bottom:4px">4TH PLACE</div>
          <div style="font-size:0.85rem;font-weight:700;color:#546E7A">{fourth}</div>
          <div style="font-size:0.7rem;color:#9E9E9E;margin-top:2px">Lost Eliminator</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<p style="' + _HINT + '">Top 4 teams qualify for playoffs. '
    'NRR (Net Run Rate) is used as tie-breaker when two teams have equal points. '
    'NR = No Result matches (rain/DLS). '
    'Playoff Finish column shows final tournament result for qualifying teams.</p>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ── 5. Collapse Propensity ───────────────────────────────────────────────────
st.header("5. Batting Collapse Propensity")
st.markdown(
    '<p style="' + _HINT + '">A <b>collapse</b> = losing 3 or more wickets in any consecutive 12 balls '
    'within the same innings. The bar shows what <b>% of innings</b> contained at least one such collapse. '
    'Example: Gujarat Titans at ~29% means roughly 1 in every 3.5 innings they bat, '
    'their middle order caves in within 2 overs. Longer bar = more fragile batting lineup.</p>',
    unsafe_allow_html=True,
)

collapse_df = collapse_propensity(d, t, season_min_dd, season_max_dd)
collapse_df = collapse_df[collapse_df["team_name"].notna()].copy()
collapse_df["collapses"]     = collapse_df["collapses"].astype(int)
collapse_df["innings_count"] = collapse_df["innings_count"].astype(int)
collapse_df["collapse_pct"]  = collapse_df["collapse_per_100_innings"]

plot_df = collapse_df.head(16).sort_values("collapse_pct").reset_index(drop=True)

fig_col = go.Figure(go.Bar(
    y=plot_df["team_name"],
    x=plot_df["collapse_pct"],
    orientation="h",
    marker=dict(
        color=plot_df["collapse_pct"],
        colorscale="OrRd",
        showscale=False,
    ),
    text=[f"{p:.1f}%  ({int(c)} of {int(i)} innings)"
          for p, c, i in zip(plot_df["collapse_pct"],
                             plot_df["collapses"],
                             plot_df["innings_count"])],
    textposition="outside",
    textfont=dict(size=11, color="#1A1A2E"),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Collapse in <b>%{customdata[0]}</b> out of <b>%{customdata[1]}</b> innings batted<br>"
        "= <b>%{x:.1f}%</b> of innings<br>"
        "<i>Collapse = 3+ wickets in any 12 consecutive balls</i>"
        "<extra></extra>"
    ),
    customdata=list(zip(plot_df["collapses"], plot_df["innings_count"])),
))

fig_col.update_layout(
    title="Batting Collapse Rate — actual innings, not projections",
    height=460, template="plotly_white",
    xaxis=dict(title="% of Innings with a Collapse", ticksuffix="%", range=[0, plot_df["collapse_pct"].max() * 1.22]),
    yaxis=dict(title=""),
    margin=dict(l=10, r=20),
)
st.plotly_chart(fig_col, use_container_width=True)

# Plain-English callout
if not collapse_df.empty:
    best_row  = collapse_df.loc[collapse_df["collapse_pct"].idxmin()]
    worst_row = collapse_df.loc[collapse_df["collapse_pct"].idxmax()]
    st.markdown(
        f'<div style="background:#F3F4F6;border-left:4px solid #546E7A;border-radius:8px;'
        f'padding:12px 16px;font-size:0.82rem;color:#37474F;margin-bottom:8px">'
        f'<b>How to read:</b> The bar shows what % of a team\'s actual innings had a collapse. '
        f'Example: if a team shows <b>28.6% (40 of 140 innings)</b>, it means in 40 real innings '
        f'out of 140 they batted, 3+ wickets fell within just 2 overs. &nbsp;'
        f'Most stable: <b>{best_row["team_name"]}</b> — collapsed in only {int(best_row["collapses"])} '
        f'of {int(best_row["innings_count"])} innings ({best_row["collapse_pct"]:.1f}%). &nbsp;'
        f'Most fragile: <b>{worst_row["team_name"]}</b> — collapsed in {int(worst_row["collapses"])} '
        f'of {int(worst_row["innings_count"])} innings ({worst_row["collapse_pct"]:.1f}%).'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── 6. Venue Toss Recommender ─────────────────────────────────────────────────
st.header("6. Venue Toss-Decision Recommender")
st.markdown(
    '<p style="' + _HINT + '">How to read: Based on historical results at the selected venue — '
    'what is the smarter toss decision? Green bar = the recommended choice based on past data. '
    'The number on each bar = win rate for teams that chose that option. The dotted line at 50% '
    'is the neutral mark. Confidence level (High/Medium/Low) reflects how many matches are available — '
    'more matches = more reliable recommendation.</p>',
    unsafe_allow_html=True,
)

venues_available = sorted(m[m["is_completed"]]["venue"].dropna().unique().tolist())
selected_venue = st.selectbox("Select Venue", venues_available,
    index=venues_available.index("Wankhede Stadium") if "Wankhede Stadium" in venues_available else 0)

rec = toss_recommender(m, selected_venue)

# ── KPI strip ─────────────────────────────────────────────────────────────────
_conf_color = {"High": "#2E7D32", "Medium": "#E65100", "Low": "#C62828"}.get(rec["confidence"], "#546E7A")
_rec_icon   = "sports_cricket" if rec["recommendation"].startswith("Field") else "sports_baseball"
_rec_label  = "Field First (Chase)" if rec["recommendation"].startswith("Field") else "Bat First (Defend)"

if rec["low_sample"]:
    st.warning(f"Only {rec['sample']} matches at this venue — treat recommendation with caution.", icon="⚠️")

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:12px 0 16px">
  <div style="background:#fff;border:1px solid #E3E8EF;border-radius:10px;padding:14px 18px;border-left:4px solid #1565C0">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#546E7A;margin-bottom:6px">Recommendation</div>
    <div style="font-size:1.1rem;font-weight:700;color:#1A1A2E">{_rec_label}</div>
  </div>
  <div style="background:#fff;border:1px solid #E3E8EF;border-radius:10px;padding:14px 18px">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#546E7A;margin-bottom:6px">Chase Win %</div>
    <div style="font-size:1.5rem;font-weight:700;color:#1A1A2E">{rec['chase_success_pct']:.1f}%</div>
    <div style="font-size:0.7rem;color:#9E9E9E">% of matches won batting 2nd</div>
  </div>
  <div style="background:#fff;border:1px solid #E3E8EF;border-radius:10px;padding:14px 18px">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#546E7A;margin-bottom:6px">Avg 1st Innings Score</div>
    <div style="font-size:1.5rem;font-weight:700;color:#1A1A2E">{rec['avg_first_innings_score']:.0f}</div>
    <div style="font-size:0.7rem;color:#9E9E9E">runs typically set as target</div>
  </div>
  <div style="background:#fff;border:1px solid #E3E8EF;border-radius:10px;padding:14px 18px">
    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#546E7A;margin-bottom:6px">Confidence</div>
    <div style="font-size:1.1rem;font-weight:700;color:{_conf_color}">{rec['confidence']}</div>
    <div style="font-size:0.7rem;color:#9E9E9E">based on {rec['sample']} matches</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Bar chart ──────────────────────────────────────────────────────────────────
chase_pct   = rec["chase_success_pct"]
defend_pct  = rec["defend_success_pct"]
is_chase_better = rec["recommendation"].startswith("Field")

fig_rec = go.Figure()
fig_rec.add_trace(go.Bar(
    x=["Field First\n(Chase)", "Bat First\n(Defend)"],
    y=[chase_pct, defend_pct],
    marker_color=[
        "#2E7D32" if is_chase_better  else "#90CAF9",
        "#2E7D32" if not is_chase_better else "#90CAF9",
    ],
    text=[f"{chase_pct:.1f}%", f"{defend_pct:.1f}%"],
    textposition="inside",
    insidetextanchor="middle",
    textfont=dict(color="white", size=16, family="Arial Black"),
    width=0.4,
    hovertemplate="<b>%{x}</b><br>Win rate: %{y:.1f}%<extra></extra>",
))

# 50% reference line — annotation placed to the RIGHT of chart, not on bars
fig_rec.add_hline(
    y=50, line_dash="dot", line_color="#BDBDBD", line_width=1.5,
)
fig_rec.add_annotation(
    x=1.45, y=50, xref="x", yref="y",
    text="50% neutral", showarrow=False,
    font=dict(size=10, color="#BDBDBD"),
    xanchor="left",
)

fig_rec.update_layout(
    xaxis=dict(tickfont=dict(size=13)),
    yaxis=dict(range=[0, 100], title="Win %", ticksuffix="%", dtick=25),
    height=320, template="plotly_white", showlegend=False,
    margin=dict(t=20, b=20, l=60, r=80),
    plot_bgcolor="#FAFBFD",
)
st.plotly_chart(fig_rec, use_container_width=True)
