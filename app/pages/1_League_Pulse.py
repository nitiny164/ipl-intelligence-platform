"""
Module 1 — League Pulse
Business question: How has the IPL evolved as a product over 18 seasons?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_loader import load_matches, load_deliveries, processed_exists
from app.style import inject_styles, page_header, section_header, CHART_COLORS
from src.module_1_league import (
    season_scoring_trends,
    toss_decision_trend,
    venue_scoring_profile,
    competitiveness_trend,
    phase_scoring_era,
    impact_player_effect,
)

st.set_page_config(page_title="League Pulse | IPL Intelligence", layout="wide")
inject_styles()

if not processed_exists():
    st.error("Run `python -m src.module_0_foundation` first.")
    st.stop()

m = load_matches()
d = load_deliveries()

page_header("trending_up", "League Pulse",
            "How has the IPL evolved as a product — scoring, strategy, and competitiveness — over 18 seasons?")

# ── Precompute ──────────────────────────────────────────────────────────────
scoring   = season_scoring_trends(m, d)
toss      = toss_decision_trend(m)
venue     = venue_scoring_profile(m, min_matches=10)
compete   = competitiveness_trend(m)
phase_rr  = phase_scoring_era(d)
ip_effect = impact_player_effect(d, m)

SEASON_AXIS = dict(tickformat="d", dtick=1, title="Season")

_HINT = "color:#9E9E9E;font-size:0.72rem;font-style:italic;margin:2px 0 12px"

# ── Top KPIs ────────────────────────────────────────────────────────────────
all_seasons = sorted(scoring["season"].astype(int).tolist())
kpi_col_s, kpi_col_e = st.columns(2)
with kpi_col_s:
    season_from = st.selectbox("Compare: Base season", all_seasons,
                               index=0, key="kpi_from")
with kpi_col_e:
    season_to = st.selectbox("Compare: Target season", all_seasons,
                             index=len(all_seasons)-1, key="kpi_to")
st.markdown('<p style="color:#9E9E9E;font-size:0.72rem;font-style:italic;margin:-6px 0 10px">'
            'KPI cards below compare the target season against the base season — change either dropdown to update.</p>',
            unsafe_allow_html=True)

_base   = scoring[scoring["season"].astype(int) == season_from].iloc[0]
_target = scoring[scoring["season"].astype(int) == season_to].iloc[0]

def _kpi_card(label, value, base_value=None, suffix="", invert=False, vs_label=""):
    delta_part = ""
    if base_value is not None:
        diff = value - base_value
        positive = diff > 0 if not invert else diff < 0
        color = "#2E7D32" if positive else ("#C62828" if diff != 0 else "#546E7A")
        arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "—")
        sign  = "+" if diff > 0 else ""
        vs    = vs_label if vs_label else season_from
        delta_part = f'<p style="margin:4px 0 0;font-size:0.78rem;font-weight:600;color:{color}">{arrow} {sign}{diff:.1f}{suffix} vs {vs}</p>'
    return (
        f'<div style="background:#FFFFFF;border:1px solid #E3E8EF;border-radius:10px;'
        f'padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
        f'<p style="margin:0 0 6px;font-size:0.72rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#78909C">{label}</p>'
        f'<p style="margin:0;font-size:1.75rem;font-weight:800;color:#1A1A2E;line-height:1.1">{value:.1f}{suffix}</p>'
        f'{delta_part}'
        f'</div>'
    )

kpi_html = "".join([
    _kpi_card(f"Avg Score ({season_to})",   _target["avg_first_innings_score"], _base["avg_first_innings_score"]),
    _kpi_card(f"Boundary % ({season_to})",  _target["boundary_pct"],            _base["boundary_pct"],            suffix="%"),
    _kpi_card(f"Dot Ball % ({season_to})",  _target["dot_ball_pct"],            _base["dot_ball_pct"],            suffix="%", invert=True),
    _kpi_card(f"Six % ({season_to})",       _target["six_pct"],                 _base["six_pct"],                 suffix="%"),
    _kpi_card(f"Avg Score ({season_from})", _base["avg_first_innings_score"],   _target["avg_first_innings_score"], vs_label=season_to),
])

st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:8px">{kpi_html}</div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ── 1. Season scoring trend ─────────────────────────────────────────────────
st.subheader("1. Average First-Innings Score by Season")
st.markdown(
    '<p style="' + _HINT + '">How to read: Each dot is one IPL season. The blue line shows the trend — '
    'rising means teams scored more that year on average. The orange shaded area marks the Impact Player era '
    '(2023 onwards) when teams can substitute one player mid-match. Hover over any point to see exact values.</p>',
    unsafe_allow_html=True,
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=scoring["season"], y=scoring["avg_first_innings_score"],
    mode="lines+markers", name="Avg Score",
    line=dict(color="#1565C0", width=2.5),
    marker=dict(size=7),
    hovertemplate="<b>Season %{x}</b><br>Avg Score: %{y:.1f}<extra></extra>",
))
fig.add_vrect(
    x0=2022.5, x1=scoring["season"].max() + 0.5,
    fillcolor="rgba(255,152,0,0.10)", line_width=0,
    annotation_text="Impact Player era →",
    annotation_position="top left",
    annotation_font=dict(size=11, color="#E65100"),
)
fig.update_layout(
    xaxis=SEASON_AXIS,
    yaxis_title="Average First-Innings Score",
    height=380, template="plotly_white",
    hovermode="x unified",
    margin=dict(t=30, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# YoY bar chart
scoring_bar = scoring.copy()
scoring_bar["season"] = scoring_bar["season"].astype(int).astype(str)
fig2 = px.bar(
    scoring_bar.dropna(subset=["yoy_score_change_pct"]),
    x="season", y="yoy_score_change_pct",
    color="yoy_score_change_pct",
    color_continuous_scale=["#C62828", "#FFF9C4", "#2E7D32"],
    color_continuous_midpoint=0,
    labels={"yoy_score_change_pct": "YoY Change (%)", "season": "Season"},
    title="Year-on-Year Change in Average First-Innings Score (%)",
)
fig2.update_layout(height=300, template="plotly_white", coloraxis_showscale=False, margin=dict(t=40))
st.plotly_chart(fig2, use_container_width=True)
st.markdown(
    '<p style="' + _HINT + '">How to read: Each bar compares that season\'s average first-innings score with '
    'the previous season. Green bar = teams scored more than the year before. Red bar = scoring dipped vs the '
    'prior season. Example: a bar at 2024 of +3% means teams averaged 3% more runs in 2024 compared to 2023.</p>',
    unsafe_allow_html=True,
)

# ── 2. Boundary / Dot / Six era trends ─────────────────────────────────────
st.subheader("2. Batting Style Evolution — Boundaries, Sixes & Dot Balls")
st.markdown(
    '<p style="' + _HINT + '">How to read: Three lines over seasons. Pink = boundary balls as % of all legal '
    'balls faced. Orange = six % only (subset of boundaries). Grey dashed = dot balls (no run scored). '
    'If boundaries rise while dots fall, batters are under greater pressure to attack. '
    'Lines crossing each other are key moments of style change.</p>',
    unsafe_allow_html=True,
)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=scoring["season"], y=scoring["boundary_pct"],
    mode="lines+markers", name="Boundary %", line=dict(color="#AD1457", width=2), marker=dict(size=6)))
fig3.add_trace(go.Scatter(x=scoring["season"], y=scoring["six_pct"],
    mode="lines+markers", name="Six %", line=dict(color="#E65100", width=2), marker=dict(size=6)))
fig3.add_trace(go.Scatter(x=scoring["season"], y=scoring["dot_ball_pct"],
    mode="lines+markers", name="Dot Ball %", line=dict(color="#546E7A", width=2, dash="dash"), marker=dict(size=6)))
fig3.update_layout(
    xaxis=SEASON_AXIS,
    yaxis_title="% of Legal Balls",
    height=360, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=40),
)
st.plotly_chart(fig3, use_container_width=True)

# ── 3. Phase-wise run rate ──────────────────────────────────────────────────
st.subheader("3. Phase-wise Run Rate — Is Death-Over Scoring Rising Faster?")
st.markdown(
    '<p style="' + _HINT + '">How to read: Three lines — Powerplay (overs 1–6), Middle (overs 7–15), '
    'Death (overs 16–20). Y-axis = average runs scored per over in that phase. '
    'A widening gap between Death and Powerplay lines means batters are increasingly attacking at the end '
    'vs the start. Hover to compare all three phases in any given season.</p>',
    unsafe_allow_html=True,
)

phase_pivot = phase_rr.pivot(index="season", columns="over_phase", values="run_rate").reset_index()
phase_colors = {"powerplay": "#1565C0", "middle": "#2E7D32", "death": "#C62828"}
fig4 = go.Figure()
for phase in ["powerplay", "middle", "death"]:
    if phase in phase_pivot.columns:
        fig4.add_trace(go.Scatter(
            x=phase_pivot["season"], y=phase_pivot[phase],
            mode="lines+markers", name=phase.capitalize(),
            line=dict(color=phase_colors[phase], width=2.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{phase.capitalize()}</b> — Season %{{x}}: %{{y:.2f}} rpo<extra></extra>",
        ))
fig4.update_layout(
    xaxis=SEASON_AXIS,
    yaxis_title="Run Rate (runs per over)",
    height=380, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=40),
)
st.plotly_chart(fig4, use_container_width=True)

# ── 4. Toss decision trend ──────────────────────────────────────────────────
st.subheader("4. Toss Strategy — Has Chase-First Become the Default?")
st.markdown(
    '<p style="' + _HINT + '">How to read: Blue bars (left axis) = what % of captains chose to field '
    'first (i.e. chase) after winning the toss. Red line (right axis) = how often the chasing team '
    'actually won. When bars are tall AND the red line is above 50%, chasing is both preferred and effective. '
    'The dotted grey line at 50% is the break-even point.</p>',
    unsafe_allow_html=True,
)

toss_bar = toss.copy()
toss_bar["season"] = toss_bar["season"].astype(int).astype(str)
fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=toss_bar["season"], y=toss_bar["field_first_pct"],
    name="Chose to Field %", marker_color="rgba(21,101,192,0.5)",
    hovertemplate="<b>Season %{x}</b><br>Field first: %{y:.1f}%<extra></extra>",
))
fig5.add_trace(go.Scatter(
    x=toss_bar["season"], y=toss_bar["field_win_pct"],
    mode="lines+markers", name="Chase Win %", yaxis="y2",
    line=dict(color="#C62828", width=2.5), marker=dict(size=7),
    hovertemplate="<b>Season %{x}</b><br>Chase win %: %{y:.1f}%<extra></extra>",
))
fig5.add_hline(y=50, line_dash="dot", line_color="#90A4AE", annotation_text="50%", yref="y2")
fig5.update_layout(
    xaxis_title="Season",
    yaxis=dict(title="% Chose to Field", range=[0, 100]),
    yaxis2=dict(title="Chase Win %", overlaying="y", side="right", range=[0, 100]),
    height=370, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=40),
)
st.plotly_chart(fig5, use_container_width=True)

# ── 5. Impact Player rule ───────────────────────────────────────────────────
st.subheader("5. Impact Player Rule — Before vs After 2023")
st.markdown(
    '<p style="' + _HINT + '">How to read: Side-by-side comparison of two eras — before the Impact Player '
    'rule (pre-2023) and after (2023 onwards). Left chart: average total runs scored by the batting team '
    'in the first innings. Right chart: average run rate (runs per over). A meaningful jump between the '
    'two bars suggests the rule changed batting behaviour.</p>',
    unsafe_allow_html=True,
)

ip_df = pd.DataFrame(ip_effect)
col1, col2 = st.columns(2)
with col1:
    fig_ip1 = px.bar(ip_df, x="era", y="avg_first_innings_score",
        color="era", text="avg_first_innings_score",
        labels={"avg_first_innings_score": "Avg 1st Innings Score", "era": ""},
        color_discrete_sequence=["#90CAF9", "#1565C0"],
        title="Avg 1st Innings Score")
    fig_ip1.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_ip1.update_layout(height=340, template="plotly_white", showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig_ip1, use_container_width=True)
with col2:
    fig_ip2 = px.bar(ip_df, x="era", y="avg_run_rate",
        color="era", text="avg_run_rate",
        labels={"avg_run_rate": "Avg Run Rate (rpo)", "era": ""},
        color_discrete_sequence=["#FFCC80", "#E65100"],
        title="Avg Run Rate (1st Innings, Legal Balls)")
    fig_ip2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_ip2.update_layout(height=340, template="plotly_white", showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig_ip2, use_container_width=True)

# ── 6. Venue scoring profile ────────────────────────────────────────────────
st.subheader("6. Venue Scoring Profile")
st.markdown(
    '<p style="' + _HINT + '">How to read: Each bar is a venue — length = how many runs the batting '
    'team typically scored in the first innings there. Colour = how often the chasing team won at that '
    'venue (green = chaser-friendly, red = defender-friendly). Venues are sorted from lowest to highest '
    'scoring, making it easy to spot bowler vs batter-friendly grounds.</p>',
    unsafe_allow_html=True,
)

min_m = st.slider("Minimum matches at venue", 5, 30, 10, key="venue_min")
st.markdown(
    '<p style="' + _HINT + '">Slider: drag to filter out venues with fewer matches. '
    'Higher threshold = more reliable data but fewer venues shown.</p>',
    unsafe_allow_html=True,
)
top_venues = venue_scoring_profile(m, min_matches=min_m)
top_venues = top_venues[~top_venues["low_sample"]].head(25)

fig6 = px.bar(
    top_venues.sort_values("avg_first_innings_score"),
    y="venue", x="avg_first_innings_score",
    orientation="h",
    color="chase_success_pct",
    color_continuous_scale="RdYlGn",
    hover_data={"match_count": True, "chase_success_pct": ":.1f"},
    labels={"avg_first_innings_score": "Avg 1st Innings Score", "chase_success_pct": "Chase Win %", "venue": ""},
    title="Avg First-Innings Score by Venue  (colour = Chase Win %)",
)
fig6.update_layout(
    height=max(420, len(top_venues) * 24),
    template="plotly_white",
    margin=dict(t=40, l=10),
    coloraxis_colorbar=dict(title="Chase\nWin %", thickness=14),
)
st.plotly_chart(fig6, use_container_width=True)

# ── 7. Competitiveness trend ────────────────────────────────────────────────
st.subheader("7. Match Competitiveness — Are Matches Getting Closer?")
st.markdown(
    '<p style="' + _HINT + '">How to read: Purple bars (right axis) = % of matches in that season '
    'that were "close" — won by ≤15 runs when batting first, or by ≤2 wickets when chasing. '
    'Purple line (left axis) = a closeness score from 0 to 1, where 1 = last-ball finish and '
    '0 = 10-wicket blowout. A rising line + taller bars = an increasingly competitive league.</p>',
    unsafe_allow_html=True,
)

compete_bar = compete.copy()
compete_bar["season"] = compete_bar["season"].astype(int).astype(str)
fig7 = go.Figure()
fig7.add_trace(go.Bar(
    x=compete_bar["season"], y=compete_bar["close_match_pct"],
    name="Close Match %", marker_color="rgba(103,58,183,0.3)",
    yaxis="y2",
    hovertemplate="<b>Season %{x}</b><br>Close matches: %{y:.1f}%<extra></extra>",
))
fig7.add_trace(go.Scatter(
    x=compete_bar["season"], y=compete_bar["avg_closeness"],
    mode="lines+markers", name="Avg Closeness Score",
    line=dict(color="#4527A0", width=2.5), marker=dict(size=7),
    hovertemplate="<b>Season %{x}</b><br>Closeness: %{y:.3f}<extra></extra>",
))
fig7.update_layout(
    xaxis_title="Season",
    yaxis=dict(title="Avg Closeness Score (0–1)", range=[0, 1]),
    yaxis2=dict(title="Close Match %", overlaying="y", side="right", range=[0, 100]),
    height=370, template="plotly_white", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=40),
)
st.plotly_chart(fig7, use_container_width=True)

# ── Raw data toggle ─────────────────────────────────────────────────────────
with st.expander("View underlying data tables"):
    st.write("**Season Scoring Trends**")
    scoring_disp = scoring.copy()
    scoring_disp["season"] = scoring_disp["season"].astype(int)
    st.dataframe(
        scoring_disp.round(2),
        use_container_width=True, hide_index=True,
        column_config={
            "season": st.column_config.NumberColumn("Season", format="%d"),
            "avg_first_innings_score": st.column_config.NumberColumn("Avg Score", format="%.1f"),
            "boundary_pct": st.column_config.ProgressColumn("Boundary %", min_value=0, max_value=50, format="%.1f%%"),
            "dot_ball_pct": st.column_config.ProgressColumn("Dot Ball %", min_value=0, max_value=60, format="%.1f%%"),
            "six_pct": st.column_config.ProgressColumn("Six %", min_value=0, max_value=20, format="%.1f%%"),
            "yoy_score_change_pct": st.column_config.NumberColumn("YoY Change %", format="%.2f"),
        }
    )
    st.write("**Toss Decision Trend**")
    st.dataframe(toss.round(2), use_container_width=True, hide_index=True,
        column_config={
            "season": st.column_config.NumberColumn("Season", format="%d"),
            "field_first_pct": st.column_config.ProgressColumn("Chose Field %", min_value=0, max_value=100, format="%.1f%%"),
            "field_win_pct": st.column_config.ProgressColumn("Chase Win %", min_value=0, max_value=100, format="%.1f%%"),
        }
    )
    st.write("**Competitiveness Trend**")
    st.dataframe(compete.round(3), use_container_width=True, hide_index=True,
        column_config={
            "season": st.column_config.NumberColumn("Season", format="%d"),
            "avg_closeness": st.column_config.ProgressColumn("Closeness Score", min_value=0, max_value=1, format="%.3f"),
            "close_match_pct": st.column_config.ProgressColumn("Close Match %", min_value=0, max_value=100, format="%.1f%%"),
        }
    )
