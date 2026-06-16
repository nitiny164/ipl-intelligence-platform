"""
Team vs Team — Head-to-head deep dive between any two IPL franchises.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.data_loader import load_matches, load_deliveries, load_players, load_teams, processed_exists
from app.style import inject_styles, page_header, CHART_COLORS
from src.module_2_teams import id_to_name
from src.module_2b_h2h import (
    h2h_summary, h2h_season_trend, h2h_match_history,
    h2h_phase_battle, h2h_pom_leaders, h2h_venue_split, h2h_toss_impact,
)

st.set_page_config(page_title="Team vs Team | IPL Intelligence", layout="wide")
inject_styles()

if not processed_exists():
    st.error("Run `python -m src.module_0_foundation` first.")
    st.stop()

m = load_matches()
d = load_deliveries()
p = load_players()
t = load_teams()

id2name  = id_to_name(t)
name2id  = {v: k for k, v in id2name.items()}
active   = set(
    m[m["is_completed"]]["team1"].dropna().astype(int).tolist() +
    m[m["is_completed"]]["team2"].dropna().astype(int).tolist()
)
all_teams = sorted([id2name[tid] for tid in active if tid in id2name])

page_header("compare_arrows", "Team vs Team",
            "Pick any two franchises and see every dimension of their rivalry — history, phases, venues, key players.")

_HINT = "color:#9E9E9E;font-size:0.72rem;font-style:italic;margin:2px 0 12px"

# ── Team selectors ────────────────────────────────────────────────────────────
col_a, col_mid, col_b = st.columns([5, 1, 5])
with col_a:
    team_a_name = st.selectbox("Team A", all_teams,
        index=all_teams.index("Mumbai Indians") if "Mumbai Indians" in all_teams else 0,
        key="tvt_a")
with col_mid:
    st.markdown("<div style='text-align:center;font-size:1.4rem;font-weight:700;padding-top:28px;color:#546E7A'>vs</div>",
                unsafe_allow_html=True)
with col_b:
    default_b = "Chennai Super Kings" if "Chennai Super Kings" in all_teams else all_teams[1]
    team_b_name = st.selectbox("Team B", all_teams,
        index=all_teams.index(default_b),
        key="tvt_b")

if team_a_name == team_b_name:
    st.warning("Please select two different teams.")
    st.stop()

team_a = name2id[team_a_name]
team_b = name2id[team_b_name]

COLOR_A = "#1565C0"
COLOR_B = "#C62828"

# Season filter
seasons = sorted(m["season"].unique())
season_range = st.slider("Season range", int(min(seasons)), int(max(seasons)),
                         (int(min(seasons)), int(max(seasons))), key="tvt_seasons")
st.markdown('<p style="' + _HINT + '">Drag to zoom into a specific era — e.g. 2016–2026 for modern IPL only.</p>',
            unsafe_allow_html=True)

m_filtered = m[m["season"].between(season_range[0], season_range[1])]

# ── 1. Summary KPI strip ──────────────────────────────────────────────────────
summ = h2h_summary(m_filtered, team_a, team_b)

if not summ:
    st.info(f"No completed matches found between {team_a_name} and {team_b_name} in the selected range.")
    st.stop()

st.markdown("---")

# Win % donut + KPI cards side by side
left, right = st.columns([2, 3])

with left:
    fig_donut = go.Figure(go.Pie(
        labels=[team_a_name, team_b_name, "No Result"],
        values=[summ["a_wins"], summ["b_wins"], summ["no_result"]],
        hole=0.6,
        marker_colors=[COLOR_A, COLOR_B, "#E0E0E0"],
        textinfo="percent",
        textposition="inside",
        textfont=dict(size=12, color="white"),
        automargin=True,
        hovertemplate="<b>%{label}</b><br>%{value} wins (%{percent})<extra></extra>",
    ))
    fig_donut.update_layout(
        height=300, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5,
                    xanchor="left", x=1.02,
                    font=dict(size=12, color="#1A1A2E")),
        annotations=[dict(text=f"{summ['total']}<br>matches", x=0.5, y=0.5,
                          font_size=14, showarrow=False, font_color="#1A1A2E")],
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with right:
    # Last 5 matches strip
    last5_html = ""
    for res in summ["last5"]:
        if res == "A":
            badge = f'<span style="background:{COLOR_A};color:white;padding:3px 9px;border-radius:4px;font-size:0.75rem;font-weight:700;margin:2px">{team_a_name.split()[0]}</span>'
        elif res == "B":
            badge = f'<span style="background:{COLOR_B};color:white;padding:3px 9px;border-radius:4px;font-size:0.75rem;font-weight:700;margin:2px">{team_b_name.split()[0]}</span>'
        else:
            badge = '<span style="background:#E0E0E0;color:#546E7A;padding:3px 9px;border-radius:4px;font-size:0.75rem;margin:2px">NR</span>'
        last5_html += badge

    # Biggest wins
    a_best = ""
    if summ["a_biggest_runs"]:
        a_best += f"Biggest win: {summ['a_biggest_runs']} runs"
    if summ["a_biggest_wkts"]:
        a_best += f"  /  {summ['a_biggest_wkts']} wkts"

    b_best = ""
    if summ["b_biggest_runs"]:
        b_best += f"Biggest win: {summ['b_biggest_runs']} runs"
    if summ["b_biggest_wkts"]:
        b_best += f"  /  {summ['b_biggest_wkts']} wkts"

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
      <div style="background:#EEF4FF;border-left:4px solid {COLOR_A};border-radius:8px;padding:14px">
        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:{COLOR_A}">{team_a_name}</div>
        <div style="font-size:2rem;font-weight:800;color:#1A1A2E">{summ['a_wins']}<span style="font-size:0.85rem;font-weight:400;color:#546E7A"> wins ({summ['a_win_pct']}%)</span></div>
        <div style="font-size:0.72rem;color:#9E9E9E;margin-top:2px">{a_best}</div>
      </div>
      <div style="background:#FFEBEE;border-left:4px solid {COLOR_B};border-radius:8px;padding:14px">
        <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;color:{COLOR_B}">{team_b_name}</div>
        <div style="font-size:2rem;font-weight:800;color:#1A1A2E">{summ['b_wins']}<span style="font-size:0.85rem;font-weight:400;color:#546E7A"> wins ({summ['b_win_pct']}%)</span></div>
        <div style="font-size:0.72rem;color:#9E9E9E;margin-top:2px">{b_best}</div>
      </div>
    </div>
    <div style="background:#fff;border:1px solid #E3E8EF;border-radius:8px;padding:12px 16px">
      <div style="font-size:0.7rem;color:#9E9E9E;margin-bottom:6px">LAST 5 MEETINGS (oldest → newest)</div>
      <div>{last5_html}</div>
    </div>
    <div style="font-size:0.7rem;color:#9E9E9E;margin-top:6px">
      First meeting: {summ['first_meeting']} &nbsp;·&nbsp; Last meeting: {summ['last_meeting']}
    </div>
    """, unsafe_allow_html=True)

# ── 2. Cumulative Win % trend ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Win Dominance Over Time")
st.markdown('<p style="' + _HINT + '">How to read: Each point = one match played between these two teams. '
            'The blue line = cumulative win % of Team A up to that match. Red = Team B. '
            'Lines crossing = shift in dominance. A steadily rising blue line = Team A is consistently stronger.</p>',
            unsafe_allow_html=True)

trend = h2h_season_trend(m_filtered, team_a, team_b)
if not trend.empty:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["match_num"], y=trend["cumulative_a_win_pct"],
        mode="lines", name=team_a_name,
        line=dict(color=COLOR_A, width=2.5),
        fill="tozeroy", fillcolor=f"rgba(21,101,192,0.08)",
        hovertemplate=f"<b>Match %{{x}}</b><br>{team_a_name}: %{{y:.1f}}%<extra></extra>",
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend["match_num"], y=trend["cumulative_b_win_pct"],
        mode="lines", name=team_b_name,
        line=dict(color=COLOR_B, width=2.5),
        hovertemplate=f"<b>Match %{{x}}</b><br>{team_b_name}: %{{y:.1f}}%<extra></extra>",
    ))
    fig_trend.add_hline(y=50, line_dash="dot", line_color="#BDBDBD",
                        annotation_text="50% (equal)", annotation_font_color="#BDBDBD")

    # Mark wins as scatter dots
    a_wins_idx = trend[trend["winner_flag"] == "A"]
    b_wins_idx = trend[trend["winner_flag"] == "B"]
    fig_trend.add_trace(go.Scatter(
        x=a_wins_idx["match_num"], y=a_wins_idx["cumulative_a_win_pct"],
        mode="markers", name=f"{team_a_name} win",
        marker=dict(color=COLOR_A, size=7, symbol="circle"),
        showlegend=False,
        hovertemplate="<b>%{text}</b><extra></extra>",
        text=[f"{team_a_name} won at {v}" for v in a_wins_idx["venue"]],
    ))
    fig_trend.add_trace(go.Scatter(
        x=b_wins_idx["match_num"], y=b_wins_idx["cumulative_b_win_pct"],
        mode="markers", name=f"{team_b_name} win",
        marker=dict(color=COLOR_B, size=7, symbol="circle"),
        showlegend=False,
        hovertemplate="<b>%{text}</b><extra></extra>",
        text=[f"{team_b_name} won at {v}" for v in b_wins_idx["venue"]],
    ))

    fig_trend.update_layout(
        xaxis_title="Match Number (chronological order)",
        yaxis=dict(title="Cumulative Win %", range=[0, 100], ticksuffix="%"),
        height=380, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# ── 3. Phase Battle ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Phase Battle — Who Dominates Each Phase?")
st.markdown('<p style="' + _HINT + '">How to read: Each panel = one phase (Powerplay / Middle / Death). '
            'Blue bars = Team A batting in this fixture. Red bars = Team B batting. '
            'Compare run rates and dot ball % to see which team controls each phase when these two meet.</p>',
            unsafe_allow_html=True)

phase = h2h_phase_battle(d, m_filtered, team_a, team_b)
phase_a = phase.get("team_a", pd.DataFrame())
phase_b = phase.get("team_b", pd.DataFrame())

if not phase_a.empty and not phase_b.empty:
    from plotly.subplots import make_subplots

    phases     = ["powerplay", "middle", "death"]
    phase_disp = ["Powerplay\n(Overs 1-6)", "Middle\n(Overs 7-15)", "Death\n(Overs 16-20)"]
    metrics    = [("run_rate", "Run Rate (rpo)"), ("boundary_pct", "Boundary %"), ("dot_pct", "Dot Ball %")]

    fig_phase = make_subplots(rows=1, cols=3,
                               subplot_titles=[f"<b>{ph}</b>" for ph in phase_disp],
                               shared_yaxes=False, horizontal_spacing=0.08)

    show_leg = True
    for ci, (metric, metric_label) in enumerate(metrics, 1):
        a_vals = [float(phase_a[phase_a["over_phase"] == ph][metric].iloc[0])
                  if ph in phase_a["over_phase"].values else 0 for ph in phases]
        b_vals = [float(phase_b[phase_b["over_phase"] == ph][metric].iloc[0])
                  if ph in phase_b["over_phase"].values else 0 for ph in phases]

        fig_phase.add_trace(go.Bar(
            name=team_a_name, x=phases, y=a_vals,
            marker_color=COLOR_A, showlegend=show_leg,
            text=[f"{v:.1f}" for v in a_vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=11, family="Arial Black"),
            width=0.38,
            hovertemplate=f"<b>%{{x}}</b><br>{team_a_name}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=ci)

        fig_phase.add_trace(go.Bar(
            name=team_b_name, x=phases, y=b_vals,
            marker_color=COLOR_B, showlegend=show_leg,
            text=[f"{v:.1f}" for v in b_vals],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="white", size=11, family="Arial Black"),
            width=0.38,
            hovertemplate=f"<b>%{{x}}</b><br>{team_b_name}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=ci)

        show_leg = False

    fig_phase.update_layout(
        barmode="group", height=400, template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5, font=dict(size=12)),
        margin=dict(t=60, b=80),
    )
    fig_phase.update_annotations(font=dict(size=12))
    st.plotly_chart(fig_phase, use_container_width=True)

# ── 4. Toss Impact ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Toss & Chase Dynamics")
st.markdown('<p style="' + _HINT + '">How to read: In this fixture, does winning the toss help? '
            'And which strategy — batting first or chasing — produces more wins?</p>',
            unsafe_allow_html=True)

toss = h2h_toss_impact(m_filtered, team_a, team_b)
if toss:
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Toss Winner Wins", f"{toss['toss_winner_wins']}/{toss['total']}",
               f"{toss['toss_winner_win_pct']}%",
               help="How often the team that won the toss also won the match")
    tc2.metric("Wins Batting First", str(toss["bat_first_wins"]),
               help="Matches won by the team that batted first (won by runs)")
    tc3.metric("Wins Chasing", str(toss["chase_wins"]),
               help="Matches won by the team batting second (won by wickets)")

# ── 5. Venue Breakdown ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Venue Breakdown")
st.markdown('<p style="' + _HINT + '">How to read: Bars show how many times each team won at each venue. '
            'Longer blue bar = Team A dominates at that ground. '
            'Venues sorted by total matches played between these teams.</p>',
            unsafe_allow_html=True)

venue_df = h2h_venue_split(m_filtered, team_a, team_b, id2name)
if not venue_df.empty:
    name_a_col = id2name.get(team_a, "Team A")
    name_b_col = id2name.get(team_b, "Team B")
    top_v = venue_df.head(15)

    fig_venue = go.Figure()
    fig_venue.add_trace(go.Bar(
        name=team_a_name, y=top_v["Venue"], x=top_v[name_a_col],
        orientation="h", marker_color=COLOR_A,
        text=top_v[name_a_col], textposition="inside",
        textfont=dict(color="white", size=11),
        hovertemplate=f"<b>%{{y}}</b><br>{team_a_name}: %{{x}} wins<extra></extra>",
    ))
    fig_venue.add_trace(go.Bar(
        name=team_b_name, y=top_v["Venue"], x=top_v[name_b_col],
        orientation="h", marker_color=COLOR_B,
        text=top_v[name_b_col], textposition="inside",
        textfont=dict(color="white", size=11),
        hovertemplate=f"<b>%{{y}}</b><br>{team_b_name}: %{{x}} wins<extra></extra>",
    ))
    fig_venue.update_layout(
        barmode="group",
        xaxis_title="Wins", yaxis_title="",
        height=max(360, len(top_v) * 28),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20, b=40, l=10),
    )
    st.plotly_chart(fig_venue, use_container_width=True)

# ── 6. Player of the Match Leaders ────────────────────────────────────────────
st.markdown("---")
st.subheader("Match-Winners — Player of the Match Leaders")
st.markdown('<p style="' + _HINT + '">Players who have won the most Player of the Match awards '
            'whenever these two teams met. These are the players who consistently show up in this rivalry.</p>',
            unsafe_allow_html=True)

pom = h2h_pom_leaders(m_filtered, p, team_a, team_b, top_n=10)
if not pom.empty:
    fig_pom = px.bar(
        pom, x="Awards", y="Player", orientation="h",
        color="Awards", color_continuous_scale=[COLOR_A, "#4FC3F7"],
        text="Awards",
        labels={"Awards": "POM Awards", "Player": ""},
    )
    fig_pom.update_traces(textposition="inside", textfont=dict(color="white", size=12))
    fig_pom.update_layout(
        height=max(300, len(pom) * 36),
        template="plotly_white",
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Number of Player of Match Awards", dtick=1),
        margin=dict(t=10, b=40, l=10),
    )
    st.plotly_chart(fig_pom, use_container_width=True)

# ── 7. Full Match History ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Complete Match History")
st.markdown('<p style="' + _HINT + '">All matches between these two teams — newest first. '
            'Expand to search, sort, or export.</p>',
            unsafe_allow_html=True)

history = h2h_match_history(m_filtered, p, team_a, team_b, id2name)
if not history.empty:
    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date":           st.column_config.TextColumn("Date",             width="small"),
            "Season":         st.column_config.NumberColumn("Season",         format="%d", width="small"),
            "Venue":          st.column_config.TextColumn("Venue",            width="large"),
            "Winner":         st.column_config.TextColumn("Winner",           width="medium"),
            "Margin":         st.column_config.TextColumn("Margin",           width="small"),
            "Player of Match":st.column_config.TextColumn("Player of Match",  width="medium"),
        },
    )
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button("Download match history as CSV", csv,
                       file_name=f"h2h_{team_a_name}_vs_{team_b_name}.csv",
                       mime="text/csv")
