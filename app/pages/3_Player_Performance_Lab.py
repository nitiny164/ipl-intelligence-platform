"""
Module 3 — Player Performance Lab
Business question: How good is this player, really — beyond raw stats?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.data_loader import load_matches, load_deliveries, load_players, processed_exists, DOCS_DIR
from app.style import inject_styles, page_header, section_header, CHART_COLORS, PALETTE, kpi_grid
from src.module_3_players import (
    career_batting_stats, career_bowling_stats, season_batting_stats,
    player_phase_profile, compute_impact_score, compute_clutch_differential,
    archetype_quadrant, finisher_index, consistency_score,
)

st.set_page_config(page_title="Player Performance Lab | IPL Intelligence", layout="wide")

inject_styles()

if not processed_exists():
    st.error("Run `python -m src.module_0_foundation` first.")
    st.stop()

page_header("science", "Player Performance Lab",
            "Beyond raw stats — Impact Score, Clutch Differential, archetype analysis.")

m = load_matches()
d = load_deliveries()
p = load_players()


# ── Pre-compute (cached via Streamlit) ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_all():
    bat  = career_batting_stats(d, p, min_innings=10)
    bowl = career_bowling_stats(d, p, min_overs=10)
    imp  = compute_impact_score(d, bat, bowl, matches=m)
    bat_c, bowl_c = compute_clutch_differential(d, m)
    arch = archetype_quadrant(bat)
    fi   = finisher_index(d, p, min_death_balls=60)
    cs   = consistency_score(d, p, min_innings=15)
    bat_phase, bowl_phase = player_phase_profile(d, p)
    season_bat = season_batting_stats(d)
    return bat, bowl, imp, bat_c, bowl_c, arch, fi, cs, bat_phase, bowl_phase, season_bat

bat, bowl, imp, bat_c, bowl_c, arch, fi, cs, bat_phase, bowl_phase, season_bat = _load_all()

@st.cache_data(show_spinner=False)
def _impact_for_season(season_id):
    """Recompute impact score filtered to a single season (no shrinkage, lower thresholds)."""
    d_s = d[d["season_id"] == season_id]
    m_s = m[m["season_id"] == season_id]
    bat_s  = career_batting_stats(d_s, p, min_innings=1)
    bowl_s = career_bowling_stats(d_s, p, min_overs=1)
    return compute_impact_score(d_s, bat_s, bowl_s, matches=m_s, career_mode=False)

@st.cache_data(show_spinner=False)
def _clutch_for_season(season_id, career_bat_avg: float, career_sr: float):
    """Recompute clutch for a single season using career-level league baselines.

    Passing fixed_league_bat_avg / fixed_league_avg_sr ensures season scores are
    on the same scale as career scores — a +10 in 2016 means the same as +10 career.
    """
    d_s = d[d["season_id"] == season_id]
    m_s = m[m["season_id"] == season_id]
    return compute_clutch_differential(
        d_s, m_s,
        min_bat_innings=5, min_target=140, min_bat_balls_per_inn=10,
        min_bowl_normal=100, min_bowl_pressure=30, min_bowl_innings=2,
        fixed_league_bat_avg=career_bat_avg,
        fixed_league_avg_sr=career_sr,
    )

all_seasons = sorted(d["season_id"].unique().tolist())

# Player selector
all_batters  = sorted(bat["batter"].unique().tolist())
all_bowlers  = sorted(bowl["bowler"].unique().tolist())
name_map     = dict(zip(p["player_name"], p["player_full_name"]))
def _fmt(short):
    full = name_map.get(short, "")
    return f"{short} ({full})" if full and full != short else short

tabs = st.tabs(["Batter Analysis", "Bowler Analysis", "Impact Score", "Clutch Score", "Comparison Tool", "Death Overs"])

def _styled_table(df: pd.DataFrame, num_cols: list = None, pct_cols: list = None,
                  highlight_col: str = None, highlight_ascending: bool = False):
    """Render a DataFrame as a styled HTML table."""
    num_cols  = num_cols  or []
    pct_cols  = pct_cols  or []

    header_cells = "".join(
        f'<th style="background:#1565C0;color:white;padding:10px 14px;text-align:left;'
        f'font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;'
        f'white-space:nowrap">{c}</th>'
        for c in df.columns
    )

    # Determine highlight range for optional colour gradient
    hl_min = hl_max = None
    if highlight_col and highlight_col in df.columns:
        hl_min = df[highlight_col].min()
        hl_max = df[highlight_col].max()

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFF"
        cells = ""
        for col in df.columns:
            val = row[col]
            is_first = col == df.columns[0]
            weight   = "700" if is_first else "400"
            color    = "#1A1A2E" if is_first else "#37474F"
            align    = "left" if is_first else "right"

            # Format value
            if col in pct_cols and isinstance(val, (int, float)):
                display = f"{val:.1f}%"
            elif col in num_cols and isinstance(val, (int, float)) and not pd.isna(val):
                display = f"{val:,.2f}" if val != int(val) else f"{int(val):,}"
            else:
                display = str(val) if pd.notna(val) else "—"

            cells += (
                f'<td style="padding:9px 14px;text-align:{align};font-size:0.82rem;'
                f'font-weight:{weight};color:{color};white-space:nowrap;border-bottom:1px solid #EEF1F5">'
                f'{display}</td>'
            )
        rows_html += f'<tr style="background:{bg}">{cells}</tr>'

    html = (
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #E3E8EF;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:16px">'
        f'<table style="width:100%;border-collapse:collapse;font-family:sans-serif">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — BATTER ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Batter Career Profile")
    sel_batter = st.selectbox("Select Batter", all_batters,
        index=all_batters.index("V Kohli") if "V Kohli" in all_batters else 0,
        format_func=_fmt, key="batter_sel")

    bat_row = bat[bat["batter"] == sel_batter]
    if not bat_row.empty:
        r = bat_row.iloc[0]
        kpis = [
            ("#1565C0", "", "Total Runs",   f"{int(r['total_runs']):,}", "Career runs scored"),
            ("#546E7A", "", "Innings",       f"{int(r['innings'])}",     "Times batted"),
            ("#2E7D32", "", "Strike Rate",   f"{r['strike_rate']:.1f}",  "Runs per 100 balls"),
            ("#6A1B9A", "", "Average",       f"{r['batting_avg']:.1f}",  "Runs per dismissal"),
            ("#E65100", "", "Boundary %",    f"{r['boundary_pct']:.1f}%","% balls hit to boundary"),
            ("#C62828", "", "Dot Ball %",    f"{r['dot_pct']:.1f}%",     "% balls with no run"),
        ]
        cards_html = "".join([
            f'<div style="background:#FFFFFF;border:1px solid #E3E8EF;border-top:4px solid {c};'
            f'border-radius:10px;padding:16px 14px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<p style="margin:0 0 4px;font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{c}">{icon} {label}</p>'
            f'<p style="margin:0;font-size:1.6rem;font-weight:800;color:#1A1A2E;line-height:1.1">{val}</p>'
            f'<p style="margin:4px 0 0;font-size:0.68rem;color:#9E9E9E">{desc}</p>'
            f'</div>'
            for c, icon, label, val, desc in kpis
        ])
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px">{cards_html}</div>',
            unsafe_allow_html=True,
        )

    # Career trajectory
    traj = season_bat[season_bat["batter"] == sel_batter].sort_values("season")
    if len(traj) >= 2:
        st.subheader(f"Season Trajectory — {name_map.get(sel_batter, sel_batter)}")
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Bar(x=traj["season"], y=traj["runs"],
            name="Runs", marker_color="rgba(31,119,180,0.6)"))
        fig_traj.add_trace(go.Scatter(x=traj["season"], y=traj["strike_rate"],
            name="Strike Rate", yaxis="y2", line=dict(color="#d62728", width=2),
            mode="lines+markers"))
        fig_traj.update_layout(
            yaxis=dict(title="Runs"),
            yaxis2=dict(title="Strike Rate", overlaying="y", side="right"),
            height=340, template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    # Phase profile
    phase_row = bat_phase[bat_phase["batter"] == sel_batter]
    if not phase_row.empty:
        pr = phase_row.iloc[0]
        st.subheader("Phase Contribution Breakdown")
        ph_kpis = [
            ("#1565C0", "", "Batting Role",  pr["batting_role"],                  "Playing style classification"),
            ("#1B5E20", "", "Powerplay %",   f"{pr['bat_pp_pct']*100:.1f}%",     "% runs in overs 1–6"),
            ("#E65100", "", "Middle Overs %",f"{pr['bat_mid_pct']*100:.1f}%",    "% runs in overs 7–15"),
            ("#B71C1C", "", "Death Overs %", f"{pr['bat_death_pct']*100:.1f}%",  "% runs in overs 16–20"),
        ]
        ph_html = "".join([
            f'<div style="background:#FFFFFF;border:1px solid #E3E8EF;border-top:4px solid {c};'
            f'border-radius:10px;padding:16px 14px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<p style="margin:0 0 4px;font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{c}">{icon} {label}</p>'
            f'<p style="margin:0;font-size:1.5rem;font-weight:800;color:#1A1A2E;line-height:1.1">{val}</p>'
            f'<p style="margin:4px 0 0;font-size:0.68rem;color:#9E9E9E">{desc}</p>'
            f'</div>'
            for c, icon, label, val, desc in ph_kpis
        ])
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px">{ph_html}</div>',
            unsafe_allow_html=True,
        )

        _ph_vals = [pr["bat_pp_pct"]*100, pr["bat_mid_pct"]*100, pr["bat_death_pct"]*100]
        fig_phase = go.Figure(go.Bar(
            x=["Powerplay", "Middle", "Death"],
            y=_ph_vals,
            marker_color=["#1f77b4","#2ca02c","#d62728"],
            text=[f"{v:.1f}%" for v in _ph_vals],
            textposition="outside",
            textfont=dict(size=13, color="#1A1A2E"),
        ))
        fig_phase.update_layout(
            yaxis=dict(title="% of Total Runs", range=[0, max(_ph_vals) * 1.25]),
            height=320, template="plotly_white", showlegend=False,
        )
        st.plotly_chart(fig_phase, use_container_width=True)

    # Archetype quadrant chart
    st.subheader("Batter Archetype Quadrant (SR × Average)")
    st.caption("Median strike rate and average as axes. Your selected batter is highlighted.")
    if not arch.empty:
        color_map = {"Match-Winner":"#2ca02c","Hitter":"#ff7f0e","Accumulator":"#1f77b4","Passenger":"#7f7f7f"}
        fig_arch = px.scatter(arch, x="batting_avg", y="strike_rate",
            color="archetype", color_discrete_map=color_map,
            hover_name="batter", hover_data={"total_runs": True, "innings": True},
            labels={"batting_avg": "Batting Average", "strike_rate": "Strike Rate"},
            opacity=0.65,
        )
        if sel_batter in arch["batter"].values:
            hl = arch[arch["batter"] == sel_batter].iloc[0]
            fig_arch.add_trace(go.Scatter(x=[hl["batting_avg"]], y=[hl["strike_rate"]],
                mode="markers+text", marker=dict(size=14, color="black", symbol="star"),
                text=[sel_batter], textposition="top right", showlegend=False, name="Selected"))
        sr_med  = arch["sr_median"].iloc[0]
        avg_med = arch["avg_median"].iloc[0]
        fig_arch.add_hline(y=sr_med,  line_dash="dot", line_color="gray")
        fig_arch.add_vline(x=avg_med, line_dash="dot", line_color="gray")
        fig_arch.update_layout(height=460, template="plotly_white")
        st.plotly_chart(fig_arch, use_container_width=True)

    # Consistency table — belongs in Batter Analysis
    st.subheader("Batter Consistency — True Consistent Performers")
    st.caption(
        "Consistency Score = Batting Average × Stability (1/CV). "
        "A player must score high AND score steadily to rank here. "
        "High average + low variance = truly consistent. Low average players rank lower even if stable."
    )
    # Merge batting_role from bat_phase and archetype from arch
    _cs_base = cs.head(20)[["batter","player_full_name","batting_avg","mean_runs",
                             "std_runs","stability_score","consistency_score","boundary_dependency"]].copy()
    _role_map  = dict(zip(bat_phase["batter"], bat_phase["batting_role"]))
    _arch_map  = dict(zip(arch["batter"],      arch["archetype"]))
    _cs_base["Phase Role"] = _cs_base["batter"].map(_role_map).fillna("—")
    _cs_base["Style"]      = _cs_base["batter"].map(_arch_map).fillna("—")
    # Combined role label
    _cs_base["Role"] = _cs_base.apply(
        lambda r: f"{r['Phase Role']} · {r['Style']}" if r["Phase Role"] != "—" else r["Style"],
        axis=1
    )
    top_cs = _cs_base[["player_full_name","Role","batting_avg","mean_runs","std_runs","stability_score","consistency_score","boundary_dependency"]].copy()
    top_cs.columns = ["Player","Role","Bat Avg","Avg Runs/Inns","Std Dev","Stability","Consistency Score","Boundary Dep%"]
    top_cs["Bat Avg"]       = top_cs["Bat Avg"].round(1)
    top_cs["Avg Runs/Inns"] = top_cs["Avg Runs/Inns"].round(1)
    top_cs["Std Dev"]       = top_cs["Std Dev"].round(1)
    _styled_table(top_cs,
                  num_cols=["Bat Avg","Avg Runs/Inns","Std Dev","Stability","Consistency Score"],
                  pct_cols=["Boundary Dep%"],
                  highlight_col="Consistency Score")

    st.markdown("""
<div style="margin-top:12px;padding:14px 18px;background:#F8FAFF;border-radius:8px;border:1px solid #E3E8EF;">
<p style="margin:0 0 8px;font-size:0.78rem;font-weight:700;color:#5C6BC0;letter-spacing:0.05em;text-transform:uppercase;">Role Legend</p>
<div style="display:flex;flex-wrap:wrap;gap:18px;">
<div><span style="font-size:0.8rem;font-weight:600;color:#1A1A2E;">Phase Role</span>
<ul style="margin:4px 0 0 0;padding-left:16px;font-size:0.78rem;color:#555;line-height:1.8;">
<li><b>Powerplay Hitter</b> — Scores majority of runs in overs 1–6; exploits fielding restrictions</li>
<li><b>Anchor</b> — Runs concentrated in middle overs 7–15; builds innings, rotates strike</li>
<li><b>Finisher</b> — Runs concentrated in death overs 16–20; maximises at the end</li>
<li><b>Versatile</b> — Contributes evenly across all phases; no dominant zone</li>
</ul></div>
<div><span style="font-size:0.8rem;font-weight:600;color:#1A1A2E;">Archetype (SR × Average)</span>
<ul style="margin:4px 0 0 0;padding-left:16px;font-size:0.78rem;color:#555;line-height:1.8;">
<li><b>Match-Winner</b> — High average <i>and</i> high strike rate; elite all-round threat</li>
<li><b>Power Hitter</b> — High strike rate but lower average; explosive, short bursts</li>
<li><b>Accumulator</b> — High average but lower strike rate; consistent, steady scorer</li>
<li><b>Support</b> — Below-median on both; specialist role or limited sample</li>
</ul></div>
</div>
<p style="margin:10px 0 0;font-size:0.75rem;color:#888;">Consistency Score = (Batting Avg ÷ 10) × Stability &nbsp;|&nbsp; Stability = 1 ÷ Coefficient of Variation &nbsp;|&nbsp; Higher = more reliable output</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — BOWLER ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Bowler Career Profile")
    sel_bowler = st.selectbox("Select Bowler", all_bowlers,
        index=all_bowlers.index("JJ Bumrah") if "JJ Bumrah" in all_bowlers else 0,
        format_func=_fmt, key="bowler_sel")

    bowl_row = bowl[bowl["bowler"] == sel_bowler]
    if not bowl_row.empty:
        r = bowl_row.iloc[0]
        b_kpis = [
            ("#C62828", "", "Wickets",      f"{int(r['wickets'])}",       "Career wickets"),
            ("#546E7A", "", "Matches",      f"{int(r['matches'])}",       "Matches bowled in"),
            ("#1565C0", "", "Economy",      f"{r['economy']:.2f}",        "Runs per over"),
            ("#6A1B9A", "", "Bowling Avg",  f"{r['bowling_avg']:.1f}",   "Runs per wicket"),
            ("#2E7D32", "", "Strike Rate",  f"{r['bowling_sr']:.1f}",    "Balls per wicket"),
            ("#E65100", "", "Dot Ball %",   f"{r['dot_pct']:.1f}%",      "% balls with no run"),
        ]
        b_html = "".join([
            f'<div style="background:#FFFFFF;border:1px solid #E3E8EF;border-top:4px solid {c};'
            f'border-radius:10px;padding:16px 14px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<p style="margin:0 0 4px;font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{c}">{icon} {label}</p>'
            f'<p style="margin:0;font-size:1.6rem;font-weight:800;color:#1A1A2E;line-height:1.1">{val}</p>'
            f'<p style="margin:4px 0 0;font-size:0.68rem;color:#9E9E9E">{desc}</p>'
            f'</div>'
            for c, icon, label, val, desc in b_kpis
        ])
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:16px">{b_html}</div>',
            unsafe_allow_html=True,
        )

    # Bowling phase profile
    bowl_phase_row = bowl_phase[bowl_phase["bowler"] == sel_bowler]
    if not bowl_phase_row.empty:
        bpr = bowl_phase_row.iloc[0]
        st.subheader("Bowling Phase Profile")
        bp_kpis = [
            ("#C62828", "", "Bowling Role",   bpr["bowling_role"],                 "Specialist phase"),
            ("#1B5E20", "", "Powerplay %",    f"{bpr['bowl_pp_pct']*100:.1f}%",   "% wickets in overs 1–6"),
            ("#E65100", "", "Middle Overs %", f"{bpr['bowl_mid_pct']*100:.1f}%",  "% wickets in overs 7–15"),
            ("#B71C1C", "", "Death Overs %",  f"{bpr['bowl_death_pct']*100:.1f}%","% wickets in overs 16–20"),
        ]
        bp_html = "".join([
            f'<div style="background:#FFFFFF;border:1px solid #E3E8EF;border-top:4px solid {c};'
            f'border-radius:10px;padding:16px 14px;box-shadow:0 1px 4px rgba(0,0,0,0.06)">'
            f'<p style="margin:0 0 4px;font-size:0.68rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:{c}">{icon} {label}</p>'
            f'<p style="margin:0;font-size:1.5rem;font-weight:800;color:#1A1A2E;line-height:1.1">{val}</p>'
            f'<p style="margin:4px 0 0;font-size:0.68rem;color:#9E9E9E">{desc}</p>'
            f'</div>'
            for c, icon, label, val, desc in bp_kpis
        ])
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px">{bp_html}</div>',
            unsafe_allow_html=True,
        )

    # Role map used by charts below
    _bowl_role_map = dict(zip(bowl_phase["bowler"], bowl_phase["bowling_role"]))

    # Phase wickets bar chart — mirrors the batter phase chart
    if not bowl_phase_row.empty:
        _bph_vals = [bpr["bowl_pp_pct"]*100, bpr["bowl_mid_pct"]*100, bpr["bowl_death_pct"]*100]
        fig_bphase = go.Figure(go.Bar(
            x=["Powerplay (1–6)", "Middle (7–15)", "Death (16–20)"],
            y=_bph_vals,
            marker_color=["#1565C0", "#2E7D32", "#C62828"],
            text=[f"{v:.1f}%" for v in _bph_vals],
            textposition="outside",
            textfont=dict(size=13, color="#1A1A2E"),
        ))
        fig_bphase.update_layout(
            title=dict(text=f"{sel_bowler} — Wicket Distribution by Phase", font=dict(size=13)),
            yaxis=dict(title="% of Career Wickets", range=[0, max(_bph_vals) * 1.3]),
            height=320, template="plotly_white", showlegend=False,
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_bphase, use_container_width=True)

    # Economy vs Wickets scatter — equivalent of batter archetype quadrant
    st.subheader("Bowler Landscape — Economy vs Wickets")
    st.caption("Each dot is a bowler. Lower economy = harder to score off. More wickets = more impact. Selected bowler highlighted.")
    _bowl_scatter = bowl[bowl["wickets"] >= 20].copy()
    _bowl_scatter["Role"] = _bowl_scatter["bowler"].map(_bowl_role_map).fillna("Other")
    _bowl_scatter["player_label"] = _bowl_scatter["player_full_name"]
    _bowl_scatter["dot_pct_fmt"] = _bowl_scatter["dot_pct"].round(1).astype(str) + "%"

    _role_colors = {
        "New-Ball Bowler": "#1565C0",
        "Middle-Overs Specialist": "#2E7D32",
        "Death Specialist": "#C62828",
        "Other": "#9E9E9E",
    }
    fig_bscatter = px.scatter(
        _bowl_scatter,
        x="economy", y="wickets",
        color="Role",
        color_discrete_map=_role_colors,
        hover_name="player_label",
        hover_data={"economy": ":.2f", "wickets": True, "dot_pct_fmt": True, "matches": True, "Role": False},
        labels={"economy": "Economy (runs per over)", "wickets": "Career Wickets",
                "dot_pct_fmt": "Dot Ball %", "matches": "Matches"},
        opacity=0.65,
    )
    if sel_bowler in _bowl_scatter["bowler"].values:
        _hl = _bowl_scatter[_bowl_scatter["bowler"] == sel_bowler].iloc[0]
        fig_bscatter.add_trace(go.Scatter(
            x=[_hl["economy"]], y=[_hl["wickets"]],
            mode="markers+text",
            marker=dict(size=14, color="black", symbol="star"),
            text=[sel_bowler], textposition="top right",
            showlegend=False, name="Selected",
        ))
    _eco_med = _bowl_scatter["economy"].median()
    _wkt_med = _bowl_scatter["wickets"].median()
    fig_bscatter.add_hline(y=_wkt_med, line_dash="dot", line_color="gray")
    fig_bscatter.add_vline(x=_eco_med, line_dash="dot", line_color="gray")
    fig_bscatter.update_layout(
        height=460, template="plotly_white",
        xaxis=dict(title="Economy — runs per over (lower = better)"),
        yaxis=dict(title="Career Wickets (higher = better)"),
        font=dict(family="Inter, sans-serif"),
        annotations=[
            dict(x=_bowl_scatter["economy"].min()+0.05, y=_bowl_scatter["wickets"].max()-5,
                 text="Best: low economy + high wickets", showarrow=False,
                 font=dict(size=10, color="#2E7D32"), xanchor="left"),
        ],
    )
    st.plotly_chart(fig_bscatter, use_container_width=True)

    # Top wicket-takers bar chart
    st.subheader("Top 20 Wicket-Takers — Career")
    _tb_chart = bowl[bowl["wickets"] >= 20].head(20).copy()
    _tb_chart["Role"] = _tb_chart["bowler"].map(_bowl_role_map).fillna("Other")
    _tb_chart["eco_fmt"] = _tb_chart["economy"].apply(lambda x: f"{x:.2f}")
    fig_tb = px.bar(
        _tb_chart.sort_values("wickets"),
        y="player_full_name", x="wickets", orientation="h",
        color="wickets", color_continuous_scale="Reds",
        text=_tb_chart.sort_values("wickets")["wickets"].astype(int),
        labels={"player_full_name": "", "wickets": "Career Wickets"},
        hover_data={"economy": ":.2f", "bowling_avg": ":.1f", "dot_pct": ":.1f"},
    )
    fig_tb.update_layout(
        height=560, template="plotly_white", coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=11)),
        font=dict(family="Inter, sans-serif", size=11),
        margin=dict(l=10, r=20, t=20, b=20),
    )
    fig_tb.update_traces(textposition="outside")
    st.plotly_chart(fig_tb, use_container_width=True)

    # Top wicket-takers leaderboard table
    st.subheader("Top Wicket-Takers — Detailed Stats")
    top_bowl = (
        bowl[bowl["wickets"] >= 20]
        .head(20)[["player_full_name","wickets","matches","overs","economy","bowling_avg","bowling_sr","dot_pct"]]
        .copy()
    )
    top_bowl["Role"] = bowl[bowl["wickets"] >= 20].head(20)["bowler"].map(_bowl_role_map).fillna("—").values
    top_bowl.columns = ["Player","Wickets","Matches","Overs","Economy","Avg","SR","Dot%","Role"]
    # Reorder: Player, Role first, then stats
    top_bowl = top_bowl[["Player","Role","Wickets","Matches","Overs","Economy","Avg","SR","Dot%"]]
    _styled_table(top_bowl, num_cols=["Wickets","Matches","Overs","Economy","Avg","SR"],
                  pct_cols=["Dot%"], highlight_col="Wickets")

    st.markdown("""
<div style="margin-top:12px;padding:14px 18px;background:#F8FAFF;border-radius:8px;border:1px solid #E3E8EF;">
<p style="margin:0 0 8px;font-size:0.78rem;font-weight:700;color:#5C6BC0;letter-spacing:0.05em;text-transform:uppercase;">Bowler Role Legend</p>
<div style="display:flex;flex-wrap:wrap;gap:18px;">
<div><span style="font-size:0.8rem;font-weight:600;color:#1A1A2E;">Phase Role</span>
<ul style="margin:4px 0 0 0;padding-left:16px;font-size:0.78rem;color:#555;line-height:1.9;">
<li><b>New-Ball Bowler</b> — Takes majority of wickets in Powerplay (overs 1–6); exploits swing, seam, and early movement to dismiss openers</li>
<li><b>Middle-Overs Specialist</b> — Most effective in overs 7–15; builds pressure through dot balls and spin, disrupts the set batter</li>
<li><b>Death Specialist</b> — Dominates overs 16–20; skilled at yorkers, slower balls, and restricting big hitters under pressure</li>
</ul></div>
</div>
<p style="margin:10px 0 0;font-size:0.75rem;color:#888;">Avg = Runs per wicket (lower is better) &nbsp;|&nbsp; SR = Balls per wicket (lower is better) &nbsp;|&nbsp; Dot% = % of balls that are dot balls (higher = more pressure) &nbsp;|&nbsp; Min. 20 career wickets</p>
</div>
""", unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPACT SCORE
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Impact Score — Original Metric")
    st.info(
        "**What is Impact Score?** Dream11-inspired event-based scoring — granular, "
        "context-aware, and calibrated so batting and bowling are on the same scale.\n\n"
        "**Batting:** runs (+0.5 each) + boundary bonuses (4→+1, 6→+2) + milestones "
        "(25/50/75/100 → +4/8/12/16) + SR adjustment (≥170→+6, <70→−6) + pressure "
        "chase multiplier (RRR>10 → ×1.5).\n\n"
        "**Bowling:** wicket base +16 + quality bonus +8 (dismissed premium batter SR>130) "
        "+ Bowled/LBW +4 + haul bonuses (3W→+4, 4W→+8, 5W→+16) + economy bands "
        "(<6→+6, <7→+2, >9→−8) + low-total/death-chase context ×1.3.\n\n"
        "**Career view:** per-match mean × Bayesian shrinkage (k=25) → penalises small "
        "samples; Kohli at 275 matches correctly outranks a 5-match wonder. "
        "**Season view:** total points across all season matches → Kohli's 16-match 2016 "
        "campaign (~895 pts) correctly tops the leaderboard."
    )

    # Season slicer
    _s_options = ["All Seasons"] + [str(s) for s in all_seasons]
    _s_sel = st.selectbox("Season", _s_options, index=0, key="imp_season")

    if _s_sel == "All Seasons":
        imp_use = imp
        _s_label = "Career (All Seasons)"
        st.caption(
            "**Career view** \u2014 min 20 qualifying innings (chasing target \u2265150, facing \u226515 balls). Batting average includes not-outs \u2014 finishing a chase unbeaten is full credit."
        )
    else:
        imp_use = _impact_for_season(int(_s_sel))
        _s_label = f"Season {_s_sel}"
        st.caption(
            f"**Season {_s_sel} view** — scores are total accumulated points across all matches in the season "
            "(typical range: 200–1000+ for top batters, 300–700 for top bowlers). "
            "A higher number means the player sustained their contribution across more matches. "
            "Players need ≥3 batting innings or ≥4 overs bowled to qualify."
        )

    # Methodology expandable
    methodology_path = DOCS_DIR / "impact_score_methodology.md"
    if methodology_path.exists():
        with st.expander("Full methodology & worked example (click to expand)"):
            st.markdown(methodology_path.read_text(encoding="utf-8"))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Top Batting Impact — {_s_label}**")
        top_bat_imp = imp_use.sort_values("batting_impact_score", ascending=False).head(20)
        _bat_name_map = dict(zip(p["player_name"], p["player_full_name"]))
        top_bat_imp = top_bat_imp.copy()
        top_bat_imp["player_label"] = top_bat_imp["player"].map(_bat_name_map).fillna(top_bat_imp["player"])
        fig_bat_imp = px.bar(top_bat_imp.sort_values("batting_impact_score"),
            y="player_label", x="batting_impact_score", orientation="h",
            color="batting_impact_score", color_continuous_scale="Blues",
            labels={"batting_impact_score":"Batting Impact Score","player_label":"Player"},
        )
        fig_bat_imp.update_layout(height=520, template="plotly_white", coloraxis_showscale=False,
                                   yaxis_title="", xaxis_title="Batting Impact Score")
        st.plotly_chart(fig_bat_imp, use_container_width=True)

    with col2:
        st.markdown(f"**Top Bowling Impact — {_s_label}**")
        top_bowl_imp = imp_use[imp_use["bowling_impact_score"] > 0].sort_values("bowling_impact_score", ascending=False).head(20)
        if not top_bowl_imp.empty:
            top_bowl_imp = top_bowl_imp.copy()
            top_bowl_imp["player_label"] = top_bowl_imp["player"].map(_bat_name_map).fillna(top_bowl_imp["player"])
            fig_bowl_imp = px.bar(top_bowl_imp.sort_values("bowling_impact_score"),
                y="player_label", x="bowling_impact_score", orientation="h",
                color="bowling_impact_score", color_continuous_scale="Reds",
                labels={"bowling_impact_score":"Bowling Impact Score","player_label":"Player"},
            )
            fig_bowl_imp.update_layout(height=520, template="plotly_white", coloraxis_showscale=False,
                                        yaxis_title="", xaxis_title="Bowling Impact Score")
            st.plotly_chart(fig_bowl_imp, use_container_width=True)
        else:
            st.info("No bowling data available for the selected season.")

    st.markdown(f"**Combined Impact Score Leaderboard — {_s_label}**")
    imp_display = imp_use.sort_values("combined_impact_score", ascending=False).head(30).copy()
    imp_display["Player"] = imp_display["player"].map(_bat_name_map).fillna(imp_display["player"])
    imp_display = imp_display.rename(columns={
        "batting_impact_score": "Batting Impact",
        "bowling_impact_score": "Bowling Impact",
        "combined_impact_score": "Combined Impact",
        "bat_matches": "Matches Batted",
        "bowl_matches": "Matches Bowled",
    })[["Player","Batting Impact","Bowling Impact","Combined Impact","Matches Batted","Matches Bowled"]]
    _styled_table(imp_display,
                  num_cols=["Batting Impact","Bowling Impact","Combined Impact","Matches Batted","Matches Bowled"],
                  highlight_col="Combined Impact")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUTCH DIFFERENTIAL
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Clutch Score — Original Metric")
    st.info(
        "**What is Clutch Score?** A Chase Batting Index measuring who performs best in competitive T20 chases.\n\n"
        "**Batting Clutch = (Batting Average \u2212 League Avg) \u00d7 60% + SR component \u00d7 25% + Win Contribution \u00d7 15% \u00d7 Shrinkage**\n\n"
        "\u2460 **Batting Average (with Not-Outs)** \u2014 runs / dismissals. A player who scores 70\u2605 to finish a chase gets full credit. League average: ~51 runs per qualifying pressure innings.\n\n"
        "\u2461 **Strike Rate component** \u2014 how efficiently you score vs league SR (~144). High SR in chases = more pressure absorbed per over.\n\n"
        "\u2462 **Win Contribution Rate** \u2014 % of pressure innings where you scored \u226525 AND your team won. Pure individual + outcome measure.\n\n"
        "\u2463 **Shrinkage (k=25)** \u2014 innings / (innings + 25). Needs ~25 qualifying innings before getting full weight. Penalises small samples heavily.\n\n"
        "**Pressure definition:** 2nd innings, chasing a target \u2265150, player faced \u226515 balls in that innings.\n"
        "**Why target \u2265150?** Captures genuine competitive T20 chases. Easy chases (target 120) tell us nothing about pressure performance."
    )

    methodology_path2 = DOCS_DIR / "clutch_contribution_methodology.md"
    if methodology_path2.exists():
        with st.expander("Full methodology & pressure-situation definition"):
            st.markdown(methodology_path2.read_text(encoding="utf-8"))

    # ── Season slicer ────────────────────────────────────────────────────────
    _cs_options = ["All Seasons"] + [str(s) for s in all_seasons]
    _cs_sel = st.selectbox("Season", _cs_options, index=0, key="clutch_season")

    # Career baselines — passed to season view so scores stay on the same scale
    _career_bat_avg = float(bat_c["league_bat_avg"].iloc[0]) if not bat_c.empty else 51.1
    _career_sr      = float(bat_c["league_avg_sr"].iloc[0])  if not bat_c.empty else 144.1

    if _cs_sel == "All Seasons":
        bat_c_use, bowl_c_use = bat_c, bowl_c
        _cs_label = "Career (All Seasons)"
        st.caption(
            "**Career view** — min 20 qualifying innings (chasing target ≥150, "
            "facing ≥15 balls). Batting average includes not-outs — "
            "finishing a chase unbeaten is full credit."
        )
    else:
        bat_c_use, bowl_c_use = _clutch_for_season(int(_cs_sel), _career_bat_avg, _career_sr)
        _cs_label = f"Season {_cs_sel}"
        st.caption(
            f"**Season {_cs_sel} view** \u2014 lower thresholds (\u22655 qualifying innings, \u226510 balls per innings, target \u2265140). Same batting-average formula, season scope only."
        )

    # ── helper: resolve short name → full name ──────────────────────────────
    _bat_nm  = dict(zip(p["player_name"], p["player_full_name"]))
    _bowl_nm = _bat_nm  # same mapping

    # ── BATTING CLUTCH ───────────────────────────────────────────────────────
    st.markdown(f"### Batting Clutch Score — {_cs_label}")
    st.caption(
        "**Clutch Score** = mean(innings Runs-Above-Average × win weight × SR modifier) × shrinkage.  "
        "Positive = consistently scores above league avg (16.3 runs) in pressure chases. Hover for details."
    )

    if bat_c_use.empty:
        st.warning("Not enough pressure-situation data for batting clutch in this season.")
    else:
        bat_c_full = bat_c_use.copy()
        bat_c_full["Player"]          = bat_c_full["batter"].map(_bat_nm).fillna(bat_c_full["batter"])
        bat_c_full["Clutch Score"]    = bat_c_full["clutch_score_bat"].round(1)
        bat_c_full["Bat Avg (Chase)"] = bat_c_full["bat_avg"].round(1)
        bat_c_full["SR (Pressure)"]   = bat_c_full["avg_pressure_sr"].round(1)
        bat_c_full["Win%"]            = (bat_c_full["pressure_win_rate"] * 100).round(0).astype(int)
        bat_c_full["NotOut%"]         = (bat_c_full["not_out_rate"] * 100).round(0).astype(int)
        bat_c_full["50s"]             = bat_c_full["fifties_pressure"].astype(int)
        bat_c_full["Innings"]         = bat_c_full["pressure_matches"].astype(int)

        _bc_top = bat_c_full.sort_values("Clutch Score", ascending=False).head(15)
        _bc_bot = bat_c_full.sort_values("Clutch Score", ascending=True).head(10)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("**Top 15 — Clutch Stars** (elevated pressure performance + team won)")
            fig_bc = px.bar(
                _bc_top.sort_values("Clutch Score"),
                y="Player", x="Clutch Score", orientation="h",
                color="Clutch Score",
                color_continuous_scale=["#ffffbf","#1a9850"],
                color_continuous_midpoint=0,
                hover_data={"Bat Avg (Chase)": True, "SR (Pressure)": True,
                            "Win%": True, "NotOut%": True, "Innings": True},
                labels={"Clutch Score": "Clutch Score"},
            )
            fig_bc.add_vline(x=0, line_color="black", line_width=1)
            fig_bc.update_layout(height=480, template="plotly_white", coloraxis_showscale=False,
                                  yaxis_title="", xaxis_title="Clutch Score (Runs-Above-Avg × Win Weight × Shrinkage)")
            st.plotly_chart(fig_bc, use_container_width=True)

        with col2:
            st.markdown("**Bottom 10 — Under Pressure** (performance drops or low win rate)")
            fig_bc2 = px.bar(
                _bc_bot.sort_values("Clutch Score", ascending=False),
                y="Player", x="Clutch Score", orientation="h",
                color="Clutch Score",
                color_continuous_scale=["#d73027","#ffffbf"],
                color_continuous_midpoint=0,
                hover_data={"Bat Avg (Chase)": True, "SR (Pressure)": True,
                            "Win%": True, "NotOut%": True, "Innings": True},
                labels={"Clutch Score": "Clutch Score"},
            )
            fig_bc2.add_vline(x=0, line_color="black", line_width=1)
            fig_bc2.update_layout(height=380, template="plotly_white", coloraxis_showscale=False,
                                   yaxis_title="", xaxis_title="Clutch Score")
            st.plotly_chart(fig_bc2, use_container_width=True)

        with st.expander("Full batting clutch data table"):
            _bc_show = bat_c_full[[
                "Player","Bat Avg (Chase)","SR (Pressure)","Win%","NotOut%","50s","Innings","Clutch Score"
            ]].sort_values("Clutch Score", ascending=False).reset_index(drop=True)
            _styled_table(_bc_show,
                          num_cols=["Bat Avg (Chase)","SR (Pressure)","Win%","NotOut%","50s","Innings","Clutch Score"],
                          highlight_col="Clutch Score")

        with st.expander("Analyst Notes — understanding the results"):
            st.markdown("""
**David Miller at #1 ("Killer Miller", Batting Avg ~96, 55% not-out rate)** — cricket-correct.
Miller bats at #5-6, comes in during difficult chases, and consistently finishes not-out.
His 55% not-out rate is the highest of any qualified batter — meaning he literally carries the bat
to victory in more than half his qualifying innings. The batting average formula (runs/dismissals)
correctly rewards this: a 70* to win a chase counts far more than a 70 and getting out.

**Virat Kohli at #5 (Batting Avg ~68, 30 pressure fifties, 59 innings)** — the volume king.
Kohli has the most qualifying innings and the most pressure fifties of any batter in IPL history
in competitive chases. His 53% win rate (higher than Rohit's 44%) confirms he genuinely contributes
to wins, not just padding stats in losses. He IS the chase master — the metric confirms it.

**AB de Villiers at #3 (Batting Avg ~78, 54% win rate, 42% not-out)** — the complete finisher.
High batting average, explosive SR (152), and strong win rate. ABD exemplifies the clutch archetype.

**Rohit Sharma at #17 (Batting Avg ~55)** — correctly below Kohli. MI's team quality inflated
Rohit's reputation in close matches; this individual metric isolates his personal batting contribution.

**MS Dhoni at #11 (Batting Avg ~62, 40% not-out rate)** — the tactical finisher.
40% not-out rate is elite (he wins chases with the last ball). Lower SR (~140) reflects his
"build early, attack at the death" strategy — correctly reflected by the metric.

**Rahul Jadeja at #8 (Batting Avg ~73, 58% not-out rate, 27% win rate)** — a surprise.
Very high not-out rate inflates batting average, but 27% win rate suggests he often comes in
when the chase is already nearly won — not rescuing a genuine crisis. "Mop-up" role.

**Why batting average (not runs per innings)?** Staying unbeaten to finish a chase is the
ultimate individual clutch act. A player who scores 70* to win contributes 70 runs with 0
dismissals. Batting average = 70+/0 = reflects their true impact. Runs-per-innings gives
the same 70, but misses whether they finished the job or got out shortly after.
""")

    st.markdown("---")

    # ── BOWLING CLUTCH ───────────────────────────────────────────────────────
    st.markdown(f"### Bowling Clutch Score — {_cs_label}")
    st.caption(
        "**Clutch Score** = (70% Eco Diff + 30% WktRate Diff × 6) × Win Rate Factor × Sample Shrinkage.  "
        "Economy always rises in death overs — this shows relative improvement vs each bowler's own baseline.  "
        "Win Rate Factor rewards bowlers whose death-over performances actually helped the team win."
    )

    if bowl_c_use.empty:
        st.warning("Not enough pressure-situation data for bowling clutch in this season.")
    else:
        bowl_c_full = bowl_c_use.copy()
        bowl_c_full["Player"]          = bowl_c_full["bowler"].map(_bowl_nm).fillna(bowl_c_full["bowler"])
        bowl_c_full["Clutch Score"]    = bowl_c_full["clutch_score_bowl"].round(2)
        bowl_c_full["Raw Diff"]        = bowl_c_full["clutch_differential_bowl"].round(2)
        bowl_c_full["Eco Normal"]      = bowl_c_full["eco_normal"].round(2)
        bowl_c_full["Eco Pressure"]    = bowl_c_full["eco_pressure"].round(2)
        bowl_c_full["Eco Diff"]        = bowl_c_full["eco_diff"].round(2)
        bowl_c_full["Wkts/Over (P)"]   = bowl_c_full["wpo_pressure"].round(3)
        bowl_c_full["Wkts/Over (N)"]   = bowl_c_full["wpo_normal"].round(3)
        bowl_c_full["Win% (Pressure)"] = (bowl_c_full["pressure_win_rate"] * 100).round(0).astype(int)
        bowl_c_full["Death Matches"]   = bowl_c_full["innings_pressure"].astype(int)
        bowl_c_full["Death Balls"]     = bowl_c_full["balls_pressure"].astype(int)

        _wc_top = bowl_c_full.sort_values("Clutch Score", ascending=False).head(15)
        _wc_bot = bowl_c_full.sort_values("Clutch Score", ascending=True).head(10)

        col3, col4 = st.columns([3, 2])
        with col3:
            st.markdown("**Top 15 — Death-Over Clutch Bowlers**")
            fig_bowlc = px.bar(
                _wc_top.sort_values("Clutch Score"),
                y="Player", x="Clutch Score", orientation="h",
                color="Clutch Score",
                color_continuous_scale=["#ffffbf","#1a9850"],
                color_continuous_midpoint=0,
                hover_data={"Eco Normal": True, "Eco Pressure": True,
                            "Wkts/Over (P)": True, "Win% (Pressure)": True, "Death Matches": True},
                labels={"Clutch Score": "Clutch Score"},
            )
            fig_bowlc.add_vline(x=0, line_color="black", line_width=1)
            fig_bowlc.update_layout(height=480, template="plotly_white", coloraxis_showscale=False,
                                     yaxis_title="", xaxis_title="Clutch Score (Eco+Wkt Diff × Win Rate × Sample Weight)")
            st.plotly_chart(fig_bowlc, use_container_width=True)

        with col4:
            st.markdown("**Bottom 10 — Expensive in Crunch**")
            fig_bowlc2 = px.bar(
                _wc_bot.sort_values("Clutch Score", ascending=False),
                y="Player", x="Clutch Score", orientation="h",
                color="Clutch Score",
                color_continuous_scale=["#d73027","#ffffbf"],
                color_continuous_midpoint=0,
                hover_data={"Eco Normal": True, "Eco Pressure": True,
                            "Win% (Pressure)": True, "Death Matches": True},
                labels={"Clutch Score": "Clutch Score"},
            )
            fig_bowlc2.add_vline(x=0, line_color="black", line_width=1)
            fig_bowlc2.update_layout(height=380, template="plotly_white", coloraxis_showscale=False,
                                      yaxis_title="", xaxis_title="Clutch Score")
            st.plotly_chart(fig_bowlc2, use_container_width=True)

        with st.expander("Full bowling clutch data table"):
            _wc_show = bowl_c_full[[
                "Player","Eco Normal","Eco Pressure","Eco Diff",
                "Wkts/Over (N)","Wkts/Over (P)","Win% (Pressure)","Death Matches","Death Balls","Clutch Score"
            ]].sort_values("Clutch Score", ascending=False).reset_index(drop=True)
            _styled_table(_wc_show,
                          num_cols=["Eco Normal","Eco Pressure","Eco Diff",
                                    "Wkts/Over (N)","Wkts/Over (P)","Win% (Pressure)",
                                    "Death Matches","Death Balls","Clutch Score"],
                          highlight_col="Clutch Score")

        with st.expander("Analyst Notes — understanding surprising results"):
            st.markdown("""
**Jasprit Bumrah scores near-zero (−0.07, 31 death-match appearances)** — this is not a weakness.
Bumrah's economy in *normal* overs is already 7.12 (exceptional). His economy in death close-match
overs rises to 7.78 — the smallest deterioration of any qualified bowler. His wicket rate also
rises from 0.318 to 0.537 per over under pressure. The composite nearly cancels to zero because
**he is already bowling at near-maximum effort in every situation** — there is no extra gear to show.

**Bhuvneshwar Kumar at the bottom** (economy 7.38 → 9.71, 45 close-match death appearances) —
this is a well-known cricket fact. Bhuvneshwar is an exceptional swing bowler in powerplay overs
(overs 1–6) but has always struggled with slower balls and yorkers in death overs against attacking
hitters. The metric with 45 appearances is extremely reliable.

**Economy always rises in death overs** — all bowlers show higher economy under pressure. The metric
compares each bowler *to their own baseline*. A bowler at Eco 8.69 in death close matches (Arshdeep)
is bowling at an exceptional absolute level, even though it is slightly above his 8.90 normal economy.
""")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARISON TOOL
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Player Comparison Tool")
    st.caption("Select 2–3 batters to compare across career stats, Impact Score, and Clutch Score.")

    compare_players = st.multiselect(
        "Select players to compare (2–3)",
        options=all_batters, format_func=_fmt,
        default=["V Kohli", "RG Sharma", "DA Warner"] if all(
            p in all_batters for p in ["V Kohli","RG Sharma","DA Warner"]) else all_batters[:3],
        max_selections=3,
    )

    # Metric definitions: (column label, "higher is better"?, raw-value formatter)
    _RADAR_METRICS = ["SR", "Avg", "Boundary%", "Impact Score", "Clutch Score"]
    _METRIC_HELP = {
        "SR": "Career strike rate (runs per 100 balls)",
        "Avg": "Career batting average (runs per dismissal)",
        "Boundary%": "% of runs scored in 4s and 6s",
        "Impact Score": "Dream11-style per-match impact (career, shrinkage applied)",
        "Clutch Score": "Pressure performance × win rate × sample weight",
    }

    if len(compare_players) >= 2:
        # ── Build raw comparison table ──────────────────────────────────────
        rows = []
        for pl in compare_players:
            b_row  = bat[bat["batter"] == pl]
            i_row  = imp[imp["player"] == pl]
            bc_row = bat_c[bat_c["batter"] == pl]
            row = {"Player": name_map.get(pl, pl)}
            if not b_row.empty:
                r = b_row.iloc[0]
                row.update({"Runs": int(r["total_runs"]), "Innings": int(r["innings"]),
                            "SR": round(float(r["strike_rate"]), 1),
                            "Avg": round(float(r["batting_avg"]), 1),
                            "Boundary%": round(float(r["boundary_pct"]), 1)})
            row["Impact Score"] = round(float(i_row.iloc[0]["batting_impact_score"]), 1) if not i_row.empty else float("nan")
            row["Clutch Score"] = round(float(bc_row.iloc[0]["clutch_score_bat"]), 1) if not bc_row.empty else float("nan")
            rows.append(row)

        comp_df = pd.DataFrame(rows)

        # ── Head-to-head table with best-value highlighting ─────────────────
        st.markdown("#### Head-to-Head Stats")
        st.caption("Bold = best value in each row among the selected players.")

        _disp_cols = ["Runs", "Innings", "SR", "Avg", "Boundary%", "Impact Score", "Clutch Score"]
        # Build a transposed view: metrics as rows, players as columns (easier to scan)
        t_rows = []
        for metric in _disp_cols:
            vals = comp_df.set_index("Player")[metric]
            best_player = vals.idxmax() if vals.notna().any() else None
            cells = []
            for pl_name in comp_df["Player"]:
                v = vals[pl_name]
                if pd.isna(v):
                    cells.append("—")
                else:
                    txt = f"{int(v):,}" if metric in ("Runs", "Innings") else f"{v:,.1f}"
                    cells.append(f"[best] {txt}" if pl_name == best_player else txt)
            t_rows.append({"Metric": metric, **dict(zip(comp_df["Player"], cells))})

        t_df = pd.DataFrame(t_rows)
        st.dataframe(t_df, use_container_width=True, hide_index=True)

        with st.expander("What do these metrics mean?"):
            for mname, mhelp in _METRIC_HELP.items():
                st.markdown(f"- **{mname}** — {mhelp}")

        # ── Radar chart (profile shape) ─────────────────────────────────────
        st.markdown("#### Profile Radar")
        st.caption(
            "Each spoke is scaled so the **best player on that metric reaches the outer edge (100)** "
            "and others show proportionally. Hover any point to see the real value. "
            "A bigger filled shape = a more complete all-round batter among those selected."
        )

        # Normalise: value / max among selected × 100 (clamp negatives to 0 so the
        # shape stays readable). Clutch Score can be negative → a player below 0 shows
        # nothing on that spoke, which correctly reads as "no positive pressure edge".
        norm = comp_df.set_index("Player")[_RADAR_METRICS].copy()
        raw  = norm.copy()
        for col in _RADAR_METRICS:
            pos_max = max(norm[col].max(), 0.01)
            norm[col] = (norm[col].clip(lower=0) / pos_max * 100).round(1)

        # Semi-transparent fills (rgba) so overlapping shapes stay legible
        colors      = ["#1565C0", "#E64A19", "#2E7D32"]
        fill_colors = ["rgba(21,101,192,0.18)", "rgba(230,74,25,0.18)", "rgba(46,125,50,0.18)"]
        fig_radar = go.Figure()
        for i, pl_name in enumerate(norm.index):
            n_vals = norm.loc[pl_name].tolist()
            r_vals = raw.loc[pl_name].tolist()
            c = colors[i % len(colors)]
            fig_radar.add_trace(go.Scatterpolar(
                r=n_vals + [n_vals[0]],
                theta=_RADAR_METRICS + [_RADAR_METRICS[0]],
                name=pl_name, mode="lines+markers",
                line=dict(color=c, width=2.5),
                marker=dict(size=7, color=c),
                fill="toself", fillcolor=fill_colors[i % len(fill_colors)],
                customdata=[[v] for v in (r_vals + [r_vals[0]])],
                hovertemplate="<b>%{theta}</b><br>" + pl_name +
                              ": %{customdata[0]:.1f}<br>(scaled: %{r:.0f}/100)<extra></extra>",
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
            height=480, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select at least 2 players to compare.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — DEATH OVERS (16–20)
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section_header("bolt", "Death Overs Intelligence — Overs 16–20")

    # ── Season filter ────────────────────────────────────────────────────────
    all_seasons_do = sorted(d["season_id"].dropna().unique().astype(int).tolist())
    _fc1, _fc2, _fc3 = st.columns([2, 2, 5])
    with _fc1:
        do_season_from = st.selectbox("From Season", all_seasons_do, index=0, key="do_from")
    with _fc2:
        do_season_to   = st.selectbox("To Season", all_seasons_do, index=len(all_seasons_do)-1, key="do_to")
    with _fc3:
        st.markdown(
            '<p style="color:#546E7A;font-size:0.76rem;margin-top:30px;">'
            'Season range applies to all charts below.</p>',
            unsafe_allow_html=True,
        )

    # ── Base filter ──────────────────────────────────────────────────────────
    d_death = d[
        (d["over_phase"] == "death") &
        (~d["is_super_over"]) &
        (d["is_legal_ball"]) &
        (d["season_id"].between(do_season_from, do_season_to))
    ].copy()
    d_death["is_boundary"] = d_death["batter_runs"].isin([4, 6])
    d_death["is_six"]      = d_death["batter_runs"] == 6
    d_death["is_dot"]      = d_death["batter_runs"] == 0

    # ── KPI row ──────────────────────────────────────────────────────────────
    _tb  = len(d_death)
    _tr  = int(d_death["batter_runs"].sum())
    _rr  = round(_tr / max(_tb, 1) * 6, 2)
    _sp  = round(d_death["is_six"].mean() * 100, 1)
    _dp  = round(d_death["is_dot"].mean() * 100, 1)
    _tw  = int(d_death["is_wicket"].sum())

    kpi_grid([
        {"label": "Avg Run Rate",   "value": f"{_rr:.2f}",    "icon": "speed"},
        {"label": "Six Rate",       "value": f"{_sp}%",        "icon": "sports_cricket"},
        {"label": "Dot Ball %",     "value": f"{_dp}%",        "icon": "block"},
        {"label": "Total Wickets",  "value": f"{_tw:,}",       "icon": "emoji_events"},
        {"label": "Total Balls",    "value": f"{_tb:,}",       "icon": "fiber_manual_record"},
    ], columns=5)

    # ── Shared chart theme ───────────────────────────────────────────────────
    _LAYOUT = dict(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color="#1A1A2E"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=5, r=65, t=14, b=40),
        xaxis=dict(showgrid=True, gridcolor="#F0F2F5", gridwidth=1,
                   zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )
    _HINT = "color:#546E7A;font-size:0.76rem;margin:2px 0 12px"

    def _hbar(df, x_col, y_col, color, label, hover_extra="", fmt=None, height=400):
        vals = df[x_col]
        txt  = vals.apply(fmt) if fmt else vals.apply(str)
        fig  = go.Figure(go.Bar(
            y=df[y_col], x=vals, orientation="h",
            marker=dict(color=color, opacity=0.9, line=dict(width=0)),
            text=txt, textposition="outside",
            textfont=dict(size=11, color="#1A1A2E"),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x}}{hover_extra}<extra></extra>",
        ))
        layout = dict(_LAYOUT)
        layout["height"] = height
        layout["yaxis"] = dict(autorange="reversed", showgrid=False, tickfont=dict(size=11))
        fig.update_layout(**layout)
        return fig

    # ── Inner sub-tabs ───────────────────────────────────────────────────────
    do_tabs = st.tabs(["Batting", "Bowling", "Teams"])

    # ════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — BATTING
    # ════════════════════════════════════════════════════════════════════════
    with do_tabs[0]:
        min_balls_bat = st.slider(
            "Minimum balls faced", 50, 300, 100, step=25, key="do_bat_balls",
            help="Filter out batters with fewer than this many death-over balls",
        )

        bat_death = (
            d_death.groupby("batter").agg(
                balls      = ("batter_runs", "count"),
                runs       = ("batter_runs", "sum"),
                sixes      = ("is_six",      "sum"),
                boundaries = ("is_boundary", "sum"),
                dots       = ("is_dot",      "sum"),
                wickets    = ("is_wicket",   "sum"),
            ).reset_index()
        )
        bat_death["strike_rate"]  = (bat_death["runs"]       / bat_death["balls"] * 100).round(1)
        bat_death["six_pct"]      = (bat_death["sixes"]      / bat_death["balls"] * 100).round(1)
        bat_death["boundary_pct"] = (bat_death["boundaries"] / bat_death["balls"] * 100).round(1)
        bat_death["dot_pct"]      = (bat_death["dots"]       / bat_death["balls"] * 100).round(1)
        bat_death["player_full"]  = bat_death["batter"].map(name_map).fillna(bat_death["batter"])
        bat_death_f = bat_death[bat_death["balls"] >= min_balls_bat].copy()

        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**Most Runs in Death Overs**")
            st.markdown(f'<p style="{_HINT}">Total runs scored in overs 16–20 across career.</p>',
                        unsafe_allow_html=True)
            _top = bat_death_f.sort_values("runs", ascending=False).head(15)
            fig = _hbar(_top, "runs", "player_full", CHART_COLORS[0],
                        "Runs", "<br>SR: %{customdata}", fmt=lambda v: str(int(v)))
            fig.update_traces(customdata=_top["strike_rate"])
            st.plotly_chart(fig, use_container_width=True)

        with bc2:
            st.markdown("**Highest Strike Rate**")
            st.markdown(f'<p style="{_HINT}">Runs per 100 balls in death overs. Higher = more explosive.</p>',
                        unsafe_allow_html=True)
            _top = bat_death_f.sort_values("strike_rate", ascending=False).head(15)
            fig = _hbar(_top, "strike_rate", "player_full", CHART_COLORS[2],
                        "SR", "<br>Balls: %{customdata}", fmt=lambda v: f"{v:.0f}")
            fig.update_traces(customdata=_top["balls"])
            st.plotly_chart(fig, use_container_width=True)

        bc3, bc4 = st.columns(2)
        with bc3:
            st.markdown("**Most Sixes**")
            st.markdown(f'<p style="{_HINT}">Six-hitters dominating the final 5 overs.</p>',
                        unsafe_allow_html=True)
            _top = bat_death_f.sort_values("sixes", ascending=False).head(15)
            fig = _hbar(_top, "sixes", "player_full", CHART_COLORS[3],
                        "Sixes", "<br>Six rate: %{customdata:.1f}%", fmt=lambda v: str(int(v)))
            fig.update_traces(customdata=_top["six_pct"])
            st.plotly_chart(fig, use_container_width=True)

        with bc4:
            st.markdown("**Lowest Dot Ball %** *(best scoreboard movers)*")
            st.markdown(f'<p style="{_HINT}">Fewest wasted balls — keeps the chase alive every delivery.</p>',
                        unsafe_allow_html=True)
            _top = bat_death_f.sort_values("dot_pct").head(15)
            fig = _hbar(_top, "dot_pct", "player_full", CHART_COLORS[1],
                        "Dot%", "", fmt=lambda v: f"{v:.1f}%")
            fig.update_layout(xaxis=dict(ticksuffix="%", showgrid=True,
                                         gridcolor="#F0F2F5", zeroline=False))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full Batting Stats Table"):
            _tbl = bat_death_f.rename(columns={
                "player_full": "Player", "balls": "Balls", "runs": "Runs",
                "strike_rate": "SR", "sixes": "6s", "boundaries": "4s+6s",
                "six_pct": "Six%", "boundary_pct": "Bdry%", "dot_pct": "Dot%",
            })
            st.dataframe(
                _tbl[["Player","Runs","Balls","SR","6s","4s+6s","Six%","Bdry%","Dot%"]]
                    .sort_values("SR", ascending=False).reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — BOWLING
    # ════════════════════════════════════════════════════════════════════════
    with do_tabs[1]:
        min_overs_bowl = st.slider(
            "Minimum overs bowled", 10, 100, 20, step=5, key="do_bowl_overs",
            help="Filter bowlers with fewer than this many death overs bowled",
        )

        bowl_death = (
            d_death.groupby("bowler").agg(
                balls      = ("batter_runs", "count"),
                runs       = ("batter_runs", "sum"),
                wickets    = ("is_wicket",   "sum"),
                dots       = ("is_dot",      "sum"),
                boundaries = ("is_boundary", "sum"),
            ).reset_index()
        )
        bowl_death["overs"]        = (bowl_death["balls"] / 6).round(1)
        bowl_death["economy"]      = (bowl_death["runs"]  / bowl_death["overs"]).round(2)
        bowl_death["dot_pct"]      = (bowl_death["dots"]  / bowl_death["balls"] * 100).round(1)
        bowl_death["boundary_pct"] = (bowl_death["boundaries"] / bowl_death["balls"] * 100).round(1)
        bowl_death["bowling_sr"]   = (bowl_death["balls"] / bowl_death["wickets"].clip(lower=1)).round(1)
        bowl_death["player_full"]  = bowl_death["bowler"].map(name_map).fillna(bowl_death["bowler"])
        bowl_death_f = bowl_death[bowl_death["overs"] >= min_overs_bowl].copy()

        bwc1, bwc2 = st.columns(2)
        with bwc1:
            st.markdown("**Best Economy Rate**")
            st.markdown(f'<p style="{_HINT}">Runs conceded per over. Lower = harder to score off in death.</p>',
                        unsafe_allow_html=True)
            _top = bowl_death_f.sort_values("economy").head(15)
            fig = _hbar(_top, "economy", "player_full", CHART_COLORS[1],
                        "Economy", "<br>Overs: %{customdata:.1f}", fmt=lambda v: f"{v:.2f}")
            fig.update_traces(customdata=_top["overs"])
            st.plotly_chart(fig, use_container_width=True)

        with bwc2:
            st.markdown("**Most Wickets**")
            st.markdown(f'<p style="{_HINT}">Death-over wicket takers ranked by volume across career.</p>',
                        unsafe_allow_html=True)
            _top = bowl_death_f.sort_values("wickets", ascending=False).head(15)
            fig = _hbar(_top, "wickets", "player_full", CHART_COLORS[2],
                        "Wickets", "<br>Bowl SR: %{customdata}", fmt=lambda v: str(int(v)))
            fig.update_traces(customdata=_top["bowling_sr"])
            st.plotly_chart(fig, use_container_width=True)

        bwc3, bwc4 = st.columns(2)
        with bwc3:
            st.markdown("**Highest Dot Ball %**")
            st.markdown(f'<p style="{_HINT}">More dots = more pressure = less time for the batter to accelerate.</p>',
                        unsafe_allow_html=True)
            _top = bowl_death_f.sort_values("dot_pct", ascending=False).head(15)
            fig = _hbar(_top, "dot_pct", "player_full", CHART_COLORS[4],
                        "Dot%", "<br>Overs: %{customdata:.1f}", fmt=lambda v: f"{v:.1f}%")
            fig.update_traces(customdata=_top["overs"])
            fig.update_layout(xaxis=dict(ticksuffix="%", showgrid=True,
                                          gridcolor="#F0F2F5", zeroline=False))
            st.plotly_chart(fig, use_container_width=True)

        with bwc4:
            st.markdown("**Best Bowling Strike Rate** *(min 10 wkts)*")
            st.markdown(f'<p style="{_HINT}">Balls per wicket — fewer balls needed = more dangerous in death.</p>',
                        unsafe_allow_html=True)
            _top = bowl_death_f[bowl_death_f["wickets"] >= 10].sort_values("bowling_sr").head(15)
            fig = _hbar(_top, "bowling_sr", "player_full", CHART_COLORS[0],
                        "Bowl SR", "<br>Wickets: %{customdata}", fmt=lambda v: f"{v:.1f}")
            fig.update_traces(customdata=_top["wickets"])
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Full Bowling Stats Table"):
            _tbl = bowl_death_f.rename(columns={
                "player_full": "Player", "overs": "Overs", "runs": "Runs",
                "wickets": "Wkts", "economy": "Economy", "dot_pct": "Dot%",
                "boundary_pct": "Bdry%", "bowling_sr": "Bowl SR",
            })
            st.dataframe(
                _tbl[["Player","Overs","Runs","Wkts","Economy","Dot%","Bdry%","Bowl SR"]]
                    .sort_values("Economy").reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — TEAMS
    # ════════════════════════════════════════════════════════════════════════
    with do_tabs[2]:
        from src.module_2_teams import id_to_name as _id2name_fn
        from src.data_loader import load_teams
        _teams   = load_teams()
        _id2name = _id2name_fn(_teams)

        team_bat_death = (
            d_death.groupby("team_batting").agg(
                balls = ("batter_runs", "count"),
                runs  = ("batter_runs", "sum"),
                sixes = ("is_six",      "sum"),
                dots  = ("is_dot",      "sum"),
            ).reset_index()
        )
        team_bat_death["run_rate"]  = (team_bat_death["runs"] / team_bat_death["balls"] * 6).round(2)
        team_bat_death["six_rate"]  = (team_bat_death["sixes"] / team_bat_death["balls"] * 100).round(1)
        team_bat_death["dot_pct"]   = (team_bat_death["dots"]  / team_bat_death["balls"] * 100).round(1)
        team_bat_death["team_name"] = team_bat_death["team_batting"].map(_id2name)
        team_bat_death = team_bat_death[team_bat_death["team_name"].notna()]

        team_bowl_death = (
            d_death.groupby("team_bowling").agg(
                balls   = ("batter_runs", "count"),
                runs    = ("batter_runs", "sum"),
                wickets = ("is_wicket",   "sum"),
                dots    = ("is_dot",      "sum"),
            ).reset_index()
        )
        team_bowl_death["economy"]   = (team_bowl_death["runs"] / team_bowl_death["balls"] * 6).round(2)
        team_bowl_death["dot_pct"]   = (team_bowl_death["dots"] / team_bowl_death["balls"] * 100).round(1)
        team_bowl_death["team_name"] = team_bowl_death["team_bowling"].map(_id2name)
        team_bowl_death = team_bowl_death[team_bowl_death["team_name"].notna()]

        st.markdown(
            f'<p style="{_HINT}">Teams in the top-right quadrant (high batting RR, low bowling economy) '
            'dominate the death phase both ways.</p>',
            unsafe_allow_html=True,
        )

        # ── Scatter ──────────────────────────────────────────────────────
        _merged = pd.merge(
            team_bat_death[["team_name","run_rate","six_rate"]],
            team_bowl_death[["team_name","economy","dot_pct"]],
            on="team_name",
        )
        _avg_rr_t  = _merged["run_rate"].mean()
        _avg_eco_t = _merged["economy"].mean()

        _scatter_colors = (CHART_COLORS * 4)[:len(_merged)]
        fig_scatter = go.Figure()
        for i, (_, row) in enumerate(_merged.iterrows()):
            _short = row["team_name"].split()[-1]
            fig_scatter.add_trace(go.Scatter(
                x=[row["economy"]], y=[row["run_rate"]],
                mode="markers+text",
                text=[_short], textposition="top center",
                textfont=dict(size=10, color="#1A1A2E"),
                marker=dict(size=16, color=_scatter_colors[i],
                            opacity=0.9, line=dict(width=1.5, color="white")),
                hovertemplate=(
                    f"<b>{row['team_name']}</b><br>"
                    f"Batting RR: {row['run_rate']:.2f}<br>"
                    f"Bowl Economy: {row['economy']:.2f}<br>"
                    f"Six Rate: {row['six_rate']:.1f}%<br>"
                    f"Dot%: {row['dot_pct']:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))
        fig_scatter.add_hline(y=_avg_rr_t, line_dash="dash", line_color="#CBD5E1", line_width=1)
        fig_scatter.add_vline(x=_avg_eco_t, line_dash="dash", line_color="#CBD5E1", line_width=1)
        fig_scatter.update_layout(
            height=420, template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12, color="#1A1A2E"),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis=dict(title="Bowling Economy — runs conceded per over (lower = better ←)",
                       showgrid=True, gridcolor="#F0F2F5", zeroline=False),
            yaxis=dict(title="Batting Run Rate — runs per over (higher = better ↑)",
                       showgrid=True, gridcolor="#F0F2F5", zeroline=False),
            margin=dict(l=10, r=20, t=24, b=50),
            annotations=[
                dict(x=_merged["economy"].min()+0.05, y=_merged["run_rate"].max()-0.02,
                     text="Best quadrant (high bat, low bowl eco)",
                     showarrow=False, font=dict(size=10, color=PALETTE["success"]),
                     xanchor="left"),
                dict(x=_avg_eco_t+0.03, y=_merged["run_rate"].min()+0.01,
                     text="league avg eco", showarrow=False,
                     font=dict(size=9, color="#94A3B8"), xanchor="left"),
                dict(x=_merged["economy"].min()+0.03, y=_avg_rr_t+0.01,
                     text="league avg RR", showarrow=False,
                     font=dict(size=9, color="#94A3B8"), xanchor="left"),
            ],
        )
        st.markdown("**Batting RR vs Bowling Economy — all-time death overs**")
        st.plotly_chart(fig_scatter, use_container_width=True)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Batting Run Rate — Death Overs**")
            st.markdown(f'<p style="{_HINT}">Runs scored per over in overs 16–20.</p>',
                        unsafe_allow_html=True)
            _top = team_bat_death.sort_values("run_rate", ascending=False)
            fig = _hbar(_top, "run_rate", "team_name", CHART_COLORS[0],
                        "Run Rate", "<br>Six Rate: %{customdata:.1f}%",
                        fmt=lambda v: f"{v:.2f}", height=460)
            fig.update_traces(customdata=_top["six_rate"])
            st.plotly_chart(fig, use_container_width=True)

        with tc2:
            st.markdown("**Bowling Economy — Death Overs**")
            st.markdown(f'<p style="{_HINT}">Runs conceded per over in overs 16–20.</p>',
                        unsafe_allow_html=True)
            _top = team_bowl_death.sort_values("economy")
            fig = _hbar(_top, "economy", "team_name", CHART_COLORS[1],
                        "Economy", "<br>Dot%: %{customdata:.1f}%",
                        fmt=lambda v: f"{v:.2f}", height=460)
            fig.update_traces(customdata=_top["dot_pct"])
            st.plotly_chart(fig, use_container_width=True)
