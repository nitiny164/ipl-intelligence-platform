"""
Module 5 — Data Explorer
Self-serve analytics: one filter bar, five views, full CSV download.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_loader import load_matches, load_deliveries, load_players, load_teams, processed_exists
from app.style import inject_styles, page_header, section_header, kpi_grid, PALETTE, CHART_COLORS
from src.module_5_explorer import (
    apply_match_filters, apply_delivery_filters,
    match_summary_view, batter_scorecard, bowler_scorecard,
    venue_scorecard, phase_breakdown, compare_batters, compare_bowlers,
    get_player_records,
)

st.set_page_config(page_title="Data Explorer | IPL Intelligence", layout="wide")
inject_styles()

if not processed_exists():
    st.error("Run the data pipeline first (`python -m src.module_0_foundation`).")
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def _load():
    return load_matches(), load_deliveries(), load_players(), load_teams()

matches, deliveries, players, teams = _load()
# full unfiltered copies used only by records engine (all-time superlatives)
_all_deliveries = deliveries
_all_matches    = matches
id2name  = dict(zip(teams["team_id"], teams["team_name"]))
name2id  = {v: k for k, v in id2name.items()}
pid2name = dict(zip(players["player_id"], players["player_name"]))
pname2id = {v: k for k, v in pid2name.items()}

page_header("manage_search", "Data Explorer",
            "Set your filters once — every view updates instantly. Download any table as CSV.")

# ════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FILTER BAR (sidebar)
# ════════════════════════════════════════════════════════════════════════════
st.sidebar.header("Filters")

all_seasons = sorted(matches["season"].dropna().unique().tolist())
sel_seasons = st.sidebar.multiselect("Season(s)", all_seasons, default=all_seasons[-5:])

all_teams_sorted = sorted(id2name.values())
sel_team_names = st.sidebar.multiselect("Team(s)", all_teams_sorted)
sel_team_ids   = [name2id[n] for n in sel_team_names if n in name2id]

all_venues = sorted(matches["venue"].dropna().unique().tolist())
sel_venues = st.sidebar.multiselect("Venue(s)", all_venues)

all_phases = ["powerplay", "middle", "death"]
sel_phases = st.sidebar.multiselect("Phase(s)", all_phases, default=all_phases)

sel_innings = st.sidebar.multiselect("Innings", [1, 2], default=[1, 2])

min_balls_bat = st.sidebar.slider("Min balls faced (batter filter)", 10, 200, 50)
min_balls_bow = st.sidebar.slider("Min balls bowled (bowler filter)", 20, 300, 60)

# Apply filters
filtered_matches = apply_match_filters(
    matches,
    seasons=sel_seasons if sel_seasons else None,
    team_ids=sel_team_ids if sel_team_ids else None,
    venues=sel_venues if sel_venues else None,
)
filtered_deliveries = apply_delivery_filters(
    deliveries,
    match_ids=filtered_matches["match_id"].tolist(),
    phases=sel_phases if sel_phases else None,
    innings=sel_innings if sel_innings else None,
)

# ── KPI strip ────────────────────────────────────────────────────────────────
n_matches  = int(filtered_matches["is_completed"].sum())
n_balls    = len(filtered_deliveries)
n_seasons  = len(filtered_matches["season"].unique())
n_venues   = filtered_matches["venue"].nunique()

kpi_grid([
    {"icon": "sports_cricket", "label": "Matches in selection", "value": f"{n_matches:,}"},
    {"icon": "fiber_manual_record", "label": "Total balls",       "value": f"{n_balls:,}"},
    {"icon": "calendar_today",     "label": "Seasons covered",    "value": str(n_seasons)},
    {"icon": "location_on",        "label": "Venues covered",     "value": str(n_venues)},
], columns=4)

st.markdown('<hr style="border:none;border-top:1px solid #E3E8EF;margin:0 0 1rem">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABBED VIEWS
# ════════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["Matches", "Batters", "Bowlers", "Venues", "Phase Analysis", "Compare"])

_CHART_CFG = dict(template="plotly_white", font=dict(family="Inter, sans-serif", size=11))


# ── Tab 1: Match Summary ─────────────────────────────────────────────────────
with tabs[0]:
    section_header("table_view", f"Match Summary — {len(filtered_matches):,} matches")

    msv = match_summary_view(filtered_matches, id2name)
    st.dataframe(msv, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", msv.to_csv(index=False), "ipl_matches.csv", "text/csv")

    col_hist, col_toss = st.columns(2)

    with col_hist:
        section_header("bar_chart", "1st Innings Score Distribution")
        if len(filtered_matches) >= 2 and "first_innings_total" in filtered_matches.columns:
            fig_hist = px.histogram(
                filtered_matches[filtered_matches["is_completed"]],
                x="first_innings_total", nbins=30,
                color_discrete_sequence=[PALETTE["primary"]],
                labels={"first_innings_total": "1st Innings Score", "count": "Matches"},
            )
            fig_hist.update_layout(**_CHART_CFG, height=320,
                                   margin=dict(l=10, r=10, t=20, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_toss:
        section_header("flip", "Toss Impact — Did Winning the Toss Help?")
        comp = filtered_matches[filtered_matches["is_completed"]].copy()
        if len(comp) >= 5 and "toss_winner" in comp.columns and "match_winner" in comp.columns:
            comp["toss_won_match"] = comp["toss_winner"] == comp["match_winner"]
            toss_win_pct = round(comp["toss_won_match"].mean() * 100, 1)
            toss_lose_pct = round(100 - toss_win_pct, 1)

            # Toss decision breakdown
            dec = comp.groupby("toss_decision")["toss_won_match"].agg(
                wins="sum", total="count"
            ).reset_index()
            dec["Win %"] = (dec["wins"] / dec["total"] * 100).round(1)
            dec["Decision"] = dec["toss_decision"].str.capitalize()

            fig_toss = go.Figure()
            colors = [PALETTE["primary"], PALETTE["success"]]
            for i, row in dec.iterrows():
                fig_toss.add_trace(go.Bar(
                    x=[row["Decision"]],
                    y=[row["Win %"]],
                    name=row["Decision"],
                    marker_color=colors[i % len(colors)],
                    text=f"{row['Win %']}%",
                    textposition="outside",
                    hovertemplate=f"<b>{row['Decision']}</b><br>Win %: {row['Win %']}%<br>Sample: {row['total']} matches<extra></extra>",
                ))
            fig_toss.add_hline(y=50, line_dash="dot", line_color="#90A4AE",
                               annotation_text="50% — no advantage", annotation_position="top right",
                               annotation_font=dict(size=10, color="#546E7A"))
            fig_toss.update_layout(
                **_CHART_CFG, height=320, showlegend=False,
                yaxis=dict(title="Win % after winning toss", range=[0, 75]),
                xaxis=dict(title="Toss decision"),
                margin=dict(l=10, r=10, t=20, b=20),
            )
            st.plotly_chart(fig_toss, use_container_width=True)
            st.caption(f"Overall: toss winner won the match {toss_win_pct}% of the time in this selection.")
        else:
            st.info("Not enough completed matches to analyse toss impact.")


# ── Tab 2: Batters ───────────────────────────────────────────────────────────
with tabs[1]:
    section_header("person", "Batter Scorecard")
    if len(filtered_deliveries) == 0:
        st.info("No deliveries match the current filters.")
    else:
        bat_df = batter_scorecard(filtered_deliveries, players, min_balls=min_balls_bat)
        st.dataframe(bat_df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", bat_df.to_csv(index=False), "ipl_batters.csv", "text/csv")
        st.caption("50s = innings of 50–99 runs · 100s = innings of 100+ runs · Boundary % = (4s+6s) per ball faced")

        col1, col2 = st.columns(2)
        with col1:
            section_header("bar_chart", "Top 20 by Runs")
            top20 = bat_df.head(20)
            fig_bat = px.bar(
                top20, x="Batter", y="Runs",
                color="Strike Rate", color_continuous_scale="Blues",
                text="Runs",
            )
            fig_bat.update_traces(textposition="outside", textfont=dict(size=9))
            fig_bat.update_layout(
                **_CHART_CFG, height=380, coloraxis_showscale=False,
                xaxis=dict(tickangle=-40), margin=dict(l=10, r=10, t=10, b=80),
                yaxis=dict(title="Runs"),
            )
            st.plotly_chart(fig_bat, use_container_width=True)
            st.caption("Colour = Strike Rate (darker blue = higher SR)")

        with col2:
            section_header("scatter_plot", "Average vs Strike Rate")
            fig_sr = px.scatter(
                bat_df[bat_df["Innings"] >= 3],
                x="Average", y="Strike Rate",
                size="Balls", hover_name="Batter",
                color="Innings",
                color_continuous_scale="Blues",
                labels={"Innings": "Innings played"},
            )
            fig_sr.update_layout(
                **_CHART_CFG, height=380, margin=dict(l=10, r=10, t=10, b=20),
            )
            st.plotly_chart(fig_sr, use_container_width=True)
            st.caption("Bubble size = balls faced · Top-right = best all-round batters")


# ── Tab 3: Bowlers ───────────────────────────────────────────────────────────
with tabs[2]:
    section_header("sports_handball", "Bowler Scorecard")
    if len(filtered_deliveries) == 0:
        st.info("No deliveries match the current filters.")
    else:
        bow_df = bowler_scorecard(filtered_deliveries, players, min_balls=min_balls_bow)
        st.dataframe(bow_df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", bow_df.to_csv(index=False), "ipl_bowlers.csv", "text/csv")
        st.caption("Bowl SR = balls per wicket · Economy = runs per over · Dot % = % of dot balls bowled")

        col1, col2 = st.columns(2)
        with col1:
            section_header("bar_chart", "Top 20 by Wickets")
            top20b = bow_df.head(20)
            fig_wkt = px.bar(
                top20b, x="Bowler", y="Wickets",
                color="Economy", color_continuous_scale="RdYlGn_r",
                text="Wickets",
            )
            fig_wkt.update_traces(textposition="outside", textfont=dict(size=9))
            fig_wkt.update_layout(
                **_CHART_CFG, height=380, coloraxis_showscale=False,
                xaxis=dict(tickangle=-40), margin=dict(l=10, r=10, t=10, b=80),
            )
            st.plotly_chart(fig_wkt, use_container_width=True)
            st.caption("Colour = Economy (green = cheaper, red = expensive)")

        with col2:
            section_header("scatter_plot", "Economy vs Average")
            fig_eco = px.scatter(
                bow_df[bow_df["Wickets"] >= 5],
                x="Economy", y="Average",
                size="Wickets", hover_name="Bowler",
                color="Dot %", color_continuous_scale="Greens",
                labels={"Dot %": "Dot ball %"},
            )
            fig_eco.update_layout(
                **_CHART_CFG, height=380, margin=dict(l=10, r=10, t=10, b=20),
            )
            st.plotly_chart(fig_eco, use_container_width=True)
            st.caption("Bottom-left = elite (low economy, low average) · Bubble size = wickets")


# ── Tab 4: Venues ─────────────────────────────────────────────────────────────
with tabs[3]:
    section_header("location_on", "Venue Scorecard")
    ven_df = venue_scorecard(filtered_matches)
    st.dataframe(ven_df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", ven_df.to_csv(index=False), "ipl_venues.csv", "text/csv")

    if len(ven_df) >= 3:
        section_header("scatter_plot", "Venue Profile — Scoring vs Chase Success")
        fig_ven = px.scatter(
            ven_df[ven_df["Matches"] >= 5],
            x="Avg 1st Inn", y="Chase Win %",
            size="Matches", hover_name="Venue",
            color="Chase Win %", color_continuous_scale="RdYlGn",
        )
        fig_ven.add_vline(x=ven_df["Avg 1st Inn"].mean(), line_dash="dot", line_color="#90A4AE",
                          annotation_text="League avg score",
                          annotation_font=dict(size=10, color="#546E7A"))
        fig_ven.add_hline(y=50, line_dash="dot", line_color="#90A4AE",
                          annotation_text="50% chase rate",
                          annotation_font=dict(size=10, color="#546E7A"))
        fig_ven.update_layout(
            **_CHART_CFG, height=440, margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_ven, use_container_width=True)
        st.caption(
            "Top-right = high-scoring, chase-friendly grounds. "
            "Bottom-left = low-scoring, defend-friendly. "
            "Bubble size = number of matches."
        )


# ── Tab 5: Phase Analysis ─────────────────────────────────────────────────────
with tabs[4]:
    section_header("timeline", "Phase Scoring Breakdown")
    if len(filtered_deliveries) == 0:
        st.info("No deliveries match the current filters.")
    else:
        ph_df = phase_breakdown(filtered_deliveries)
        st.dataframe(ph_df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", ph_df.to_csv(index=False), "ipl_phase.csv", "text/csv")

        phase_colors = {"powerplay": CHART_COLORS[1], "middle": PALETTE["warning"], "death": PALETTE["danger"]}

        c1, c2, c3, c4 = st.columns(4)
        charts = [
            ("Run Rate",    "run_rate",     "Run Rate per Phase",     "Runs per over · Death overs are highest — teams attack in final overs"),
            ("Dot %",       "dot_pct",      "Dot Ball % per Phase",   "Middle overs have most dots — bowlers dominate there"),
            ("Boundary %",  "boundary_pct", "Boundary % per Phase",   "% of balls hit to boundary — death overs see most boundaries"),
            ("Wkts/Over",   "wkt_per_over", "Wickets per Over",       "How many wickets fall per over in each phase"),
        ]
        for col, (ylabel, col_name, title, caption_txt) in zip([c1, c2, c3, c4], charts):
            if col_name not in ph_df.columns:
                continue
            with col:
                fig = go.Figure()
                for _, row in ph_df.iterrows():
                    phase = str(row["Phase"]).lower()
                    fig.add_trace(go.Bar(
                        x=[row["Phase"].capitalize()],
                        y=[row[col_name]],
                        marker_color=phase_colors.get(phase, PALETTE["primary"]),
                        text=[f"{row[col_name]}"],
                        textposition="outside",
                        showlegend=False,
                    ))
                fig.update_layout(
                    **_CHART_CFG, height=280, title=dict(text=ylabel, font=dict(size=12)),
                    yaxis=dict(showgrid=True, gridcolor="#F0F2F5"),
                    xaxis=dict(showgrid=False),
                    margin=dict(l=5, r=5, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(caption_txt)


# ── Tab 6: Compare ────────────────────────────────────────────────────────────
with tabs[5]:
    section_header("compare_arrows", "Entity Comparison")
    st.caption("Compare how batters or bowlers perform across phases under your current filters.")

    compare_mode = st.radio("Compare", ["Batters", "Bowlers"], horizontal=True)

    def _render_records(player_name: str, player_id: int, mode: str, col) -> None:
        """Render record badges for one player inside a column."""
        recs = get_player_records(player_id, _all_deliveries, _all_matches, players, mode=mode)
        with col:
            st.markdown(
                f'<div style="font-size:0.75rem;font-weight:600;letter-spacing:0.07em;'
                f'text-transform:uppercase;color:#546E7A;margin-bottom:8px">'
                f'{player_name} — IPL Records</div>',
                unsafe_allow_html=True,
            )
            if not recs:
                st.markdown(
                    '<div style="font-size:0.82rem;color:#90A4AE;padding:8px 0">'
                    'No all-time top-3 records in this dataset.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for rec in recs:
                    st.markdown(
                        f"""
                        <div style="background:#FFFFFF;border:1px solid #E3E8EF;border-radius:10px;
                                    padding:10px 14px;margin-bottom:8px;
                                    border-left:4px solid {rec['colour']};
                                    box-shadow:0 1px 3px rgba(0,0,0,0.05)">
                          <div style="display:flex;align-items:center;gap:8px">
                            <span class="material-icons-round"
                                  style="font-size:18px;color:{rec['colour']}">{rec['icon']}</span>
                            <div>
                              <div style="font-size:0.78rem;color:#546E7A">{rec['label']}</div>
                              <div style="font-size:1.05rem;font-weight:700;color:#1A1A2E">{rec['value']}</div>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    if compare_mode == "Batters":
        active_bids   = filtered_deliveries["batter_id"].dropna().unique().tolist() if len(filtered_deliveries) > 0 else []
        active_names  = sorted([pid2name[b] for b in active_bids if b in pid2name])
        sel_compare   = st.multiselect("Select 2–3 batters to compare", active_names, max_selections=3)

        if len(sel_compare) >= 2:
            sel_ids = [pname2id[n] for n in sel_compare if n in pname2id]
            cmp_df  = compare_batters(filtered_deliveries, sel_ids, players)

            # ── Records section ──────────────────────────────────────────
            section_header("emoji_events", "IPL Records & Superlatives")
            st.caption("Automatically detected — only shown if the player ranks in the all-time top 3 for that category.")
            rec_cols = st.columns(len(sel_compare))
            for name, pid, col in zip(sel_compare, sel_ids, rec_cols):
                _render_records(name, pid, "batter", col)

            st.markdown('<hr style="border:none;border-top:1px solid #E3E8EF;margin:16px 0">', unsafe_allow_html=True)

            if cmp_df.empty:
                st.info("Insufficient delivery data for selected batters under current filters.")
            else:
                section_header("bar_chart", "Head-to-Head Phase Comparison")
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig_sr = px.bar(cmp_df, x="Phase", y="SR", color="Batter", barmode="group",
                                    title="Strike Rate by Phase",
                                    text="SR", color_discrete_sequence=CHART_COLORS)
                    fig_sr.update_traces(textposition="outside")
                    fig_sr.update_layout(**_CHART_CFG, height=360,
                                         margin=dict(l=10, r=10, t=40, b=20))
                    st.plotly_chart(fig_sr, use_container_width=True)

                with col2:
                    fig_runs = px.bar(cmp_df, x="Phase", y="Runs", color="Batter", barmode="group",
                                      title="Runs by Phase",
                                      text="Runs", color_discrete_sequence=CHART_COLORS)
                    fig_runs.update_traces(textposition="outside")
                    fig_runs.update_layout(**_CHART_CFG, height=360,
                                           margin=dict(l=10, r=10, t=40, b=20))
                    st.plotly_chart(fig_runs, use_container_width=True)
        else:
            st.info("Select at least 2 batters above to activate comparison.")

    else:  # Bowlers
        active_bowids   = filtered_deliveries["bowler_id"].dropna().unique().tolist() if len(filtered_deliveries) > 0 else []
        active_bownames = sorted([pid2name[b] for b in active_bowids if b in pid2name])
        sel_compare     = st.multiselect("Select 2–3 bowlers to compare", active_bownames, max_selections=3)

        if len(sel_compare) >= 2:
            sel_ids = [pname2id[n] for n in sel_compare if n in pname2id]
            cmp_df  = compare_bowlers(filtered_deliveries, sel_ids, players)

            # ── Records section ──────────────────────────────────────────
            section_header("emoji_events", "IPL Records & Superlatives")
            st.caption("Automatically detected — only shown if the player ranks in the all-time top 3 for that category.")
            rec_cols = st.columns(len(sel_compare))
            for name, pid, col in zip(sel_compare, sel_ids, rec_cols):
                _render_records(name, pid, "bowler", col)

            st.markdown('<hr style="border:none;border-top:1px solid #E3E8EF;margin:16px 0">', unsafe_allow_html=True)

            if cmp_df.empty:
                st.info("Insufficient delivery data for selected bowlers under current filters.")
            else:
                section_header("bar_chart", "Head-to-Head Phase Comparison")
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig_eco = px.bar(cmp_df, x="Phase", y="Economy", color="Bowler", barmode="group",
                                     title="Economy Rate by Phase",
                                     text="Economy", color_discrete_sequence=CHART_COLORS)
                    fig_eco.update_traces(textposition="outside")
                    fig_eco.update_layout(**_CHART_CFG, height=360,
                                          margin=dict(l=10, r=10, t=40, b=20))
                    st.plotly_chart(fig_eco, use_container_width=True)
                    st.caption("Lower economy = cheaper to bat against = better for the bowler")

                with col2:
                    fig_wkt = px.bar(cmp_df, x="Phase", y="Wickets", color="Bowler", barmode="group",
                                     title="Wickets by Phase",
                                     text="Wickets", color_discrete_sequence=CHART_COLORS)
                    fig_wkt.update_traces(textposition="outside")
                    fig_wkt.update_layout(**_CHART_CFG, height=360,
                                          margin=dict(l=10, r=10, t=40, b=20))
                    st.plotly_chart(fig_wkt, use_container_width=True)
        else:
            st.info("Select at least 2 bowlers above to activate comparison.")
