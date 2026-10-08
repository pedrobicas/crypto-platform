"""Identidade visual: paleta, template do Plotly e CSS complementar."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG = "#121626"
PANEL = "#1A2036"
GRID = "#262D47"
TEXT = "#E8EAF3"
MUTED = "#8F96B3"
GOLD = "#E3B341"
UP = "#34C38F"
DOWN = "#F2665E"
PERI = "#8C9EFF"
LILAC = "#C08CF0"
CYAN = "#5CC8E0"
SAND = "#D9A87E"
SLATE = "#9FB0C8"

SERIES = [GOLD, PERI, UP, LILAC, CYAN, DOWN, SAND, SLATE]
FONT = "IBM Plex Sans, Segoe UI, Roboto, sans-serif"
HEADING_FONT = "Bricolage Grotesque, IBM Plex Sans, sans-serif"

TEMPLATE_NAME = "cripto"


def _register_template() -> None:
    if TEMPLATE_NAME in pio.templates:
        return
    axis = dict(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickcolor=GRID,
        tickfont=dict(color=MUTED, size=11),
        title=dict(font=dict(color=MUTED, size=12)),
        showspikes=True,
        spikecolor=MUTED,
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
        automargin=True,
    )
    pio.templates[TEMPLATE_NAME] = go.layout.Template(
        layout=dict(
            font=dict(family=FONT, color=TEXT, size=12),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=SERIES,
            separators=",.",
            hovermode="x unified",
            hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(family=FONT, color=TEXT, size=12)),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=MUTED, size=11), bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(l=10, r=10, t=36, b=10),
            title=dict(font=dict(family=HEADING_FONT, size=15, color=TEXT), x=0, xanchor="left"),
        )
    )


_register_template()


CSS = f"""
<style>
[data-testid="stMetricValue"], .num {{ font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 2.2rem; }}
.coin-head {{ display:flex; flex-wrap:wrap; align-items:flex-end; gap:0.4rem 2.2rem; margin: 0.2rem 0 0.6rem 0; }}
.coin-head .name {{ font-family:{HEADING_FONT}; font-size:1.05rem; color:{MUTED}; font-weight:500; margin:0; }}
.coin-head .name b {{ color:{TEXT}; font-weight:700; font-size:1.35rem; margin-right:.45rem; }}
.coin-head .price {{ font-family:{HEADING_FONT}; font-size:clamp(2.1rem, 4.2vw, 3.3rem); font-weight:700; line-height:1;
  letter-spacing:-0.02em; color:{TEXT}; font-variant-numeric: tabular-nums; }}
.coin-head .chg {{ font-size:1.05rem; font-weight:600; font-variant-numeric: tabular-nums; }}
.coin-head .chg.up {{ color:{UP}; }} .coin-head .chg.down {{ color:{DOWN}; }}
.coin-head .meta {{ color:{MUTED}; font-size:0.85rem; }}
.verdict {{ font-family:{HEADING_FONT}; font-size:1.35rem; font-weight:700; margin:0; }}
.verdict.up {{ color:{UP}; }} .verdict.down {{ color:{DOWN}; }} .verdict.flat {{ color:{GOLD}; }}
.muted {{ color:{MUTED}; font-size:0.88rem; }}
.vote-row {{ display:flex; gap:1.1rem; font-size:0.9rem; color:{MUTED}; margin-top:.2rem; }}
.vote-row b {{ font-variant-numeric: tabular-nums; }}
.vote-row .u b {{ color:{UP}; }} .vote-row .d b {{ color:{DOWN}; }} .vote-row .n b {{ color:{TEXT}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def change_color(value: float) -> str:
    return UP if value >= 0 else DOWN
