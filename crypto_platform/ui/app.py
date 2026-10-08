"""Montagem do app: configuração da página, barra lateral e navegação."""

from __future__ import annotations

import importlib

import streamlit as st

from . import nav
from .state import render_sidebar
from .theme import inject_css

PAGE_SPECS = (
    # chave, módulo, título, ícone, url
    ("overview", "overview", "Visão geral", ":material/space_dashboard:", "visao-geral"),
    ("technical", "technical", "Técnica", ":material/candlestick_chart:", "tecnica"),
    ("forecast", "forecast", "Previsão", ":material/insights:", "previsao"),
    ("risk", "risk", "Risco", ":material/shield:", "risco"),
    ("simulator", "simulator", "Simulador", ":material/casino:", "simulador"),
    ("backtest", "backtest", "Estratégias", ":material/query_stats:", "estrategias"),
    ("portfolio", "portfolio", "Portfólio", ":material/donut_large:", "portfolio"),
    ("about", "about", "Metodologia", ":material/menu_book:", "metodologia"),
)


def _page_runner(module_name: str):
    def run() -> None:
        module = importlib.import_module(f"crypto_platform.ui.pages.{module_name}")
        module.render()

    run.__name__ = f"page_{module_name}"
    return run


def run() -> None:
    st.set_page_config(
        page_title="Crypto Analysis & Forecast",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    inject_css()

    pages = []
    for i, (key, module, title, icon, url) in enumerate(PAGE_SPECS):
        page = st.Page(_page_runner(module), title=title, icon=icon, url_path=url, default=(i == 0))
        nav.PAGES[key] = page
        pages.append(page)

    current = st.navigation(pages, position="top")
    st.session_state["settings"] = render_sidebar()
    current.run()
