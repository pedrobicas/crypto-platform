"""Registro das páginas (preenchido pelo app) para permitir links entre elas."""

from __future__ import annotations

import streamlit as st

PAGES: dict = {}


def link(key: str, label: str, icon: str | None = None) -> None:
    page = PAGES.get(key)
    if page is not None:
        st.page_link(page, label=label, icon=icon)
