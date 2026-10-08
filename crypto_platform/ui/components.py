"""Componentes reutilizáveis entre páginas."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..data_collector import MarketData
from ..formatting import fmt_money, fmt_number, fmt_pct, fmt_price

PLOTLY_CONFIG = {"displaylogo": False, "locale": "pt-BR", "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def plot(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def local_time(dt) -> str:
    """Data/hora no fuso do navegador do usuário (quando disponível)."""
    try:
        from zoneinfo import ZoneInfo

        tz_name = getattr(st.context, "timezone", None)
        if tz_name:
            return dt.astimezone(ZoneInfo(tz_name)).strftime("%d/%m %H:%M")
    except Exception:
        pass
    return dt.strftime("%d/%m %H:%M UTC")


def pct_change(close: pd.Series, days: int) -> float:
    if len(close) <= days:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)


def coin_header(md: MarketData, snapshot: pd.DataFrame | None = None) -> None:
    """Cabeçalho com nome, preço atual e variação — repetido em todas as páginas."""
    close = md.close
    price = float(close.iloc[-1])
    change, change_label = pct_change(close, 1), "no dia"
    if snapshot is not None and not snapshot.empty and md.coin.coingecko_id in snapshot.index:
        snap = snapshot.loc[md.coin.coingecko_id]
        live = snap.get("price_change_percentage_24h")
        if live is not None and np.isfinite(live):
            change, change_label = float(live), "em 24 h"
    direction = "up" if change >= 0 else "down"
    arrow = "▲" if change >= 0 else "▼"
    updated = local_time(md.fetched_at)
    ohlc_note = "candles OHLC" if md.has_ohlc else "somente fechamento"
    st.markdown(
        f"""
<div class="coin-head">
  <div>
    <p class="name"><b>{html.escape(md.coin.name)}</b>{html.escape(md.coin.symbol)}/{md.currency.upper()}</p>
    <div class="price">{fmt_price(price, md.currency)}</div>
  </div>
  <div>
    <div class="chg {direction}">{arrow} {fmt_pct(abs(change))} {change_label}</div>
    <div class="meta">Fonte: {html.escape(md.source_label)} ({ohlc_note}), atualizado em {updated}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if md.stale:
        st.warning(" ".join(md.notices), icon=":material/history:")
    else:
        for notice in md.notices:
            st.info(notice, icon=":material/info:")


def verdict_html(label: str, score: float) -> str:
    cls = "up" if score >= 0.1 else "down" if score <= -0.1 else "flat"
    return f'<p class="verdict {cls}">{html.escape(label)}</p>'


def votes_html(counts: dict) -> str:
    return (
        '<div class="vote-row">'
        f'<span class="u"><b>{counts["alta"]}</b> de alta</span>'
        f'<span class="n"><b>{counts["neutro"]}</b> neutros</span>'
        f'<span class="d"><b>{counts["baixa"]}</b> de baixa</span>'
        "</div>"
    )


def esc(text: str) -> str:
    """Escapa '$' — no Markdown do Streamlit, dois cifrões viram fórmula LaTeX."""
    return text.replace("$", "\\$")


def pct(decimals: int = 1, signed: bool = False):
    return lambda v: fmt_pct(v, decimals, signed)


def num(decimals: int = 2):
    return lambda v: fmt_number(v, decimals)


def money(currency: str, decimals: int = 2):
    return lambda v: fmt_money(v, currency, decimals)


def price(currency: str):
    return lambda v: fmt_price(v, currency)


def date_br(v) -> str:
    return "—" if v is None or pd.isna(v) else pd.Timestamp(v).strftime("%d/%m/%Y")


def table(df: pd.DataFrame, formats: dict | None = None, hide_index: bool = True, **kwargs) -> None:
    """``st.dataframe`` com números no padrão brasileiro; a ordenação continua numérica."""
    styler = df.style
    if formats:
        styler = styler.format({k: v for k, v in formats.items() if k in df.columns}, na_rep="—")
    kwargs.setdefault("placeholder", "—")
    st.dataframe(styler, hide_index=hide_index, **kwargs)


def disclaimer() -> None:
    st.caption(
        "Conteúdo educacional. Nada aqui é recomendação de investimento: resultados passados e modelos estatísticos "
        "não garantem resultados futuros, e criptoativos podem perder a maior parte do valor em pouco tempo."
    )


def signal_style(value: str) -> str:
    from . import theme as T

    color = {"Alta": T.UP, "Baixa": T.DOWN}.get(value, T.MUTED)
    return f"color: {color}; font-weight: 600"
