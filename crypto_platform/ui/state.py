"""Configurações globais (barra lateral) e carregadores com cache do Streamlit."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from .. import forecast_model as fm
from .. import technical_indicators as ti
from ..config import (
    COINS,
    COINS_BY_ID,
    CURRENCIES,
    DEFAULT_FEE,
    DEFAULT_HISTORY_DAYS,
    DEFAULT_SLIPPAGE,
    DEMO_MODE,
    HISTORY_OPTIONS,
    Coin,
    find_coin,
)
from ..data_collector import (
    SOURCE_AUTO,
    SOURCE_COINGECKO,
    SOURCE_LABELS,
    SOURCE_YAHOO,
    CryptoDataCollector,
    DataSourceError,
    MarketData,
)

CUSTOM = "__custom__"
MIN_HISTORY_DAYS = 30
HISTORY_LABELS = {180: "6 meses", 365: "1 ano", 730: "2 anos", 1095: "3 anos", 1825: "5 anos"}


def period_label(days: int) -> str:
    """Rótulo amigável para um número de dias (tolerância de alguns dias por fonte)."""
    for d, label in HISTORY_LABELS.items():
        if abs(days - d) <= 5:
            return label
    return f"{days} dias"


@dataclass(frozen=True)
class Settings:
    coin: Coin
    currency: str
    days: int
    source: str
    risk_free: float  # fração anual
    fee: float
    slippage: float
    refresh_token: float


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)  # type: ignore[attr-defined]
    except Exception:
        value = None
    return value or os.environ.get(name) or None


@st.cache_resource(show_spinner=False)
def get_collector() -> CryptoDataCollector:
    return CryptoDataCollector(
        coingecko_demo_key=_secret("COINGECKO_DEMO_API_KEY"),
        coingecko_pro_key=_secret("COINGECKO_PRO_API_KEY"),
    )


def render_sidebar() -> Settings:
    sb = st.sidebar
    sb.markdown("### Crypto Analysis & Forecast")
    if DEMO_MODE:
        sb.warning("Modo demonstração: todos os preços são **sintéticos**.", icon=":material/science:")

    options = [c.coingecko_id for c in COINS] + [CUSTOM]
    choice = sb.selectbox(
        "Criptomoeda",
        options,
        format_func=lambda cid: "Outra (ID da CoinGecko)…" if cid == CUSTOM else COINS_BY_ID[cid].label,
        key="coin_choice",
    )
    if choice == CUSTOM:
        custom = (
            sb.text_input(
                "ID da moeda na CoinGecko",
                value=st.session_state.get("custom_coin", "pepe"),
                key="custom_coin",
                help="É o final do endereço da moeda no site da CoinGecko. Ex.: coingecko.com/pt/moedas/**pepe** → pepe",
            )
            .strip()
            .lower()
        )
        coin = find_coin(custom or "bitcoin")
    else:
        coin = COINS_BY_ID[choice]

    currency = (
        sb.segmented_control(
            "Cotação",
            list(CURRENCIES),
            format_func=lambda c: c.upper(),
            default="usd",
            required=True,
            key="currency",
        )
        or "usd"
    )

    days = sb.select_slider(
        "Histórico",
        options=list(HISTORY_OPTIONS),
        value=DEFAULT_HISTORY_DAYS,
        format_func=lambda d: HISTORY_LABELS.get(d, f"{d} dias"),
        key="history_days",
        help="Acima de 1 ano os dados vêm do Yahoo Finance (a CoinGecko gratuita limita a 365 dias).",
    )

    source = sb.selectbox(
        "Fonte de dados",
        [SOURCE_AUTO, SOURCE_COINGECKO, SOURCE_YAHOO],
        format_func=lambda s: SOURCE_LABELS[s],
        key="source",
        help=(
            "Automático: tenta a CoinGecko e, se falhar, o Yahoo Finance.\n\n"
            "Yahoo Finance traz candles completos (abertura, máxima, mínima e fechamento), "
            "o que deixa ATR, Estocástico e ADX mais precisos."
        ),
    )

    with sb.expander("Parâmetros avançados", icon=":material/tune:"):
        cur = CURRENCIES[currency]
        rf = st.number_input(
            f"Taxa livre de risco ({currency.upper()}, % ao ano)",
            min_value=0.0,
            max_value=50.0,
            value=cur.default_risk_free * 100,
            step=0.25,
            key=f"rf_{currency}",
            help="Usada no Sharpe e no Sortino. Valor inicial aproximado — ajuste para a taxa atual "
            "(ex.: CDI/Selic para BRL, T-bill para USD).",
        )
        fee = st.number_input("Taxa por operação (%)", 0.0, 2.0, DEFAULT_FEE * 100, 0.05, key="fee", format="%.2f")
        slip = st.number_input("Slippage por operação (%)", 0.0, 2.0, DEFAULT_SLIPPAGE * 100, 0.05, key="slippage", format="%.2f")

    if sb.button("Atualizar dados", icon=":material/refresh:", width="stretch"):
        st.session_state["refresh_token"] = time.time()
        st.cache_data.clear()

    return Settings(
        coin=coin,
        currency=currency,
        days=int(days),
        source=source,
        risk_free=rf / 100,
        fee=fee / 100,
        slippage=slip / 100,
        refresh_token=float(st.session_state.get("refresh_token", 0.0)),
    )


def get_settings() -> Settings:
    return st.session_state["settings"]


# ---------------------------------------------------------------------------
# Carregadores com cache
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def _load_market_data(
    coin_id: str, name: str, symbol: str, yahoo: str | None, currency: str, days: int, source: str, refresh_token: float
) -> MarketData:
    coin = Coin(name, symbol, coin_id, yahoo)
    return get_collector().get_market_data(coin, days, currency, source, min_saved_at=refresh_token or None)


def load_market_data(coin: Coin, settings: Settings) -> MarketData:
    return _load_market_data(
        coin.coingecko_id,
        coin.name,
        coin.symbol,
        coin.yahoo_ticker,
        settings.currency,
        settings.days,
        settings.source,
        settings.refresh_token,
    )


def load_current(settings: Settings) -> MarketData:
    """Carrega a moeda selecionada ou interrompe a página com uma mensagem útil."""
    try:
        with st.spinner(f"Carregando {settings.coin.name}…"):
            md = load_market_data(settings.coin, settings)
    except DataSourceError as exc:
        st.error(f"Não foi possível carregar os dados de **{settings.coin.name}**.\n\n{exc}", icon=":material/cloud_off:")
        st.markdown(
            "Tente: **Atualizar dados** daqui a um minuto, trocar a **fonte de dados** na barra lateral, "
            "ou configurar uma chave gratuita da CoinGecko (`COINGECKO_DEMO_API_KEY`) — veja o README."
        )
        st.stop()
    if len(md.df) < MIN_HISTORY_DAYS:
        st.warning(
            f"**{settings.coin.name}** tem só {len(md.df)} dias de cotação — são necessários pelo menos {MIN_HISTORY_DAYS} "
            "para as análises. Escolha outra moeda ou volte quando houver mais histórico.",
            icon=":material/hourglass_empty:",
        )
        st.stop()
    return md


@st.cache_data(ttl=600, show_spinner=False)
def load_snapshot(coin_ids: tuple[str, ...], currency: str, refresh_token: float) -> pd.DataFrame:
    return get_collector().get_market_snapshot(coin_ids, currency)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fear_greed(refresh_token: float) -> pd.DataFrame:
    return get_collector().get_fear_greed(120)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return ti.add_all_indicators(df)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def compute_forecast(close: pd.Series, horizon: int, models: tuple[str, ...], folds: int) -> fm.ForecastReport:
    return fm.run_forecast(close, horizon, models, folds)
