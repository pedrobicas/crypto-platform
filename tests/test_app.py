"""Testes de interface com o AppTest do Streamlit (sem navegador, dados sintéticos)."""

import pytest
from streamlit.testing.v1 import AppTest

from crypto_platform.ui.app import PAGE_SPECS

PAGE_SCRIPT = """
import streamlit as st
from crypto_platform.ui.state import render_sidebar
st.session_state["settings"] = render_sidebar()
from crypto_platform.ui.pages import {module}
{module}.render()
"""


def _page(module: str) -> AppTest:
    return AppTest.from_string(PAGE_SCRIPT.format(module=module), default_timeout=180)


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_main_entrypoint_runs():
    at = AppTest.from_file("../main.py", default_timeout=180).run()
    _assert_clean(at)
    assert any(m.label == "Últimos 7 dias" for m in at.metric)


@pytest.mark.parametrize("module", [spec[1] for spec in PAGE_SPECS])
def test_every_page_renders(module):
    at = _page(module).run()
    _assert_clean(at)


def test_overview_switch_coin_and_currency():
    at = _page("overview").run()
    at.selectbox(key="coin_choice").select("ethereum").run()
    at.segmented_control(key="currency").set_value("brl").run()
    _assert_clean(at)
    assert "R$" in "".join(m.value for m in at.metric)


def test_custom_coin_id():
    at = _page("overview").run()
    at.selectbox(key="coin_choice").select("__custom__").run()
    _assert_clean(at)


def test_forecast_with_other_parameters():
    at = _page("forecast").run()
    at.slider(key="fc_horizon").set_value(14)
    at.multiselect(key="fc_models").set_value(["Passeio aleatório c/ tendência", "Theta"])
    at.button[0].click().run()  # "Gerar previsão" (submit do formulário)
    _assert_clean(at)
    assert any("14 dias" in m.label for m in at.metric)


def test_portfolio_methods():
    at = _page("portfolio").run()
    for method in ("Mínima variância", "Máximo Sharpe", "Paridade de risco", "Manual"):
        at.segmented_control(key="pf_method").set_value(method).run()
        _assert_clean(at)


def test_portfolio_needs_two_coins():
    at = _page("portfolio").run()
    at.multiselect(key="pf_coins").set_value(["bitcoin"]).run()
    _assert_clean(at)
    assert at.info


def test_backtest_strategy_selection():
    at = _page("backtest").run()
    at.multiselect(key="bt_selected").set_value(["donchian", "bollinger"]).run()
    _assert_clean(at)


def test_risk_page_for_altcoin_loads_benchmark():
    at = _page("risk").run()
    at.selectbox(key="coin_choice").select("solana").run()
    _assert_clean(at)
    assert any(m.label.startswith("Beta") for m in at.metric)


def test_forecast_warns_when_few_test_windows():
    at = _page("forecast").run()
    at.select_slider(key="history_days").set_value(180).run()
    at.slider(key="fc_horizon").set_value(45)
    at.button[0].click().run()
    _assert_clean(at)
    assert any("janela" in w.value for w in at.warning)  # previsão exibida, com aviso de poucas janelas
    at.slider(key="fc_horizon").set_value(90)
    at.button[0].click().run()
    _assert_clean(at)
    assert any("insuficiente" in w.value for w in at.warning)  # nenhuma janela possível: explica o motivo
