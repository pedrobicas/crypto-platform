"""Portfólio: comparação entre moedas, correlação, otimização e backtest da carteira."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ... import portfolio as pf
from ...config import BENCHMARK, COINS, COINS_BY_ID, CURRENCIES
from ...data_collector import DataSourceError
from ...formatting import fmt_number, fmt_pct
from .. import charts
from ..components import disclaimer, money, num, pct, plot, table
from ..state import get_settings, load_market_data

DEFAULT_BASKET = ["bitcoin", "ethereum", "solana", "ripple", "binancecoin"]
METRIC_FORMATS = {
    "Retorno no período (%)": pct(1, True),
    "Retorno total (%)": pct(1, True),
    "Retorno anualizado (%)": pct(1, True),
    "Volatilidade anual (%)": pct(1),
    "Sharpe": num(2),
    "Sortino": num(2),
    "Drawdown máximo (%)": pct(1),
}


def render() -> None:
    s = get_settings()
    sym = CURRENCIES[s.currency].symbol
    st.markdown("## Portfólio")

    ids = st.multiselect(
        "Moedas da carteira (até 8)",
        [c.coingecko_id for c in COINS],
        default=DEFAULT_BASKET,
        max_selections=8,
        format_func=lambda cid: COINS_BY_ID[cid].label,
        key="pf_coins",
    )
    if len(ids) < 2:
        st.info("Escolha pelo menos duas moedas para comparar e montar uma carteira.", icon=":material/add_circle:")
        st.stop()

    series, failed = {}, []
    progress = st.progress(0.0, text="Carregando preços…")
    for i, cid in enumerate(ids):
        coin = COINS_BY_ID[cid]
        try:
            series[coin.symbol] = load_market_data(coin, s).close
        except DataSourceError as exc:
            failed.append(f"{coin.name}: {exc}")
        progress.progress((i + 1) / len(ids), text=f"Carregando {coin.name}…")
    progress.empty()
    for f in failed:
        st.warning(f"Moeda ignorada — {f}", icon=":material/cloud_off:")
    prices = pf.align_prices(series)
    if prices.shape[1] < 2 or len(prices) < 60:
        st.error("Dados em comum insuficientes entre as moedas escolhidas (é preciso ao menos 60 dias e 2 moedas).")
        st.stop()
    st.caption(
        f"Período em comum: {prices.index[0]:%d/%m/%Y} a {prices.index[-1]:%d/%m/%Y} ({len(prices)} dias), cotação em {s.currency.upper()}."
    )

    stats = pf.asset_stats(prices, s.risk_free)
    tabs = st.tabs(["Desempenho", "Correlação", "Risco × retorno", "Otimização", "Backtest da carteira"])

    with tabs[0]:
        plot(charts.normalized_performance(prices), key="pf_perf")
        table(stats.reset_index(names="Moeda"), METRIC_FORMATS)

    with tabs[1]:
        corr = pf.daily_returns(prices).corr()
        plot(
            charts.heatmap(corr, "Correlação dos retornos diários", colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1, height=440), key="pf_corr"
        )
        off = corr.where(~np.eye(len(corr), dtype=bool))
        st.caption(
            f"Correlação média entre pares: **{fmt_number(off.stack().mean(), 2)}**. Acima de 0,7 a diversificação entre "
            "criptomoedas é limitada — elas tendem a cair juntas nos momentos de estresse."
        )

    with tabs[2]:
        plot(charts.risk_return_scatter(stats), key="pf_scatter")
        st.caption("Canto superior esquerdo = mais retorno com menos risco no período. Retornos passados não se repetem necessariamente.")

    with tabs[3]:
        c1, c2 = st.columns([2, 1])
        method = c1.segmented_control("Método", list(pf.METHODS), default=pf.EQUAL, required=True, key="pf_method") or pf.EQUAL
        max_w = (
            c2.slider(
                "Peso máximo por moeda",
                20,
                100,
                60,
                5,
                format="%d%%",
                key="pf_maxw",
                help="Limita a concentração. Com 100% o otimizador pode pôr tudo numa moeda só.",
            )
            / 100
        )
        st.caption(pf.METHOD_HELP[method])
        manual = None
        if method == pf.MANUAL:
            cols = st.columns(min(len(prices.columns), 4))
            manual = {}
            for i, col in enumerate(prices.columns):
                manual[col] = cols[i % len(cols)].number_input(
                    f"{col} (%)", 0.0, 100.0, round(100 / len(prices.columns), 1), 1.0, key=f"pf_w_{col}"
                )
        weights = pf.optimize(prices, method, max_w, s.risk_free, manual)
        st.session_state["pf_weights"] = weights

        point = pf.portfolio_point(weights, prices, s.risk_free)
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Retorno histórico (a.a.)",
            fmt_pct(point["retorno"], 1, True),
            border=True,
            help="Crescimento anual estimado pela média ponderada dos retornos log de cada moeda. "
            "Para uma moeda isolada é igual ao retorno anualizado da aba Desempenho.",
        )
        k2.metric("Volatilidade (a.a.)", fmt_pct(point["vol"], 1), border=True, help="Com covariância encolhida (Ledoit-Wolf).")
        k3.metric("Sharpe", fmt_number(point["sharpe"], 2), border=True)

        col_w, col_f = st.columns([1, 1.4], gap="medium")
        with col_w:
            plot(charts.weights_bar(weights, pf.risk_contributions(weights, prices), height=320), key="pf_weights")
        with col_f:
            points = {method: point}
            for m in (pf.EQUAL, pf.MIN_VAR, pf.MAX_SHARPE):
                if m != method:
                    points[m] = pf.portfolio_point(pf.optimize(prices, m, max_w, s.risk_free), prices, s.risk_free)
            plot(charts.frontier_chart(pf.random_portfolios(prices, risk_free=s.risk_free), points, stats, height=380), key="pf_frontier")
        st.caption(
            'Otimizações usam o passado como se fosse o futuro: o "Máximo Sharpe" costuma concentrar nas moedas que mais subiram '
            "e decepcionar depois. Use o Backtest da carteira e prefira limites de peso."
        )

    with tabs[4]:
        weights = st.session_state.get("pf_weights", pf.optimize(prices, pf.EQUAL))
        c1, c2 = st.columns([1, 1])
        rule = (
            c1.segmented_control("Rebalanceamento", list(pf.REBALANCE_RULES), default="Mensal", required=True, key="pf_rebal") or "Mensal"
        )
        initial = c2.number_input(f"Capital inicial ({sym})", 100.0, 10_000_000.0, 1000.0, 100.0, key="pf_initial", format="%.0f")
        st.caption(
            "Usa os pesos definidos na aba Otimização: "
            + ", ".join(f"{k} {fmt_pct(v * 100, 1)}" for k, v in weights.items() if v > 0)
            + "."
        )
        main = pf.backtest_portfolio(prices, weights, pf.REBALANCE_RULES[rule], float(initial), s.fee, s.slippage, s.risk_free)
        curves = {"Sua carteira": main.equity}
        equal = pf.backtest_portfolio(
            prices, pf.optimize(prices, pf.EQUAL), pf.REBALANCE_RULES[rule], float(initial), s.fee, s.slippage, s.risk_free
        )
        curves["Pesos iguais"] = equal.equity
        bench_sym = BENCHMARK.symbol
        if bench_sym in prices.columns:
            only = pd.Series({c: 1.0 if c == bench_sym else 0.0 for c in prices.columns})
            curves[f"Só {bench_sym}"] = pf.backtest_portfolio(prices, only, None, float(initial), s.fee, s.slippage, s.risk_free).equity
        plot(charts.equity_chart(curves, y_title=f"Patrimônio ({sym})"), key="pf_bt")
        results = pd.DataFrame({"Sua carteira": main.metrics, "Pesos iguais": equal.metrics})
        if f"Só {bench_sym}" in curves:
            only_bt = pf.backtest_portfolio(prices, only, None, float(initial), s.fee, s.slippage, s.risk_free)
            results[f"Só {bench_sym}"] = pd.Series(only_bt.metrics)
        table(results.T.reset_index(names="Carteira"), {**METRIC_FORMATS, "Custos totais": money(s.currency)})
        st.caption("Pesos otimizados com o mesmo período do backtest favorecem a carteira otimizada (viés de olhar para trás).")
    disclaimer()
