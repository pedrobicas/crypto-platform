"""Risco: score explicável, volatilidade, drawdown, distribuição, correlação e tamanho de posição."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ... import risk_analyzer as ra
from ...config import BENCHMARK
from ...data_collector import DataSourceError
from ...formatting import fmt_money, fmt_number, fmt_pct, fmt_price, fmt_ratio
from .. import charts
from ..components import coin_header, disclaimer, plot
from ..state import compute_indicators, get_settings, load_current, load_market_data


def render() -> None:
    s = get_settings()
    md = load_current(s)
    coin_header(md)
    close = md.close
    returns = md.df["returns"].dropna()
    rep = ra.analyze(close, s.risk_free)

    top_l, top_r = st.columns([1, 1.6], gap="medium")
    with top_l:
        with st.container(border=True):
            st.markdown("##### Score de risco")
            plot(charts.risk_gauge(rep.score["score"]), key="rk_gauge")
            st.markdown(f"<p class='verdict flat'>Risco {rep.score['level'].lower()}</p>", unsafe_allow_html=True)
            st.caption("Escala calibrada para criptoativos. Mesmo um score baixo aqui é alto frente a ações ou renda fixa.")
    with top_r:
        with st.container(border=True):
            st.markdown("##### O que compõe o score")
            plot(charts.risk_breakdown(rep.score["components"]), key="rk_breakdown")
            st.caption("Cada fator vira uma nota de 0 a 10 e entra na média com o peso indicado.")

    m = st.columns(6)
    m[0].metric("Volatilidade", fmt_pct(rep.volatility.get(30), 1), help="Últimos 30 dias, anualizada.", border=True)
    m[1].metric(
        "VaR 95%",
        fmt_pct(rep.var95["historical_var"], 2),
        help="Perda diária que não foi ultrapassada em 95% dos dias (histórico).",
        border=True,
    )
    m[2].metric("CVaR 95%", fmt_pct(rep.var95["historical_cvar"], 2), help="Perda média nos 5% piores dias.", border=True)
    m[3].metric(
        "Pior queda", fmt_pct(rep.drawdown.max_drawdown, 1), help="Drawdown máximo: maior queda do topo ao fundo no período.", border=True
    )
    m[4].metric(
        "Sharpe",
        fmt_ratio(rep.sharpe),
        help=f"Retorno acima da taxa livre de risco ({fmt_pct(s.risk_free * 100, 2)} a.a.) por unidade de volatilidade.",
        border=True,
    )
    m[5].metric("Sortino", fmt_ratio(rep.sortino), help="Como o Sharpe, mas penaliza só a volatilidade das quedas.", border=True)

    tabs = st.tabs(["Volatilidade", "Drawdown", "Distribuição dos retornos", f"Relação com {BENCHMARK.symbol}", "Tamanho de posição"])

    with tabs[0]:
        series = {
            "30 dias": ra.rolling_volatility(returns, 30),
            "90 dias": ra.rolling_volatility(returns, 90),
            "EWMA (λ = 0,94)": ra.ewma_volatility(returns),
        }
        plot(charts.rolling_vol_chart(series), key="rk_vol")
        v = rep.volatility
        st.markdown(
            f"Volatilidade anualizada atual: **7 dias {fmt_pct(v.get(7), 1)}**, **30 dias {fmt_pct(v.get(30), 1)}**, "
            f"**90 dias {fmt_pct(v.get(90), 1)}**, **1 ano {fmt_pct(v.get(365), 1)}**. A EWMA reage mais rápido a choques "
            f"recentes (hoje: {fmt_pct(rep.ewma_vol, 1)})."
        )
        daily = (v.get(30) or np.nan) / np.sqrt(365)
        st.caption(f"Na prática: com volatilidade de 30 dias, um dia típico move o preço cerca de ±{fmt_pct(daily, 1)}.")

    with tabs[1]:
        dd = rep.drawdown
        plot(charts.drawdown_chart(dd), key="rk_dd")
        if dd.peak_date is not None:
            rec = f"recuperado em {dd.recovery_date:%d/%m/%Y}" if dd.recovery_date is not None else "**ainda não recuperado**"
            st.markdown(
                f"Pior queda: **{fmt_pct(dd.max_drawdown, 1)}**, do topo em {dd.peak_date:%d/%m/%Y} ao fundo em "
                f"{dd.trough_date:%d/%m/%Y}; {rec} ({dd.duration_days} dias desde o topo). Queda atual em relação ao "
                f"topo do período: **{fmt_pct(dd.current_drawdown, 1)}**."
            )
        st.caption(
            f"Calmar (retorno anual ÷ drawdown máximo): {fmt_ratio(rep.calmar)}. Retorno anualizado no período: {fmt_pct(rep.cagr * 100, 1)}."
        )

    with tabs[2]:
        plot(charts.returns_histogram(returns, rep.var95["historical_var"], rep.var95["historical_cvar"]), key="rk_hist")
        t = rep.tail
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"- Assimetria: **{fmt_number(t['skewness'], 2)}** "
                + (
                    "(caudas de queda mais longas)"
                    if t["skewness"] < -0.2
                    else "(caudas de alta mais longas)"
                    if t["skewness"] > 0.2
                    else "(aproximadamente simétrica)"
                )
                + f"\n- Curtose em excesso: **{fmt_number(t['excess_kurtosis'], 2)}** "
                + ("(caudas pesadas: dias extremos são mais frequentes que na curva normal)" if t["excess_kurtosis"] > 1 else "")
                + f"\n- Dias além de 3 desvios-padrão: **{fmt_pct(t['extreme_freq_pct'], 2)}** (numa normal seriam {fmt_pct(t['extreme_freq_normal_pct'], 2)})"
                + f"\n- Dias de alta: **{fmt_pct(t['positive_days_pct'], 1)}**"
            )
        with c2:
            st.markdown(
                f"- Melhor dia: **{fmt_pct(t['best_day'], 2, True)}** em {t['best_day_date']:%d/%m/%Y}"
                f"\n- Pior dia: **{fmt_pct(t['worst_day'], 2)}** em {t['worst_day_date']:%d/%m/%Y}"
                f"\n- VaR 95% histórico / normal / Cornish-Fisher: **{fmt_pct(rep.var95['historical_var'], 2)}** / "
                f"{fmt_pct(rep.var95['normal_var'], 2)} / {fmt_pct(rep.var95['cornish_fisher_var'], 2)}"
                f"\n- VaR 99% histórico: **{fmt_pct(rep.var99['historical_var'], 2)}** (CVaR {fmt_pct(rep.var99['historical_cvar'], 2)})"
            )
        st.caption(
            "Teste de Jarque-Bera: "
            + ("os retornos **não** seguem distribuição normal" if not t["is_normal"] else "não rejeita normalidade")
            + f" (p = {fmt_number(t['jarque_bera_pvalue'], 4)}). Por isso o VaR histórico é a referência: o normal ignora as caudas pesadas e o Cornish-Fisher, que corrige assimetria e curtose, pode distorcer quando as caudas são muito extremas."
        )

    with tabs[3]:
        if s.coin.coingecko_id == BENCHMARK.coingecko_id:
            st.info(
                f"A moeda selecionada já é o {BENCHMARK.name}. Escolha outra moeda para ver beta e correlação com o {BENCHMARK.symbol}."
            )
        else:
            try:
                bench = load_market_data(BENCHMARK, s)
            except DataSourceError as exc:
                st.warning(f"Não foi possível carregar o {BENCHMARK.name}: {exc}")
            else:
                bc = ra.beta_correlation(returns, bench.df["returns"])
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    f"Beta vs. {BENCHMARK.symbol}",
                    fmt_ratio(bc["beta"]),
                    help=f"Se o {BENCHMARK.symbol} sobe 1%, esta moeda tende a subir {fmt_number(bc['beta'], 2)}%.",
                    border=True,
                )
                c2.metric("Correlação", fmt_ratio(bc["correlation"]), border=True)
                c3.metric(f"Movimento explicado pelo {BENCHMARK.symbol}", fmt_pct(bc["r_squared"] * 100, 0), help="R²", border=True)
                plot(charts.rolling_corr_chart(bc["rolling_corr"], f"Correlação com {BENCHMARK.symbol}"), key="rk_corr")
                st.caption("Correlação alta significa pouca diversificação: em quedas fortes do bitcoin, a maioria das altcoins cai junto.")

    with tabs[4]:
        st.markdown(
            "Dimensionamento por risco fixo: defina quanto do capital aceita perder se o stop for atingido. "
            "A distância do stop pode ser baseada no ATR (amplitude média diária)."
        )
        ind = compute_indicators(md.df)
        atr_pct = float(ind["atr_pct"].iloc[-1]) if np.isfinite(ind["atr_pct"].iloc[-1]) else 5.0
        c1, c2, c3 = st.columns(3)
        capital = c1.number_input(
            f"Capital total ({s.currency.upper()})", min_value=100.0, value=10000.0, step=500.0, key="ps_capital", format="%.0f"
        )
        risk_pct = c2.select_slider(
            "Risco máximo por operação",
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
            value=1.0,
            format_func=lambda v: f"{fmt_pct(v, 2)} do capital",
            key="ps_risk",
        )
        mult = c3.select_slider(
            "Distância do stop",
            [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            value=2.0,
            format_func=lambda v: f"{fmt_number(v, 1)} × ATR",
            key="ps_mult",
            help=f"ATR atual: {fmt_pct(atr_pct, 2)} do preço.",
        )
        stop = atr_pct * mult
        res = ra.position_size(capital, risk_pct, stop)
        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Tamanho da posição",
            fmt_money(res["position_value"], s.currency),
            delta=f"{fmt_pct(res['position_pct'], 1)} do capital",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        r2.metric(
            "Distância do stop",
            fmt_pct(stop, 2),
            delta=f"preço de stop ≈ {fmt_price(close.iloc[-1] * (1 - stop / 100), s.currency)}",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        r3.metric("Perda se o stop for atingido", fmt_money(res["max_loss"], s.currency), border=True)
        st.caption(
            "Stops podem ser ultrapassados em quedas bruscas (gaps e baixa liquidez): a perda real pode ser maior. "
            f"Referência: o VaR 99% diário desta moeda é {fmt_pct(rep.var99['historical_var'], 2)}."
        )

    with st.expander("Todas as métricas", icon=":material/table:"):
        rows = [
            ("Retorno anualizado (CAGR)", fmt_pct(rep.cagr * 100, 2)),
            ("Volatilidade anualizada (período)", fmt_pct(rep.ann_vol * 100, 2)),
            *[(f"Volatilidade {w} dias", fmt_pct(v, 2)) for w, v in rep.volatility.items()],
            ("Volatilidade EWMA", fmt_pct(rep.ewma_vol, 2)),
            ("VaR 95% histórico (1 dia)", fmt_pct(rep.var95["historical_var"], 2)),
            ("CVaR 95% histórico (1 dia)", fmt_pct(rep.var95["historical_cvar"], 2)),
            ("VaR 95% normal", fmt_pct(rep.var95["normal_var"], 2)),
            ("VaR 95% Cornish-Fisher", fmt_pct(rep.var95["cornish_fisher_var"], 2)),
            ("VaR 99% histórico", fmt_pct(rep.var99["historical_var"], 2)),
            ("CVaR 99% histórico", fmt_pct(rep.var99["historical_cvar"], 2)),
            ("Drawdown máximo", fmt_pct(rep.drawdown.max_drawdown, 2)),
            ("Drawdown atual", fmt_pct(rep.drawdown.current_drawdown, 2)),
            ("Sharpe", fmt_ratio(rep.sharpe)),
            ("Sortino", fmt_ratio(rep.sortino)),
            ("Calmar", fmt_ratio(rep.calmar)),
            ("Assimetria", fmt_number(rep.tail["skewness"], 3)),
            ("Curtose em excesso", fmt_number(rep.tail["excess_kurtosis"], 3)),
            ("Score de risco", f"{fmt_number(rep.score['score'], 2)} ({rep.score['level']})"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Métrica", "Valor"]), hide_index=True)
    disclaimer()
