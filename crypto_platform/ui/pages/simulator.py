"""Simulador: Monte Carlo, "e se eu tivesse investido?" e cenários do modelo de previsão."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import forecast_model as fm
from ... import investment_simulator as sim
from ...config import CURRENCIES
from ...formatting import fmt_money, fmt_pct
from .. import charts
from ..components import coin_header, date_br, disclaimer, esc, money, num, pct, plot, price, table
from ..state import compute_forecast, get_settings, load_current

FREQUENCIES = {"Semanal": 7, "Quinzenal": 14, "Mensal": 30}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=16)
def _monte_carlo(close: pd.Series, investment: float, horizon: int, n: int, method: str, drift: str, fee: float, slip: float):
    return sim.monte_carlo(close, investment, horizon, n, method, drift, fee=fee, slippage=slip)


def render() -> None:
    s = get_settings()
    md = load_current(s)
    coin_header(md)
    close = md.close
    sym = CURRENCIES[s.currency].symbol

    tab_mc, tab_hist, tab_fc = st.tabs(["Futuro: Monte Carlo", "Passado: e se eu tivesse investido?", "Cenários da previsão"])

    # --- Monte Carlo ---------------------------------------------------------
    with tab_mc:
        st.markdown(
            "Simula milhares de caminhos possíveis para o preço reaproveitando blocos de retornos reais do histórico — "
            "mantém as caudas pesadas e os períodos de alta volatilidade que a curva normal ignora."
        )
        with st.form("mc_form", border=True):
            c1, c2, c3, c4 = st.columns(4)
            amount = c1.number_input(f"Valor investido ({sym})", 50.0, 10_000_000.0, 1000.0, 100.0, key="mc_amount", format="%.0f")
            horizon = c2.slider("Prazo (dias)", 7, 365, 90, key="mc_horizon")
            method = c3.selectbox(
                "Método",
                [sim.BOOTSTRAP, sim.GBM],
                format_func=lambda m: "Reamostragem histórica (blocos)" if m == sim.BOOTSTRAP else "Movimento browniano (normal)",
                key="mc_method",
            )
            drift = c4.selectbox(
                "Tendência",
                [sim.DRIFT_ZERO, sim.DRIFT_HISTORICAL],
                format_func=lambda d: "Neutra (só volatilidade)" if d == sim.DRIFT_ZERO else "Repetir a tendência do histórico",
                key="mc_drift",
                help="A tendência histórica de um período de forte alta (ou queda) costuma não se repetir — "
                "o cenário neutro é o ponto de partida mais honesto.",
            )
            n_sims = st.select_slider("Número de simulações", [1000, 2000, 5000, 10000, 20000], value=5000, key="mc_n")
            st.form_submit_button("Simular", type="primary", icon=":material/casino:")

        try:
            mc = _monte_carlo(close, float(amount), int(horizon), int(n_sims), method, drift, s.fee, s.slippage)
        except ValueError as exc:
            st.warning(str(exc), icon=":material/warning:")
            st.stop()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Resultado mediano",
            fmt_money(mc.median_value, s.currency),
            delta=fmt_pct((mc.median_value / amount - 1) * 100, 1, True),
            border=True,
            help="Metade das simulações terminou acima deste valor.",
        )
        k2.metric("Chance de terminar no prejuízo", fmt_pct(mc.prob_loss, 1), border=True)
        k3.metric(
            "Pior 5% dos cenários",
            fmt_money(mc.p5, s.currency),
            delta=f"perda de {fmt_money(mc.var95, s.currency)} ou mais",
            delta_color="off",
            delta_arrow="off",
            border=True,
            help="VaR 95% do investimento no prazo escolhido.",
        )
        k4.metric(
            "Melhor 5% dos cenários",
            fmt_money(mc.p95, s.currency),
            delta=f"chance de dobrar: {fmt_pct(mc.prob_double, 1)}",
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
        plot(charts.monte_carlo_fan(mc, close.index[-1], s.currency), key="mc_fan")
        c_l, c_r = st.columns([1.4, 1])
        with c_l:
            plot(charts.final_value_histogram(mc), key="mc_hist")
        with c_r:
            st.markdown("##### Como ler")
            st.markdown(
                esc(
                    f"- A faixa escura contém metade dos cenários; a clara, 90%.\n"
                    f"- Média dos cenários: **{fmt_money(mc.expected_value, s.currency)}** — maior que a mediana porque "
                    "poucos cenários muito bons puxam a média para cima.\n"
                    f"- Perda média nos 5% piores casos (CVaR): **{fmt_money(mc.cvar95, s.currency)}**.\n"
                    f"- Parâmetros usados: volatilidade anual de **{fmt_pct(mc.annual_vol_used, 1)}** e tendência de "
                    f"**{fmt_pct(mc.annual_drift_used, 1, True)}** ao ano (em log).\n"
                    f"- Taxas de compra e venda ({fmt_pct((s.fee + s.slippage) * 100, 2)} por lado) já descontadas."
                )
            )

    # --- Histórico: aporte único vs. DCA ------------------------------------
    with tab_hist:
        c1, c2, c3 = st.columns(3)
        total = c1.number_input(f"Valor total ({sym})", 50.0, 10_000_000.0, 1000.0, 100.0, key="dca_total", format="%.0f")
        freq_label = (
            c2.segmented_control("Frequência dos aportes", list(FREQUENCIES), default="Semanal", required=True, key="dca_freq") or "Semanal"
        )
        max_days = len(close) - 1
        period_opts = [d for d in (90, 180, 365, 730, 1095, 1825) if d <= max_days] or [max_days]
        period = c3.select_slider(
            "Começando há",
            period_opts,
            value=period_opts[-1] if 365 not in period_opts else 365,
            format_func=lambda d: f"{d} dias",
            key="dca_period",
        )
        window = close.iloc[-(period + 1) :]
        try:
            hc = sim.lump_sum_vs_dca(window, float(total), FREQUENCIES[freq_label], s.fee, s.slippage)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            lump, dca = hc.summary.iloc[0], hc.summary.iloc[1]
            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Aporte único hoje valeria",
                fmt_money(lump["Valor final"], s.currency),
                delta=fmt_pct(lump["Retorno (%)"], 1, True),
                border=True,
            )
            k2.metric(
                f"Aportes {freq_label.lower()}s hoje valeriam",
                fmt_money(dca["Valor final"], s.currency),
                delta=fmt_pct(dca["Retorno (%)"], 1, True),
                border=True,
            )
            winner = "Aporte único" if lump["Valor final"] >= dca["Valor final"] else "Aportes periódicos"
            k3.metric(
                "Melhor neste período",
                winner,
                delta=f"diferença de {fmt_money(abs(lump['Valor final'] - dca['Valor final']), s.currency)}",
                delta_color="off",
                delta_arrow="off",
                border=True,
            )
            plot(
                charts.equity_chart(
                    {"Aporte único": hc.equity["aporte_unico"], "Aportes periódicos (DCA)": hc.equity["dca"]},
                    invested=hc.equity["investido_dca"],
                    y_title=f"Patrimônio ({sym})",
                ),
                key="dca_chart",
            )
            table(
                hc.summary.reset_index(),
                {
                    "Valor final": money(s.currency),
                    "Resultado": money(s.currency),
                    "Retorno (%)": pct(2, signed=True),
                    "Preço médio": price(s.currency),
                    "Pior queda do patrimônio (%)": pct(1),
                },
            )
            st.caption(
                "Com preços reais e taxas descontadas. No DCA o dinheiro ainda não aplicado fica parado (sem rendimento). "
                "Em mercados que sobem, o aporte único tende a ganhar; o DCA reduz o risco de entrar no pior momento."
            )
            with st.expander(f"Compras do DCA ({len(hc.purchases)})"):
                table(hc.purchases, {"Data": date_br, "Preço": price(s.currency), "Valor aplicado": money(s.currency), "Unidades": num(8)})

    # --- Cenários da previsão -------------------------------------------------
    with tab_fc:
        c1, c2 = st.columns(2)
        inv = c1.number_input(f"Valor investido ({sym})", 50.0, 10_000_000.0, 1000.0, 100.0, key="sc_amount", format="%.0f")
        h = c2.slider("Prazo (dias)", 7, 90, 30, key="sc_horizon")
        try:
            with st.spinner("Calculando previsões…"):
                rep = compute_forecast(close, int(h), tuple(fm.DEFAULT_MODELS), 6)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            fc = rep.forecasts.get(fm.ENSEMBLE) or rep.forecasts[rep.best_model]
            scen = sim.forecast_scenarios(rep.last_price, fc, float(inv), s.fee, s.slippage)
            plot(charts.scenario_bars(scen, float(inv)), key="sc_bars")
            table(
                scen.reset_index(),
                {
                    "Preço no fim": price(s.currency),
                    "Valor final": money(s.currency),
                    "Resultado": money(s.currency),
                    "Retorno (%)": pct(2, True),
                },
            )
            st.caption(
                f"Baseado na {fc.name.lower()} dos modelos (página Previsão). Os cenários extremos correspondem aos limites do "
                f"intervalo de 95%: o modelo espera que o preço termine fora dessa faixa em cerca de 1 a cada 20 casos. "
                f"{'Nenhum modelo superou o passeio aleatório no backtest — o cenário base não deve ser lido como tendência.' if not rep.beats_naive else ''}"
            )
    disclaimer()
