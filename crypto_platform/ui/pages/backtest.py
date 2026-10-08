"""Estratégias: backtest de regras técnicas contra o "comprar e segurar"."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import backtesting as bt
from ...config import CURRENCIES
from ...formatting import fmt_money, fmt_pct
from .. import charts
from ..components import coin_header, date_br, disclaimer, num, pct, plot, price, table
from ..state import get_settings, load_current

PARAM_LABELS = {
    "fast": "Média rápida (dias)",
    "slow": "Média lenta (dias)",
    "period": "Período",
    "signal": "Linha de sinal",
    "lower": "Comprar abaixo de",
    "upper": "Vender acima de",
    "n_std": "Desvios-padrão",
    "entry": "Rompimento de N dias",
    "exit": "Saída na mínima de M dias",
}


@st.cache_data(ttl=3600, show_spinner=False, max_entries=16)
def _run(df: pd.DataFrame, config: tuple, capital: float, fee: float, slip: float, rf: float):
    return bt.run_backtest(df, {k: dict(v) for k, v in config}, capital, fee, slip, rf)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=16)
def _grid(df: pd.DataFrame, fee: float, slip: float, rf: float):
    return bt.parameter_grid(df, fee=fee, slippage=slip, risk_free=rf)


def render() -> None:
    s = get_settings()
    md = load_current(s)
    coin_header(md)
    sym = CURRENCIES[s.currency].symbol

    choices = [k for k in bt.STRATEGIES if k != "buy_hold"]
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        selected = c1.multiselect(
            "Estratégias para comparar com o comprar e segurar",
            choices,
            default=["sma_cross", "trend_filter", "macd", "rsi"],
            format_func=lambda k: bt.STRATEGIES[k].name,
            key="bt_selected",
        )
        capital = c2.number_input(f"Capital inicial ({sym})", 100.0, 10_000_000.0, 1000.0, 100.0, key="bt_capital", format="%.0f")
        params: dict[str, dict] = {}
        if selected:
            with st.expander("Parâmetros das estratégias", icon=":material/tune:"):
                cols = st.columns(min(len(selected), 3))
                for i, key in enumerate(selected):
                    spec = bt.STRATEGIES[key]
                    with cols[i % len(cols)]:
                        st.markdown(f"**{spec.name}**")
                        st.caption(spec.description)
                        p = {}
                        for name, default in spec.params.items():
                            if isinstance(default, float):
                                p[name] = st.number_input(
                                    PARAM_LABELS.get(name, name), value=float(default), step=0.5, key=f"bt_{key}_{name}"
                                )
                            else:
                                p[name] = int(
                                    st.number_input(
                                        PARAM_LABELS.get(name, name),
                                        value=int(default),
                                        min_value=2,
                                        max_value=400,
                                        step=1,
                                        key=f"bt_{key}_{name}",
                                    )
                                )
                        params[key] = p

    config = tuple((k, tuple(sorted(params.get(k, {}).items()))) for k in selected)
    try:
        results = _run(md.df, config, float(capital), s.fee, s.slippage, s.risk_free)
    except ValueError as exc:
        st.warning(f"{exc} Aumente o histórico na barra lateral ou reduza os períodos das estratégias.", icon=":material/warning:")
        st.stop()

    bh = results["buy_hold"]
    ordered = ["buy_hold", *[k for k in selected if k in results]]
    start = bh.equity.index[0]
    st.caption(
        f"Período avaliado: {start:%d/%m/%Y} a {bh.equity.index[-1]:%d/%m/%Y} (após o aquecimento dos indicadores), "
        f"com {fmt_pct((s.fee + s.slippage) * 100, 2)} de custo por operação."
    )

    winners = [results[k].name for k in selected if k in results and results[k].metrics["Sharpe"] > bh.metrics["Sharpe"]]
    if selected:
        if winners:
            st.info(
                f"Com melhor retorno ajustado ao risco (Sharpe) que o comprar e segurar neste período: **{', '.join(winners)}**. "
                "Um único período histórico não prova que a regra funciona — veja o mapa de parâmetros abaixo.",
                icon=":material/insights:",
            )
        else:
            st.warning(
                "Nenhuma estratégia superou o comprar e segurar em retorno ajustado ao risco neste período.", icon=":material/balance:"
            )

    log_scale = st.toggle("Escala logarítmica", value=False, key="bt_log")
    curves = {results[k].name: results[k].equity for k in ordered}
    plot(charts.equity_chart(curves, log_scale=log_scale, y_title=f"Patrimônio ({sym})"), key="bt_equity")

    metrics = pd.DataFrame({results[k].name: results[k].metrics for k in ordered}).T
    metrics.index.name = "Estratégia"
    table(
        metrics.reset_index(),
        {
            "Retorno total (%)": pct(1, True),
            "Retorno anualizado (%)": pct(1, True),
            "Volatilidade anual (%)": pct(1),
            "Sharpe": num(2),
            "Sortino": num(2),
            "Drawdown máximo (%)": pct(1),
            "Calmar": num(2),
            "Operações": num(0),
            "Taxa de acerto (%)": pct(0),
            "Tempo posicionado (%)": pct(0),
        },
        column_config={"Estratégia": st.column_config.TextColumn(width="medium")},
    )

    tab_dd, tab_trades, tab_grid = st.tabs(["Quedas do patrimônio", "Operações", "Mapa de parâmetros (risco de sobreajuste)"])
    with tab_dd:
        plot(charts.drawdown_lines(curves), key="bt_dd")
        st.caption(
            "Estratégias que saem do mercado em tendências de baixa costumam ter quedas menores — às custas de perder parte das altas."
        )
    with tab_trades:
        if not selected:
            st.info("Selecione ao menos uma estratégia.")
        else:
            which = st.selectbox(
                "Estratégia", [k for k in selected if k in results], format_func=lambda k: bt.STRATEGIES[k].name, key="bt_trades_sel"
            )
            trades = results[which].trades
            if trades.empty:
                st.info("Nenhuma operação no período.")
            else:
                table(
                    trades,
                    {
                        "Entrada": date_br,
                        "Saída": date_br,
                        "Preço de entrada": price(s.currency),
                        "Preço de saída": price(s.currency),
                        "Retorno (%)": pct(2, True),
                    },
                    column_config={"Aberta": st.column_config.CheckboxColumn("Em aberto")},
                )
                closed = trades[~trades["Aberta"]]
                if len(closed):
                    st.caption(
                        f"Ganho médio nas vencedoras: {fmt_pct(closed.loc[closed['Retorno (%)'] > 0, 'Retorno (%)'].mean(), 2, True)}; "
                        f"perda média nas perdedoras: {fmt_pct(closed.loc[closed['Retorno (%)'] <= 0, 'Retorno (%)'].mean(), 2)}."
                    )
    with tab_grid:
        grid = _grid(md.df, s.fee, s.slippage, s.risk_free)
        if grid.empty:
            st.info("Histórico curto demais para o mapa de parâmetros.")
        else:
            plot(charts.heatmap(grid, "Sharpe do cruzamento de médias para cada combinação", fmt=".2f", height=420), key="bt_grid")
            best = grid.stack().idxmax()
            st.caption(
                f"Avaliado a partir de {grid.attrs['start']:%d/%m/%Y} (após o aquecimento da maior média), por isso os valores "
                f"podem diferir da tabela acima. O melhor ponto ({best[0]} × {best[1]} dias) foi escolhido olhando o resultado — "
                "é o que se chama de sobreajuste. Se o desempenho muda muito entre células vizinhas, a regra é frágil. "
                "Prefira regiões amplas e estáveis."
            )
    st.caption(
        f"Capital inicial de {fmt_money(capital, s.currency)}. Sinais calculados com o fechamento diário e executados logo "
        "em seguida (o mercado cripto não fecha), capturando só os retornos dos dias seguintes; sem alavancagem nem venda "
        "a descoberto; impostos não considerados."
    )
    disclaimer()
