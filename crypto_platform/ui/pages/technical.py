"""Análise técnica: gráfico configurável, leituras dos indicadores e níveis de preço."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import technical_indicators as ti
from ...formatting import fmt_number, fmt_pct, fmt_price
from .. import charts
from ..components import coin_header, disclaimer, plot, signal_style, verdict_html, votes_html
from ..state import compute_indicators, get_settings, load_current

WINDOWS = {"3 meses": 90, "6 meses": 180, "1 ano": 365, "Tudo": None}


def render() -> None:
    s = get_settings()
    md = load_current(s)
    coin_header(md)
    ind_full = compute_indicators(md.df)

    with st.container(border=True):
        c1, c2 = st.columns([1.2, 1])
        overlays = (
            c1.pills(
                "Sobre o preço",
                list(charts.OVERLAYS) + ["Bollinger", "Suporte e resistência", "Fibonacci"],
                selection_mode="multi",
                default=["Média 20", "Média 50", "Suporte e resistência"],
                key="ta_overlays",
                wrap=True,
            )
            or []
        )
        panels = (
            c2.pills(
                "Painéis abaixo do preço",
                list(charts.PANELS),
                selection_mode="multi",
                default=["Volume", "RSI", "MACD"],
                key="ta_panels",
                wrap=True,
            )
            or []
        )
        window_label = st.segmented_control("Janela exibida", list(WINDOWS), default="6 meses", required=True, key="ta_window")

    days = WINDOWS.get(window_label or "6 meses")
    ind = ind_full if days is None else ind_full.iloc[-days:]
    sr = ti.support_resistance(ind_full) if "Suporte e resistência" in overlays else None
    fib = ti.fibonacci_levels(ind) if "Fibonacci" in overlays else None
    fig = charts.technical_chart(
        ind,
        md.has_ohlc,
        [o for o in overlays if o in charts.OVERLAYS],
        panels,
        show_bollinger="Bollinger" in overlays,
        sr=sr,
        fib=fib,
    )
    plot(fig, key="ta_chart")
    if not md.has_ohlc:
        st.caption(
            "Fonte sem máxima/mínima diárias: ATR, Estocástico, ADX, Aroon e Williams %R usam o fechamento como aproximação. "
            "Selecione Yahoo Finance na barra lateral para candles completos."
        )

    summary = ti.technical_summary(ind_full)
    left, right = st.columns([1, 2.2], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown("##### Resumo técnico")
            plot(charts.technical_gauge(summary["score"], height=170), key="ta_gauge")
            st.markdown(verdict_html(summary["label"], summary["score"]), unsafe_allow_html=True)
            st.markdown(votes_html(summary["counts"]), unsafe_allow_html=True)
            g = summary["groups"]
            st.markdown(
                f"<p class='muted'>Osciladores: <b>{ti.score_label(g[ti.OSCILLATORS])}</b><br>"
                f"Médias e tendência: <b>{ti.score_label(g[ti.TREND])}</b></p>",
                unsafe_allow_html=True,
            )
    with right:
        table = summary["table"].drop(columns=["voto", "Grupo"])
        st.dataframe(
            table.style.map(signal_style, subset=["Sinal"]),
            hide_index=True,
            column_config={
                "Indicador": st.column_config.TextColumn(width="medium"),
                "Leitura": st.column_config.TextColumn(width="small"),
                "Sinal": st.column_config.TextColumn(width="small"),
                "Interpretação": st.column_config.TextColumn(width="large"),
            },
            height=460,
        )
    st.caption(
        "O resumo é a média dos votos de cada regra (+1 alta, 0 neutro, −1 baixa). Osciladores tendem a apontar reversões; "
        "médias e tendência apontam continuidade — divergência entre os grupos é comum e informativa."
    )

    tab_sr, tab_hist, tab_fib = st.tabs(["Suportes e resistências", "Histórico do resumo técnico", "Fibonacci"])
    with tab_sr:
        sr_all = ti.support_resistance(ind_full, max_levels=5)
        rows = []
        for kind, levels in (("Resistência", sr_all["resistances"]), ("Suporte", sr_all["supports"])):
            for lv in levels:
                rows.append(
                    {
                        "Tipo": kind,
                        "Nível": fmt_price(lv["price"], s.currency),
                        "Distância do preço atual": fmt_pct(lv["distance_pct"], 2, signed=True),
                        "Toques": lv["touches"],
                        "Último toque": lv["last_touch"].strftime("%d/%m/%Y"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(
                "Níveis formados por topos e fundos locais confirmados (7 dias de cada lado), agrupados quando estão a menos "
                "de cerca de um ATR de distância. Mais toques indicam um nível mais observado pelo mercado."
            )
        else:
            st.info("Nenhum nível confirmado no período.")
    with tab_hist:
        score = ti.technical_score(ind_full)
        plot(charts.score_history(score, ind_full["close"]), key="ta_score_hist")
        st.caption("Acima de zero, a maioria dos indicadores apontava alta naquele dia; abaixo, baixa.")
    with tab_fib:
        fib_all = ti.fibonacci_levels(ind)
        direction = "alta (retrações medidas a partir do topo)" if fib_all["uptrend"] else "baixa (retrações medidas a partir do fundo)"
        st.markdown(f"Última perna relevante na janela exibida: **{direction}**.")
        price = float(ind["close"].iloc[-1])
        st.dataframe(
            pd.DataFrame(
                {
                    "Nível": [f"{fmt_number(r * 100, 1)}%" for r in fib_all["levels"]],
                    "Preço": [fmt_price(v, s.currency) for v in fib_all["levels"].values()],
                    "Distância": [fmt_pct((v / price - 1) * 100, 2, signed=True) for v in fib_all["levels"].values()],
                }
            ),
            hide_index=True,
        )
    disclaimer()
