"""Visão geral: preço, variações, termômetro técnico, sentimento e risco."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ... import risk_analyzer as ra
from ... import technical_indicators as ti
from ...formatting import fmt_compact, fmt_money, fmt_number, fmt_pct, fmt_price
from .. import charts, nav
from ..components import coin_header, disclaimer, pct_change, plot, verdict_html, votes_html
from ..state import compute_indicators, get_settings, load_current, load_fear_greed, load_snapshot, period_label


def _change_metric(col, label: str, close: pd.Series, days: int, currency: str) -> None:
    chg = pct_change(close, days)
    if np.isnan(chg):
        col.metric(label, "—", border=True)
        return
    diff = close.iloc[-1] - close.iloc[-1 - days]
    col.metric(
        label,
        fmt_pct(chg, signed=True),
        delta=fmt_money(diff, currency, 2 if abs(diff) >= 1 else 6),
        chart_data=close.iloc[-1 - days :].tolist(),
        chart_type="area",
        border=True,
    )


def render() -> None:
    s = get_settings()
    md = load_current(s)
    snapshot = load_snapshot((s.coin.coingecko_id,), s.currency, s.refresh_token)
    coin_header(md, snapshot)

    df = md.df
    close = md.close
    ind = compute_indicators(df)
    summary = ti.technical_summary(ind)
    risk = ra.analyze(close, s.risk_free)

    # --- Variações e métricas-chave -------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    _change_metric(c1, "Últimos 7 dias", close, 7, s.currency)
    _change_metric(c2, "Últimos 30 dias", close, 30, s.currency)
    span = len(close) - 1
    _change_metric(c3, f"No período ({period_label(span)})", close, span, s.currency)

    vol_series = ra.rolling_volatility(md.df["returns"], 30).dropna()
    vol30, vol90 = risk.volatility.get(30), risk.volatility.get(90)
    c4.metric(
        "Volatilidade 30 dias",
        fmt_pct(vol30, 1),
        delta=f"{fmt_number(vol30 - vol90, 1)} p.p. vs. 90 dias" if np.isfinite(vol90) else None,
        delta_color="inverse",
        chart_data=vol_series.tail(90).round(2).tolist(),
        chart_type="line",
        border=True,
        help="Desvio-padrão dos retornos diários, anualizado. Delta: diferença para a volatilidade de 90 dias.",
    )
    snap = snapshot.loc[s.coin.coingecko_id] if not snapshot.empty and s.coin.coingecko_id in snapshot.index else None
    if snap is not None and pd.notna(snap.get("market_cap")) and snap.get("market_cap"):
        rank = snap.get("market_cap_rank")
        c5.metric(
            "Valor de mercado",
            fmt_compact(snap["market_cap"], s.currency),
            delta=f"#{int(rank)} no ranking" if pd.notna(rank) else None,
            delta_color="off",
            delta_arrow="off",
            border=True,
        )
    elif "volume" in df.columns:
        c5.metric(
            "Volume (último dia)",
            fmt_compact(df["volume"].iloc[-1], s.currency),
            chart_data=df["volume"].tail(30).tolist(),
            chart_type="bar",
            border=True,
        )

    # --- Gráfico principal + painel lateral -----------------------------
    left, right = st.columns([2.35, 1], gap="medium")
    with left:
        overlays = st.pills(
            "Médias móveis",
            ["Média 20", "Média 50", "Média 200"],
            selection_mode="multi",
            default=["Média 50", "Média 200"],
            key="ov_overlays",
            label_visibility="collapsed",
            wrap=True,
        )
        plot(charts.overview_chart(ind, md.has_ohlc, overlays or []), key="ov_chart")

    with right:
        with st.container(border=True):
            st.markdown("##### Termômetro técnico")
            plot(charts.technical_gauge(summary["score"], height=170), key="ov_tech_gauge")
            st.markdown(verdict_html(summary["label"], summary["score"]), unsafe_allow_html=True)
            st.markdown(votes_html(summary["counts"]), unsafe_allow_html=True)
            nav.link("technical", "Ver os indicadores", ":material/arrow_forward:")

        fng = load_fear_greed(s.refresh_token)
        with st.container(border=True):
            st.markdown("##### Sentimento do mercado")
            if fng.empty:
                st.caption("Índice Fear & Greed indisponível no momento.")
            else:
                last = fng.iloc[-1]
                plot(charts.fear_greed_gauge(float(last["value"]), height=150), key="ov_fng_gauge")
                week = fng["value"].iloc[-8] if len(fng) > 8 else np.nan
                st.markdown(
                    f"**{last['classification']}**"
                    + (f" <span class='muted'>(há 7 dias: {fmt_number(week, 0)})</span>" if np.isfinite(week) else ""),
                    unsafe_allow_html=True,
                )
                st.caption("Índice Fear & Greed do mercado cripto (alternative.me), de 0 a 100.")

    # --- Risco e destaques ---------------------------------------------------
    st.markdown("#### Destaques do período")
    d1, d2, d3, d4, d5 = st.columns(5)
    hi = df["high"] if "high" in df.columns else close
    lo = df["low"] if "low" in df.columns else close
    d1.metric(
        "Máxima do período", fmt_price(hi.max(), s.currency), delta=f"em {hi.idxmax():%d/%m/%Y}", delta_color="off", delta_arrow="off"
    )
    d2.metric(
        "Mínima do período", fmt_price(lo.min(), s.currency), delta=f"em {lo.idxmin():%d/%m/%Y}", delta_color="off", delta_arrow="off"
    )
    if snap is not None and pd.notna(snap.get("ath")):
        d3.metric(
            "Distância da máxima histórica",
            fmt_pct(snap.get("ath_change_percentage"), 1),
            delta=f"recorde: {fmt_price(snap['ath'], s.currency)}",
            delta_color="off",
            delta_arrow="off",
        )
    else:
        d3.metric("Queda desde o topo do período", fmt_pct(risk.drawdown.current_drawdown, 1))
    d4.metric(
        "Pior queda no período",
        fmt_pct(risk.drawdown.max_drawdown, 1),
        delta="ainda não recuperada" if risk.drawdown.recovery_date is None and risk.drawdown.peak_date is not None else "recuperada",
        delta_color="off",
        delta_arrow="off",
        help="Drawdown máximo: maior queda do topo ao fundo dentro do período analisado.",
    )
    d5.metric(
        "Score de risco",
        f"{fmt_number(risk.score['score'], 1)}/10",
        delta=risk.score["level"],
        delta_color="off",
        delta_arrow="off",
        help="Combina volatilidade, drawdown, VaR e Sharpe numa escala calibrada para criptoativos.",
    )
    nav.link("risk", "Entender o risco em detalhe", ":material/arrow_forward:")

    with st.expander("Qualidade dos dados e download", icon=":material/dataset:"):
        q = md.quality
        st.markdown(
            f"- **{q.points}** dias de {q.start:%d/%m/%Y} a {q.end:%d/%m/%Y}\n"
            f"- Dias sem dado preenchidos com o último preço: **{q.filled_days}**\n"
            f"- Registros inválidos descartados: **{q.dropped_rows}**\n"
            f"- Movimentos extremos sinalizados (dados mantidos): " + (", ".join(q.outlier_dates) if q.outlier_dates else "nenhum")
        )
        if not md.has_ohlc:
            st.caption(
                "Esta fonte fornece apenas o preço de fechamento: ATR, Estocástico, ADX e Williams %R usam o fechamento "
                "como aproximação de máxima e mínima. Para candles completos, escolha Yahoo Finance como fonte."
            )
        st.download_button(
            "Baixar dados e indicadores (CSV)",
            ind.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
            file_name=f"{s.coin.symbol.lower()}_{s.currency}_{s.days}d.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    disclaimer()
