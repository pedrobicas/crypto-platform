"""Previsão: vários modelos, validação walk-forward e intervalos de incerteza."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ... import forecast_model as fm
from ...formatting import fmt_number, fmt_pct, fmt_price
from .. import charts
from ..components import coin_header, disclaimer, esc, pct, plot, table
from ..state import compute_forecast, get_settings, load_current


def _metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=metrics.index)
    out["Erro médio (MAPE)"] = metrics["MAPE"]
    out["Erro no último dia"] = metrics["MAPE_final"]
    out["Acerto de direção"] = metrics["Direcao"]
    out["Cobertura do intervalo 95%"] = metrics["Cobertura95"]
    out["Ganho vs. passeio aleatório"] = metrics["Skill"]
    out["Janelas"] = metrics["Janelas"]
    out.index.name = "Modelo"
    return out.reset_index()


def render() -> None:
    s = get_settings()
    md = load_current(s)
    coin_header(md)

    with st.form("fc_form", border=True):
        c1, c2, c3 = st.columns([1.1, 2.4, 0.9])
        horizon = c1.slider("Horizonte (dias)", 7, 90, 30, key="fc_horizon")
        models = c2.multiselect(
            "Modelos",
            [m for m in fm.ALL_MODELS if m != fm.NAIVE],
            default=[m for m in fm.DEFAULT_MODELS if m != fm.NAIVE],
            key="fc_models",
            help="O passeio aleatório é sempre incluído como referência.",
        )
        folds = c3.number_input("Janelas de teste", 3, 12, 6, key="fc_folds", help="Quantas vezes cada modelo é testado no passado.")
        submitted = st.form_submit_button("Gerar previsão", type="primary", icon=":material/play_arrow:")

    if submitted or "fc_params" not in st.session_state:
        st.session_state["fc_params"] = (int(horizon), tuple([fm.NAIVE, *models]), int(folds))
    horizon, models_t, folds = st.session_state["fc_params"]

    try:
        with st.spinner("Treinando e validando os modelos (alguns segundos na primeira vez)…"):
            rep = compute_forecast(md.close, horizon, models_t, folds)
    except ValueError as exc:
        st.warning(str(exc), icon=":material/warning:")
        st.stop()

    for w in rep.warnings:
        st.caption(f"Aviso: {w}")

    n_windows = int(rep.metrics.attrs.get("total_windows", rep.metrics["Janelas"].max() if not rep.metrics.empty else 0))
    if n_windows < fm.MIN_WINDOWS:
        st.warning(
            f"Só foi possível testar os modelos em **{n_windows} janela(s)** — pouco para comparar. "
            "Aumente o histórico na barra lateral ou reduza o horizonte.",
            icon=":material/warning:",
        )
    elif not rep.beats_naive:
        st.warning(
            f"**Nenhum modelo superou o passeio aleatório** nas {n_windows} janelas de teste. É o resultado mais comum em "
            "cripto: no curto prazo os preços se comportam quase como um passeio aleatório. Leia a previsão pelos "
            "intervalos de incerteza, não pela linha central.",
            icon=":material/balance:",
        )
    else:
        best = rep.beats_naive[0]
        st.info(
            f"**{best}** teve o menor erro e superou o passeio aleatório em "
            f"{fmt_number(rep.metrics.loc[best, 'Skill'], 1)}% nas {n_windows} janelas de teste. Vantagens pequenas costumam "
            "não se repetir — trate como indício, não como certeza.",
            icon=":material/insights:",
        )

    options = list(rep.forecasts)
    default = fm.ENSEMBLE if fm.ENSEMBLE in options else rep.best_model
    highlight = st.segmented_control("Destacar no gráfico", options, default=default, required=True, key="fc_highlight") or default
    fc = rep.forecasts[highlight]

    final = float(fc.mean.iloc[-1])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"Mediana prevista em {horizon} dias",
        fmt_price(final, s.currency),
        delta=fmt_pct((final / rep.last_price - 1) * 100, 2, signed=True),
        border=True,
        help="Metade dos cenários do modelo fica acima e metade abaixo deste valor.",
    )
    k2.metric(
        "Faixa provável (80%)",
        f"{fmt_pct((fc.lower80.iloc[-1] / rep.last_price - 1) * 100, 1, True)} a {fmt_pct((fc.upper80.iloc[-1] / rep.last_price - 1) * 100, 1, True)}",
        delta=esc(f"{fmt_price(fc.lower80.iloc[-1], s.currency)} a {fmt_price(fc.upper80.iloc[-1], s.currency)}"),
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    k3.metric(
        "Faixa ampla (95%)",
        f"{fmt_pct((fc.lower95.iloc[-1] / rep.last_price - 1) * 100, 1, True)} a {fmt_pct((fc.upper95.iloc[-1] / rep.last_price - 1) * 100, 1, True)}",
        delta=esc(f"{fmt_price(fc.lower95.iloc[-1], s.currency)} a {fmt_price(fc.upper95.iloc[-1], s.currency)}"),
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    naive_mape = rep.metrics.loc[fm.NAIVE, "MAPE"] if fm.NAIVE in rep.metrics.index else float("nan")
    k4.metric(
        "Melhor no backtest",
        fm.SHORT_NAMES.get(rep.best_model, rep.best_model),
        help=f"{rep.best_model}. Ordenado pela redução do erro em relação ao passeio aleatório nas mesmas janelas.",
        delta=(
            f"{fmt_pct(rep.metrics.loc[rep.best_model, 'Skill'], 1, True)} vs. referência"
            if rep.best_model != fm.NAIVE
            else f"nenhum superou (erro {fmt_pct(naive_mape, 1)})"
        ),
        delta_color="off",
        delta_arrow="off",
        border=True,
    )

    c_lb, c_all = st.columns([3, 1], vertical_alignment="bottom")
    lookback = c_lb.select_slider("Histórico exibido", [60, 90, 180, 365], value=180, format_func=lambda d: f"{d} dias", key="fc_lookback")
    show_all = c_all.toggle("Mostrar os outros modelos", value=False, key="fc_show_all")
    others = [m for m in options if m != highlight] if show_all else []
    plot(charts.forecast_chart(md.close, rep, highlight, others, lookback=lookback), key="fc_chart")

    st.markdown("#### Como cada modelo se saiu em dados que não viu")
    table(
        _metrics_table(rep.metrics),
        {
            "Erro médio (MAPE)": pct(2),
            "Erro no último dia": pct(2),
            "Acerto de direção": pct(0),
            "Cobertura do intervalo 95%": pct(0),
            "Ganho vs. passeio aleatório": pct(1, signed=True),
        },
        column_config={
            "Modelo": st.column_config.TextColumn(width="medium"),
            "Erro médio (MAPE)": st.column_config.Column(help="Erro percentual médio em todos os dias previstos."),
            "Erro no último dia": st.column_config.Column(help="Erro no dia final do horizonte."),
            "Acerto de direção": st.column_config.Column(
                help="Acertou se o preço subiria ou cairia até o fim do horizonte. 50% = cara ou coroa."
            ),
            "Cobertura do intervalo 95%": st.column_config.Column(
                help="Quanto do preço real caiu dentro do intervalo de 95%. Ideal: perto de 95%."
            ),
            "Ganho vs. passeio aleatório": st.column_config.Column(
                help="Redução do erro em relação ao passeio aleatório. Negativo = pior que a referência."
            ),
        },
    )
    st.caption(
        f"Validação walk-forward: {n_windows} previsões de {horizon} dias feitas em datas passadas, cada uma treinada "
        "apenas com dados anteriores àquela data."
    )
    plot(charts.model_skill_bars(rep.metrics, fm.NAIVE, height=300), key="fc_err_bars")

    with st.expander("Diagnóstico estatístico", icon=":material/science:"):
        d = rep.diagnostics
        adf_p = d.get("adf_log_price_p")
        adf_r = d.get("adf_returns_p")
        lb = d.get("ljung_box_returns_p")
        arima = d.get("arima", {})
        lines = []
        if adf_p is not None:
            lines.append(
                f"- **Teste ADF no log do preço:** p = {fmt_number(adf_p, 3)} → "
                + (
                    "não rejeita raiz unitária: o preço não volta a uma média fixa."
                    if adf_p > 0.05
                    else "série estacionária (incomum para preços)."
                )
            )
        if adf_r is not None:
            lines.append(
                f"- **Teste ADF nos retornos:** p = {fmt_number(adf_r, 4)} → "
                + ("retornos estacionários." if adf_r <= 0.05 else "retornos não estacionários.")
            )
        if lb is not None:
            lines.append(
                f"- **Ljung-Box nos retornos (10 defasagens):** p = {fmt_number(lb, 3)} → "
                + (
                    "sem autocorrelação significativa — o passado recente pouco ajuda a prever o próximo dia."
                    if lb > 0.05
                    else "há autocorrelação — espaço para modelos como ARIMA."
                )
            )
        if arima:
            order = arima.get("ordem")
            lines.append(
                f"- **ARIMA escolhido por AICc:** ordem {order}{' com tendência' if arima.get('drift') else ''}"
                + (" — equivale ao passeio aleatório." if tuple(order or ()) == (0, 1, 0) and not arima.get("drift") else ".")
            )
        st.markdown("\n".join(lines) or "Sem diagnósticos disponíveis.")

    with st.expander("O que é cada modelo", icon=":material/help:"):
        for name in options:
            st.markdown(f"- **{name}:** {fm.MODEL_DESCRIPTIONS.get(name, '')}")

    export = pd.DataFrame({name: f.mean for name, f in rep.forecasts.items()})
    export[f"{highlight} (inf. 95%)"] = fc.lower95
    export[f"{highlight} (sup. 95%)"] = fc.upper95
    st.download_button(
        "Baixar previsões (CSV)",
        export.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"previsao_{s.coin.symbol.lower()}_{horizon}d.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    disclaimer()
