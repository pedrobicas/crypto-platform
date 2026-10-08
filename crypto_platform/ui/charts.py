"""Construtores de gráficos Plotly usados pelas páginas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import CURRENCIES
from ..formatting import price_decimals
from . import theme as T


def _fig(height: int = 420, **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=T.TEMPLATE_NAME, height=height, **layout)
    return fig


def _price_fmt(series: pd.Series) -> str:
    ref = float(series.dropna().median()) if len(series.dropna()) else 1.0
    return ",.0f" if ref >= 1000 else f",.{price_decimals(ref)}f"


DATE_TICKS = [
    dict(dtickrange=[None, 86_400_000], value="%d/%m %Hh"),
    dict(dtickrange=[86_400_000, "M1"], value="%d/%m"),
    dict(dtickrange=["M1", "M12"], value="%m/%Y"),
    dict(dtickrange=["M12", None], value="%Y"),
]


def _dates(fig: go.Figure) -> None:
    """Eixo de datas em formato numérico brasileiro (o Plotly não traz meses em português)."""
    fig.update_xaxes(showgrid=False, tickformatstops=DATE_TICKS, hoverformat="%d/%m/%Y")


def _price_hover(series: pd.Series) -> str:
    ref = float(series.dropna().median()) if len(series.dropna()) else 1.0
    return f",.{price_decimals(ref)}f"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def range_buttons(fig: go.Figure, row_axis: str = "xaxis") -> None:
    fig.update_layout(
        {
            row_axis: dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1a", step="year", stepmode="backward"),
                        dict(step="all", label="tudo"),
                    ],
                    bgcolor=T.PANEL,
                    activecolor=T.GRID,
                    font=dict(color=T.TEXT, size=11),
                    x=0,
                    y=1.0,
                    yanchor="bottom",
                ),
            )
        }
    )


# ---------------------------------------------------------------------------
# Preço e análise técnica
# ---------------------------------------------------------------------------

OVERLAYS = {
    "Média 20": ("sma_20", T.PERI),
    "Média 50": ("sma_50", T.CYAN),
    "Média 200": ("sma_200", T.LILAC),
    "MME 12": ("ema_12", T.SAND),
    "MME 26": ("ema_26", T.SLATE),
}

PANELS = ("Volume", "RSI", "MACD", "Estocástico", "ADX", "OBV", "Williams %R", "Aroon", "ATR %")


def _add_price(fig: go.Figure, ind: pd.DataFrame, has_ohlc: bool, row=None, col=None, name="Preço") -> None:
    kw = dict(row=row, col=col) if row else {}
    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=ind.index,
                open=ind["open"],
                high=ind["high"],
                low=ind["low"],
                close=ind["close"],
                name=name,
                increasing=dict(line=dict(color=T.UP, width=1), fillcolor=T.UP),
                decreasing=dict(line=dict(color=T.DOWN, width=1), fillcolor=T.DOWN),
                showlegend=False,
            ),
            **kw,
        )
    else:
        fig.add_trace(
            go.Scatter(x=ind.index, y=ind["close"], name=name, mode="lines", line=dict(color=T.GOLD, width=2), showlegend=False),
            **kw,
        )


def technical_chart(
    ind: pd.DataFrame,
    has_ohlc: bool,
    overlays: list[str],
    panels: list[str],
    show_bollinger: bool = False,
    sr: dict | None = None,
    fib: dict | None = None,
    height: int | None = None,
) -> go.Figure:
    panels = [p for p in panels if p in PANELS and (p not in ("Volume", "OBV") or "volume" in ind.columns)]
    rows = 1 + len(panels)
    heights = [0.58] + [0.42 / len(panels)] * len(panels) if panels else [1.0]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025, row_heights=heights)
    fig.update_layout(template=T.TEMPLATE_NAME, height=height or (430 + 140 * len(panels)), xaxis_rangeslider_visible=False)

    if show_bollinger and "bb_upper" in ind:
        fig.add_trace(
            go.Scatter(x=ind.index, y=ind["bb_upper"], name="Bollinger sup.", line=dict(color=T.SLATE, width=1), hoverinfo="skip"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=ind.index,
                y=ind["bb_lower"],
                name="Bollinger inf.",
                line=dict(color=T.SLATE, width=1),
                fill="tonexty",
                fillcolor=_rgba(T.SLATE, 0.08),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    _add_price(fig, ind, has_ohlc, row=1, col=1)
    for label in overlays:
        col_name, color = OVERLAYS.get(label, (None, None))
        if col_name and col_name in ind:
            fig.add_trace(go.Scatter(x=ind.index, y=ind[col_name], name=label, line=dict(color=color, width=1.4)), row=1, col=1)

    if sr:
        for lv in sr.get("supports", []):
            fig.add_hline(
                y=lv["price"],
                line=dict(color=T.UP, width=1, dash="dash"),
                opacity=0.7,
                row=1,
                col=1,
                annotation_text=f"S ({lv['touches']}x)",
                annotation_font=dict(color=T.UP, size=10),
                annotation_position="right",
            )
        for lv in sr.get("resistances", []):
            fig.add_hline(
                y=lv["price"],
                line=dict(color=T.DOWN, width=1, dash="dash"),
                opacity=0.7,
                row=1,
                col=1,
                annotation_text=f"R ({lv['touches']}x)",
                annotation_font=dict(color=T.DOWN, size=10),
                annotation_position="right",
            )
    if fib:
        for ratio, level in fib["levels"].items():
            fig.add_hline(
                y=level,
                line=dict(color=T.GOLD, width=0.8, dash="dot"),
                opacity=0.55,
                row=1,
                col=1,
                annotation_text=f"Fib {ratio * 100:.1f}%".replace(".", ","),
                annotation_font=dict(color=T.GOLD, size=9),
                annotation_position="left",
            )

    for i, panel in enumerate(panels, start=2):
        if panel == "Volume":
            colors = np.where(ind["close"].diff().fillna(0) >= 0, _rgba(T.UP, 0.55), _rgba(T.DOWN, 0.55))
            fig.add_trace(go.Bar(x=ind.index, y=ind["volume"], name="Volume", marker_color=colors, marker_line_width=0), row=i, col=1)
            if "volume_sma" in ind:
                fig.add_trace(
                    go.Scatter(x=ind.index, y=ind["volume_sma"], name="Volume (média 20)", line=dict(color=T.GOLD, width=1)), row=i, col=1
                )
        elif panel == "RSI":
            fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi"], name="RSI", line=dict(color=T.LILAC, width=1.4)), row=i, col=1)
            fig.add_hrect(y0=30, y1=70, fillcolor=_rgba(T.LILAC, 0.06), line_width=0, row=i, col=1)
            for y, c in ((70, T.DOWN), (30, T.UP)):
                fig.add_hline(y=y, line=dict(color=c, width=1, dash="dot"), row=i, col=1)
            fig.update_yaxes(range=[0, 100], row=i, col=1)
        elif panel == "MACD":
            hist_colors = np.where(ind["macd_hist"].fillna(0) >= 0, _rgba(T.UP, 0.6), _rgba(T.DOWN, 0.6))
            fig.add_trace(
                go.Bar(x=ind.index, y=ind["macd_hist"], name="Histograma", marker_color=hist_colors, marker_line_width=0), row=i, col=1
            )
            fig.add_trace(go.Scatter(x=ind.index, y=ind["macd"], name="MACD", line=dict(color=T.PERI, width=1.3)), row=i, col=1)
            fig.add_trace(go.Scatter(x=ind.index, y=ind["macd_signal"], name="Sinal", line=dict(color=T.GOLD, width=1.3)), row=i, col=1)
        elif panel == "Estocástico":
            fig.add_trace(go.Scatter(x=ind.index, y=ind["stoch_k"], name="%K", line=dict(color=T.CYAN, width=1.3)), row=i, col=1)
            fig.add_trace(go.Scatter(x=ind.index, y=ind["stoch_d"], name="%D", line=dict(color=T.GOLD, width=1.1)), row=i, col=1)
            for y, c in ((80, T.DOWN), (20, T.UP)):
                fig.add_hline(y=y, line=dict(color=c, width=1, dash="dot"), row=i, col=1)
            fig.update_yaxes(range=[0, 100], row=i, col=1)
        elif panel == "ADX":
            fig.add_trace(go.Scatter(x=ind.index, y=ind["adx"], name="ADX", line=dict(color=T.TEXT, width=1.4)), row=i, col=1)
            fig.add_trace(go.Scatter(x=ind.index, y=ind["di_plus"], name="DI+", line=dict(color=T.UP, width=1)), row=i, col=1)
            fig.add_trace(go.Scatter(x=ind.index, y=ind["di_minus"], name="DI−", line=dict(color=T.DOWN, width=1)), row=i, col=1)
            fig.add_hline(y=20, line=dict(color=T.MUTED, width=1, dash="dot"), row=i, col=1)
        elif panel == "OBV":
            fig.add_trace(go.Scatter(x=ind.index, y=ind["obv"], name="OBV", line=dict(color=T.CYAN, width=1.3)), row=i, col=1)
            fig.add_trace(go.Scatter(x=ind.index, y=ind["obv_sma"], name="OBV (média 20)", line=dict(color=T.GOLD, width=1)), row=i, col=1)
        elif panel == "Williams %R":
            fig.add_trace(
                go.Scatter(x=ind.index, y=ind["williams_r"], name="Williams %R", line=dict(color=T.SAND, width=1.3)), row=i, col=1
            )
            for y, c in ((-20, T.DOWN), (-80, T.UP)):
                fig.add_hline(y=y, line=dict(color=c, width=1, dash="dot"), row=i, col=1)
            fig.update_yaxes(range=[-100, 0], row=i, col=1)
        elif panel == "Aroon":
            fig.add_trace(go.Scatter(x=ind.index, y=ind["aroon_up"], name="Aroon alta", line=dict(color=T.UP, width=1.2)), row=i, col=1)
            fig.add_trace(
                go.Scatter(x=ind.index, y=ind["aroon_down"], name="Aroon baixa", line=dict(color=T.DOWN, width=1.2)), row=i, col=1
            )
            fig.update_yaxes(range=[0, 100], row=i, col=1)
        elif panel == "ATR %":
            fig.add_trace(
                go.Scatter(x=ind.index, y=ind["atr_pct"], name="ATR (% do preço)", line=dict(color=T.SAND, width=1.3)), row=i, col=1
            )
        fig.update_yaxes(title_text=panel, title_standoff=4, row=i, col=1)

    fig.update_yaxes(tickformat=_price_fmt(ind["close"]), hoverformat=_price_hover(ind["close"]), row=1, col=1)
    _dates(fig)
    fig.update_layout(legend=dict(y=1.0, yanchor="bottom"), margin=dict(t=30, r=60 if sr else 8))
    return fig


def overview_chart(ind: pd.DataFrame, has_ohlc: bool, overlays: list[str], height: int = 470) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])
    fig.update_layout(template=T.TEMPLATE_NAME, height=height, xaxis_rangeslider_visible=False)
    _add_price(fig, ind, has_ohlc, row=1, col=1)
    for label in overlays:
        col_name, color = OVERLAYS[label]
        if col_name in ind:
            fig.add_trace(go.Scatter(x=ind.index, y=ind[col_name], name=label, line=dict(color=color, width=1.3)), row=1, col=1)
    if "volume" in ind:
        colors = np.where(ind["close"].diff().fillna(0) >= 0, _rgba(T.UP, 0.5), _rgba(T.DOWN, 0.5))
        fig.add_trace(
            go.Bar(x=ind.index, y=ind["volume"], name="Volume", marker_color=colors, marker_line_width=0, showlegend=False), row=2, col=1
        )
    fig.update_yaxes(tickformat=_price_fmt(ind["close"]), hoverformat=_price_hover(ind["close"]), row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
    _dates(fig)
    range_buttons(fig)
    fig.update_layout(margin=dict(t=40, b=24), legend=dict(x=1, xanchor="right", y=1.0, yanchor="bottom"))
    return fig


def score_history(score: pd.Series, close: pd.Series, height: int = 300) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(template=T.TEMPLATE_NAME, height=height)
    s = score.dropna()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.clip(lower=0),
            fill="tozeroy",
            line=dict(width=0),
            fillcolor=_rgba(T.UP, 0.35),
            name="Viés de alta",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.clip(upper=0),
            fill="tozeroy",
            line=dict(width=0),
            fillcolor=_rgba(T.DOWN, 0.35),
            name="Viés de baixa",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Score técnico", line=dict(color=T.TEXT, width=1), hovertemplate="%{y:.2f}"))
    fig.add_trace(go.Scatter(x=close.index, y=close, name="Preço", line=dict(color=T.GOLD, width=1.3)), secondary_y=True)
    fig.update_yaxes(range=[-1.05, 1.05], title_text="Score", secondary_y=False)
    fig.update_yaxes(showgrid=False, tickformat=_price_fmt(close), hoverformat=_price_hover(close), secondary_y=True)
    _dates(fig)
    return fig


# ---------------------------------------------------------------------------
# Indicadores tipo "relógio"
# ---------------------------------------------------------------------------


def gauge(
    value: float,
    vmin: float,
    vmax: float,
    steps: list[tuple[float, float, str]],
    number_suffix: str = "",
    height: int = 190,
    valueformat: str = ".1f",
) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(font=dict(family=T.HEADING_FONT, size=30, color=T.TEXT), suffix=number_suffix, valueformat=valueformat),
            gauge=dict(
                axis=dict(range=[vmin, vmax], tickcolor=T.MUTED, tickfont=dict(color=T.MUTED, size=10), nticks=6),
                bar=dict(color=T.TEXT, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[dict(range=[a, b], color=c) for a, b, c in steps],
            ),
        )
    )
    fig.update_layout(template=T.TEMPLATE_NAME, height=height, margin=dict(l=24, r=24, t=16, b=0))
    return fig


def technical_gauge(score: float, height: int = 190) -> go.Figure:
    steps = [
        (-1, -0.5, _rgba(T.DOWN, 0.55)),
        (-0.5, -0.1, _rgba(T.DOWN, 0.25)),
        (-0.1, 0.1, _rgba(T.MUTED, 0.25)),
        (0.1, 0.5, _rgba(T.UP, 0.25)),
        (0.5, 1, _rgba(T.UP, 0.55)),
    ]
    return gauge(score, -1, 1, steps, height=height, valueformat="+.2f")


def risk_gauge(score: float, height: int = 190) -> go.Figure:
    steps = [(0, 2.5, _rgba(T.UP, 0.45)), (2.5, 5, _rgba(T.GOLD, 0.35)), (5, 7.5, _rgba(T.DOWN, 0.35)), (7.5, 10, _rgba(T.DOWN, 0.6))]
    return gauge(score, 0, 10, steps, height=height)


def fear_greed_gauge(value: float, height: int = 190) -> go.Figure:
    steps = [
        (0, 25, _rgba(T.DOWN, 0.55)),
        (25, 45, _rgba(T.DOWN, 0.28)),
        (45, 55, _rgba(T.MUTED, 0.25)),
        (55, 75, _rgba(T.UP, 0.28)),
        (75, 100, _rgba(T.UP, 0.55)),
    ]
    return gauge(value, 0, 100, steps, height=height, valueformat=".0f")


# ---------------------------------------------------------------------------
# Previsão
# ---------------------------------------------------------------------------


def forecast_chart(close: pd.Series, report, highlight: str, others: list[str], lookback: int = 180, height: int = 470) -> go.Figure:
    hist = close.iloc[-lookback:]
    fig = _fig(height)
    fc = report.forecasts[highlight]
    last = pd.Series([hist.iloc[-1]], index=[hist.index[-1]])
    join = lambda s: pd.concat([last, s])  # noqa: E731  liga a previsão ao último preço

    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Histórico", line=dict(color=T.TEXT, width=1.6)))
    fig.add_trace(go.Scatter(x=join(fc.upper95).index, y=join(fc.upper95), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=join(fc.lower95).index,
            y=join(fc.lower95),
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_rgba(T.GOLD, 0.12),
            name="Intervalo 95%",
            hovertemplate="95% inf.: %{y}<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=join(fc.upper80).index, y=join(fc.upper80), line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=join(fc.lower80).index,
            y=join(fc.lower80),
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_rgba(T.GOLD, 0.22),
            name="Intervalo 80%",
            hovertemplate="80% inf.: %{y}<extra></extra>",
        )
    )
    palette = [T.PERI, T.CYAN, T.LILAC, T.SAND, T.SLATE, T.UP]
    for i, name in enumerate(m for m in others if m != highlight and m in report.forecasts):
        s = join(report.forecasts[name].mean)
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, line=dict(color=palette[i % len(palette)], width=1.2, dash="dot")))
    s = join(fc.mean)
    fig.add_trace(go.Scatter(x=s.index, y=s, name=f"{highlight} (mediana)", line=dict(color=T.GOLD, width=2.6)))
    fig.add_vline(x=hist.index[-1], line=dict(color=T.MUTED, width=1, dash="dot"))
    fig.update_yaxes(tickformat=_price_fmt(close), hoverformat=_price_hover(close))
    _dates(fig)
    return fig


def model_skill_bars(metrics: pd.DataFrame, naive_name: str, height: int = 320) -> go.Figure:
    """Ganho de cada modelo sobre o passeio aleatório (mesmas janelas de teste)."""
    m = metrics.sort_values("Skill", ascending=True)

    def color(idx, skill):
        if idx == naive_name or not np.isfinite(skill) or abs(skill) < 0.05:
            return T.MUTED  # referência ou empate
        return T.UP if skill > 0 else T.DOWN

    colors = [color(idx, row["Skill"]) for idx, row in m.iterrows()]
    fig = _fig(height)
    fig.add_trace(
        go.Bar(
            y=m.index,
            x=m["Skill"],
            orientation="h",
            marker_color=colors,
            text=[(f"{v:+.1f}%".replace(".", ",") if np.isfinite(v) else "—") for v in m["Skill"]],
            textposition="outside",
            customdata=np.stack([m["MAPE"], m["Janelas"]], axis=1),
            hovertemplate="%{y}<br>Ganho: %{x:+.1f}%<br>Erro médio: %{customdata[0]:.2f}%<br>Janelas: %{customdata[1]}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=T.GOLD, width=1.4, dash="dash"))
    span = float(np.nanmax(np.abs(m["Skill"]))) if np.isfinite(m["Skill"]).any() else 1.0
    fig.update_layout(
        xaxis=dict(
            title="Redução do erro em relação ao passeio aleatório (maior é melhor)",
            ticksuffix="%",
            range=[-span * 1.35 - 1, span * 1.35 + 1],
        ),
        hovermode="closest",
        margin=dict(l=8, r=40, t=20, b=8),
    )
    fig.update_yaxes(showgrid=False, showspikes=False)
    fig.update_xaxes(showspikes=False)
    return fig


# ---------------------------------------------------------------------------
# Risco
# ---------------------------------------------------------------------------


def rolling_vol_chart(series: dict[str, pd.Series], height: int = 340) -> go.Figure:
    fig = _fig(height)
    for (name, s), color in zip(series.items(), [T.GOLD, T.PERI, T.LILAC, T.CYAN]):
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name, line=dict(color=color, width=1.5), hovertemplate="%{y:.1f}%"))
    fig.update_yaxes(title_text="Volatilidade anualizada (%)", ticksuffix="%")
    _dates(fig)
    return fig


def drawdown_chart(dd, height: int = 340) -> go.Figure:
    s = dd.series
    fig = _fig(height)
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s,
            fill="tozeroy",
            fillcolor=_rgba(T.DOWN, 0.28),
            line=dict(color=T.DOWN, width=1.2),
            name="Queda desde o topo",
            hovertemplate="%{y:.1f}%",
        )
    )
    if dd.peak_date is not None:
        end = dd.recovery_date or s.index[-1]
        fig.add_vrect(x0=dd.peak_date, x1=end, fillcolor=_rgba(T.GOLD, 0.07), line_width=0)
        fig.add_annotation(
            x=dd.trough_date,
            y=dd.max_drawdown,
            text=f"{dd.max_drawdown:.1f}%".replace(".", ","),
            showarrow=True,
            arrowcolor=T.MUTED,
            font=dict(color=T.TEXT, size=11),
            ay=28,
        )
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%")
    _dates(fig)
    fig.update_layout(showlegend=False)
    return fig


def returns_histogram(returns: pd.Series, var95: float, cvar95: float, height: int = 360) -> go.Figure:
    r = returns.dropna() * 100
    fig = _fig(height)
    fig.add_trace(
        go.Histogram(
            x=r,
            nbinsx=70,
            marker_color=_rgba(T.PERI, 0.75),
            name="Dias",
            histnorm="probability density",
            hovertemplate="Retorno: %{x:.2f}%<extra></extra>",
        )
    )
    xs = np.linspace(r.min(), r.max(), 300)
    mu, sd = r.mean(), r.std()
    pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    fig.add_trace(go.Scatter(x=xs, y=pdf, name="Distribuição normal", line=dict(color=T.GOLD, width=1.6), hoverinfo="skip"))
    fig.add_vline(
        x=var95,
        line=dict(color=T.DOWN, width=1.5, dash="dash"),
        annotation_text="VaR 95%",
        annotation_font=dict(color=T.DOWN, size=10),
        annotation_position="top left",
    )
    fig.add_vline(
        x=cvar95,
        line=dict(color=T.DOWN, width=1.5, dash="dot"),
        annotation_text="CVaR 95%",
        annotation_font=dict(color=T.DOWN, size=10),
        annotation_position="bottom left",
    )
    fig.update_layout(xaxis_title="Retorno diário (%)", yaxis_title="Densidade", hovermode="closest", bargap=0.04)
    fig.update_xaxes(ticksuffix="%", showspikes=False)
    fig.update_yaxes(showspikes=False)
    return fig


def rolling_corr_chart(corr: pd.Series, label: str, height: int = 320) -> go.Figure:
    fig = _fig(height)
    c = corr.dropna()
    fig.add_trace(
        go.Scatter(
            x=c.index,
            y=c,
            name=label,
            line=dict(color=T.CYAN, width=1.6),
            fill="tozeroy",
            fillcolor=_rgba(T.CYAN, 0.12),
            hovertemplate="%{y:.2f}",
        )
    )
    fig.update_yaxes(range=[-1, 1], title_text="Correlação (90 dias)")
    _dates(fig)
    fig.update_layout(showlegend=False)
    return fig


def risk_breakdown(components: pd.DataFrame, height: int = 220) -> go.Figure:
    c = components.iloc[::-1]
    colors = [T.UP if v < 3.5 else T.GOLD if v < 6.5 else T.DOWN for v in c["Nota (0-10)"]]
    fig = _fig(height)
    fig.add_trace(
        go.Bar(
            y=c["Fator"],
            x=c["Nota (0-10)"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}".replace(".", ",") + f"  (peso {w * 100:.0f}%)" for v, w in zip(c["Nota (0-10)"], c["Peso"])],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}/10<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 13], showgrid=False, showticklabels=False, showspikes=False),
        hovermode="closest",
        margin=dict(l=8, r=8, t=8, b=8),
    )
    fig.update_yaxes(showgrid=False, showspikes=False)
    return fig


# ---------------------------------------------------------------------------
# Simulações
# ---------------------------------------------------------------------------


def monte_carlo_fan(mc, start: pd.Timestamp, currency: str, height: int = 430) -> go.Figure:
    idx = pd.date_range(start, periods=len(mc.percentiles), freq="D")
    p = mc.percentiles.set_index(idx)
    fig = _fig(height)
    for path in mc.sample_paths[:25]:
        fig.add_trace(go.Scatter(x=idx, y=path, line=dict(color=_rgba(T.PERI, 0.16), width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=idx, y=p["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=p["p5"],
            fill="tonexty",
            fillcolor=_rgba(T.GOLD, 0.12),
            line=dict(width=0),
            name="90% dos cenários",
            hovertemplate="5%: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=idx, y=p["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=p["p25"],
            fill="tonexty",
            fillcolor=_rgba(T.GOLD, 0.25),
            line=dict(width=0),
            name="50% dos cenários",
            hovertemplate="25%: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(x=idx, y=p["p50"], line=dict(color=T.GOLD, width=2.4), name="Mediana", hovertemplate="Mediana: %{y:,.2f}<extra></extra>")
    )
    fig.add_hline(
        y=mc.investment,
        line=dict(color=T.MUTED, width=1, dash="dash"),
        annotation_text="valor investido",
        annotation_font=dict(color=T.MUTED, size=10),
        annotation_position="top left",
    )
    sym = CURRENCIES[currency].symbol
    fig.update_yaxes(title_text=f"Valor da posição ({sym})", tickformat=",.0f")
    _dates(fig)
    return fig


def final_value_histogram(mc, height: int = 340) -> go.Figure:
    v = mc.final_values
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = v[(v >= lo) & (v <= hi)]
    fig = _fig(height)
    fig.add_trace(go.Histogram(x=v[v < mc.investment], marker_color=_rgba(T.DOWN, 0.75), name="Prejuízo", nbinsx=40))
    fig.add_trace(go.Histogram(x=v[v >= mc.investment], marker_color=_rgba(T.UP, 0.75), name="Lucro", nbinsx=40))
    fig.add_vline(x=mc.investment, line=dict(color=T.TEXT, width=1.2, dash="dash"))
    fig.add_vline(
        x=mc.median_value, line=dict(color=T.GOLD, width=1.5), annotation_text="mediana", annotation_font=dict(color=T.GOLD, size=10)
    )
    fig.update_layout(barmode="overlay", bargap=0.03, xaxis_title="Valor final", yaxis_title="Nº de simulações", hovermode="closest")
    fig.update_xaxes(tickformat=",.0f", showspikes=False)
    fig.update_yaxes(showspikes=False)
    return fig


def equity_chart(
    curves: dict[str, pd.Series], log_scale: bool = False, height: int = 420, invested: pd.Series | None = None, y_title: str = "Patrimônio"
) -> go.Figure:
    fig = _fig(height)
    for i, (name, s) in enumerate(curves.items()):
        width = 2.4 if i == 0 else 1.6
        fig.add_trace(
            go.Scatter(x=s.index, y=s, name=name, line=dict(color=T.SERIES[i % len(T.SERIES)], width=width), hovertemplate="%{y:,.2f}")
        )
    if invested is not None:
        fig.add_trace(
            go.Scatter(
                x=invested.index,
                y=invested,
                name="Capital já aplicado (DCA)",
                line=dict(color=T.MUTED, width=1, dash="dot"),
                hovertemplate="%{y:,.2f}",
            )
        )
    fig.update_yaxes(type="log" if log_scale else "linear", title_text=y_title, tickformat=",.0f")
    _dates(fig)
    return fig


def drawdown_lines(curves: dict[str, pd.Series], height: int = 300) -> go.Figure:
    fig = _fig(height)
    for i, (name, s) in enumerate(curves.items()):
        dd = (s / s.cummax() - 1) * 100
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd, name=name, line=dict(color=T.SERIES[i % len(T.SERIES)], width=1.3), hovertemplate="%{y:.1f}%")
        )
    fig.update_yaxes(title_text="Queda desde o topo (%)", ticksuffix="%")
    _dates(fig)
    return fig


def heatmap(
    table: pd.DataFrame,
    title: str = "",
    colorscale: str = "RdYlGn",
    zmid: float | None = 0,
    fmt: str = ".2f",
    height: int = 380,
    zmin=None,
    zmax=None,
) -> go.Figure:
    z = table.to_numpy(dtype=float)
    text = [["" if np.isnan(v) else format(v, fmt).replace(".", ",") for v in row] for row in z]
    fig = _fig(height, title=title)
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=[str(c) for c in table.columns],
            y=[str(i) for i in table.index],
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            hoverongaps=False,
            colorbar=dict(thickness=10, tickfont=dict(color=T.MUTED, size=10), outlinewidth=0),
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(hovermode="closest")
    # "category": períodos como 30, 50, 200 viram rótulos (células de mesmo tamanho), não posições numéricas
    fig.update_xaxes(type="category", showgrid=False, showspikes=False, title_text=table.columns.name or "")
    fig.update_yaxes(type="category", showgrid=False, showspikes=False, autorange="reversed", title_text=table.index.name or "")
    return fig


def normalized_performance(prices: pd.DataFrame, height: int = 420) -> go.Figure:
    base = prices / prices.iloc[0] * 100
    fig = _fig(height)
    for i, col in enumerate(base.columns):
        fig.add_trace(
            go.Scatter(
                x=base.index,
                y=base[col],
                name=col,
                line=dict(color=T.SERIES[i % len(T.SERIES)], width=1.6),
                hovertemplate=f"{col}: %{{y:.1f}}<extra></extra>",
            )
        )
    fig.add_hline(y=100, line=dict(color=T.MUTED, width=1, dash="dot"))
    fig.update_yaxes(title_text="Base 100 no início do período", type="log")
    _dates(fig)
    return fig


def risk_return_scatter(stats: pd.DataFrame, height: int = 420) -> go.Figure:
    fig = _fig(height)
    for i, (name, row) in enumerate(stats.iterrows()):
        fig.add_trace(
            go.Scatter(
                x=[row["Volatilidade anual (%)"]],
                y=[row["Retorno anualizado (%)"]],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker=dict(size=16, color=T.SERIES[i % len(T.SERIES)], line=dict(color=T.BG, width=2)),
                name=name,
                hovertemplate=f"{name}<br>Volatilidade: %{{x:.1f}}%<br>Retorno anual: %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(showlegend=False, hovermode="closest", xaxis_title="Volatilidade anual (%)", yaxis_title="Retorno anualizado (%)")
    fig.update_xaxes(ticksuffix="%", showspikes=False)
    fig.update_yaxes(ticksuffix="%", showspikes=False, zeroline=True, zerolinecolor=T.MUTED)
    return fig


def frontier_chart(clouds: pd.DataFrame, points: dict[str, dict], assets: pd.DataFrame, height: int = 460) -> go.Figure:
    fig = _fig(height)
    fig.add_trace(
        go.Scatter(
            x=clouds["vol"],
            y=clouds["retorno"],
            mode="markers",
            name="Carteiras aleatórias",
            marker=dict(
                size=4,
                color=clouds["sharpe"],
                colorscale="Viridis",
                opacity=0.55,
                colorbar=dict(
                    title=dict(text="Sharpe", font=dict(color=T.MUTED, size=10)),
                    thickness=10,
                    tickfont=dict(color=T.MUTED, size=10),
                    outlinewidth=0,
                ),
            ),
            hovertemplate="Vol %{x:.1f}% | Ret %{y:.1f}%<extra></extra>",
        )
    )
    for name, row in assets.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Volatilidade anual (%)"]],
                y=[row["Retorno anualizado (%)"]],
                mode="markers+text",
                text=[name],
                textposition="middle right",
                marker=dict(size=8, color=T.MUTED, symbol="circle-open"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    symbols = ["star", "diamond", "square", "triangle-up", "x"]
    for i, (name, pt) in enumerate(points.items()):
        fig.add_trace(
            go.Scatter(
                x=[pt["vol"]],
                y=[pt["retorno"]],
                mode="markers",
                name=name,
                marker=dict(
                    size=16 if i == 0 else 12,
                    symbol=symbols[i % len(symbols)],
                    color=T.GOLD if i == 0 else T.SERIES[(i + 1) % len(T.SERIES)],
                    line=dict(color=T.BG, width=1.5),
                ),
                hovertemplate=f"{name}<br>Vol %{{x:.1f}}% | Ret %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(hovermode="closest", xaxis_title="Volatilidade anual (%)", yaxis_title="Retorno esperado (histórico, % a.a.)")
    fig.update_xaxes(ticksuffix="%", showspikes=False)
    fig.update_yaxes(ticksuffix="%", showspikes=False)
    return fig


def weights_bar(weights: pd.Series, contributions: pd.Series | None = None, height: int = 300) -> go.Figure:
    w = weights.sort_values(ascending=True) * 100
    fig = _fig(height)
    fig.add_trace(
        go.Bar(
            y=w.index,
            x=w,
            orientation="h",
            name="Peso no capital",
            marker_color=T.GOLD,
            text=[f"{v:.1f}%".replace(".", ",") for v in w],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra>Peso</extra>",
        )
    )
    if contributions is not None:
        c = contributions.reindex(w.index)
        fig.add_trace(
            go.Bar(
                y=c.index,
                x=c,
                orientation="h",
                name="Contribuição para o risco",
                marker_color=_rgba(T.PERI, 0.75),
                hovertemplate="%{y}: %{x:.1f}%<extra>Risco</extra>",
            )
        )
    fig.update_layout(barmode="group", hovermode="closest", xaxis=dict(ticksuffix="%", showspikes=False))
    fig.update_yaxes(showgrid=False, showspikes=False)
    return fig


def scenario_bars(table: pd.DataFrame, investment: float, height: int = 300) -> go.Figure:
    fig = _fig(height)
    colors = [T.UP if v >= investment else T.DOWN for v in table["Valor final"]]
    fig.add_trace(
        go.Bar(
            x=table.index,
            y=table["Valor final"],
            marker_color=colors,
            text=[f"{r:+.1f}%".replace(".", ",") for r in table["Retorno (%)"]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=investment, line=dict(color=T.MUTED, width=1, dash="dash"))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(tickformat=",.0f", showspikes=False)
    fig.update_xaxes(showgrid=False, showspikes=False)
    return fig


def fear_greed_history(fng: pd.DataFrame, height: int = 160) -> go.Figure:
    fig = _fig(height)
    fig.add_trace(
        go.Scatter(
            x=fng.index,
            y=fng["value"],
            line=dict(color=T.GOLD, width=1.4),
            fill="tozeroy",
            fillcolor=_rgba(T.GOLD, 0.1),
            hovertemplate="%{y:.0f}<extra></extra>",
            name="Índice",
        )
    )
    fig.add_hrect(y0=0, y1=25, fillcolor=_rgba(T.DOWN, 0.08), line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor=_rgba(T.UP, 0.08), line_width=0)
    fig.update_yaxes(range=[0, 100], showgrid=False, tickvals=[25, 50, 75])
    _dates(fig)
    fig.update_layout(showlegend=False, margin=dict(l=4, r=4, t=4, b=4))
    return fig
