"""Simulações de investimento.

- **Monte Carlo**: milhares de trajetórias futuras possíveis, reamostrando blocos
  de retornos históricos (preserva caudas pesadas e agrupamento de volatilidade)
  ou via movimento browniano geométrico (GBM).
- **Histórico**: "e se eu tivesse investido?" — aporte único vs. aportes
  periódicos (DCA) usando os preços reais do período.
- **Cenários do modelo de previsão**: pessimista/base/otimista a partir dos
  intervalos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_FEE, DEFAULT_SLIPPAGE

BOOTSTRAP = "bootstrap"
GBM = "gbm"
DRIFT_HISTORICAL = "historical"
DRIFT_ZERO = "zero"


@dataclass
class MonteCarloResult:
    percentiles: pd.DataFrame  # index: dia 0..h; colunas: p5, p25, p50, p75, p95
    final_values: np.ndarray
    sample_paths: np.ndarray  # algumas trajetórias para ilustrar
    investment: float
    horizon: int
    prob_loss: float
    prob_double: float
    expected_value: float
    median_value: float
    var95: float  # perda (valor positivo) no pior 5%
    cvar95: float
    p5: float
    p95: float
    method: str
    drift: str
    annual_drift_used: float
    annual_vol_used: float


def monte_carlo(
    close: pd.Series,
    investment: float,
    horizon: int,
    n_sims: int = 5000,
    method: str = BOOTSTRAP,
    drift: str = DRIFT_ZERO,
    block_size: int = 5,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Simula o valor de um investimento comprado hoje e vendido em ``horizon`` dias.

    ``drift``:
      - ``historical``: usa o retorno médio do histórico (extrapola a tendência passada);
      - ``zero``: remove a tendência (cenário neutro) — mantém apenas a volatilidade.
    """
    log_r = np.log(close.dropna()).diff().dropna().to_numpy()
    if len(log_r) < 30:
        raise ValueError("Histórico insuficiente para a simulação (mínimo de 30 dias).")
    if drift == DRIFT_ZERO:
        log_r = log_r - log_r.mean()
    rng = np.random.default_rng(seed)

    if method == GBM:
        mu, sigma = log_r.mean(), log_r.std(ddof=1)
        steps = rng.normal(mu, sigma, size=(n_sims, horizon))
    else:
        # Bootstrap circular de blocos: cada retorno histórico tem a mesma chance de ser
        # sorteado (sem o "efeito borda" que sub-amostra o início e o fim da série),
        # então o cenário neutro fica de fato centrado em zero.
        n = len(log_r)
        b = max(1, min(block_size, n // 4))
        n_blocks = int(np.ceil(horizon / b))
        starts = rng.integers(0, n, size=(n_sims, n_blocks))
        idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % n).reshape(n_sims, -1)[:, :horizon]
        steps = log_r[idx]

    entry = investment * (1 - fee) * (1 - slippage)  # custo de compra
    growth = np.exp(np.cumsum(steps, axis=1))
    paths = np.empty((n_sims, horizon + 1))
    paths[:, 0] = entry
    paths[:, 1:] = entry * growth
    exit_cost = (1 - fee) * (1 - slippage)
    final = paths[:, -1] * exit_cost

    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    pct = pd.DataFrame(q.T * exit_cost, columns=["p5", "p25", "p50", "p75", "p95"])
    pct.index.name = "dia"
    loss = investment - final
    var95 = float(np.percentile(loss, 95))
    tail = loss[loss >= var95]
    pick = rng.choice(n_sims, size=min(30, n_sims), replace=False)
    return MonteCarloResult(
        percentiles=pct,
        final_values=final,
        sample_paths=paths[pick] * exit_cost,
        investment=investment,
        horizon=horizon,
        prob_loss=float((final < investment).mean() * 100),
        prob_double=float((final >= 2 * investment).mean() * 100),
        expected_value=float(final.mean()),
        median_value=float(np.median(final)),
        var95=max(var95, 0.0),
        cvar95=max(float(tail.mean()) if len(tail) else var95, 0.0),
        p5=float(np.percentile(final, 5)),
        p95=float(np.percentile(final, 95)),
        method=method,
        drift=drift,
        annual_drift_used=float(log_r.mean() * 365 * 100),
        annual_vol_used=float(log_r.std(ddof=1) * np.sqrt(365) * 100),
    )


@dataclass
class HistoricalComparison:
    equity: pd.DataFrame  # colunas: aporte_unico, dca, investido_dca
    summary: pd.DataFrame  # linhas: estratégias
    purchases: pd.DataFrame  # compras do DCA


def lump_sum_vs_dca(
    close: pd.Series,
    total: float,
    frequency_days: int = 7,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> HistoricalComparison:
    """Compara, com preços reais, investir tudo no início vs. em parcelas iguais.

    No DCA o dinheiro ainda não aplicado fica parado (rendimento zero) e entra no
    patrimônio — assim as duas curvas são comparáveis desde o primeiro dia.
    """
    prices = close.dropna()
    if len(prices) < frequency_days * 2:
        raise ValueError("Período curto demais para a frequência escolhida.")
    cost = (1 - fee) * (1 - slippage)
    exit_cost = (1 - fee) * (1 - slippage)

    lump_units = total * cost / prices.iloc[0]
    lump_equity = lump_units * prices

    buy_dates = prices.index[::frequency_days]
    per_buy = total / len(buy_dates)
    units = pd.Series(0.0, index=prices.index)
    units.loc[buy_dates] = per_buy * cost / prices.loc[buy_dates]
    cum_units = units.cumsum()
    invested = pd.Series(0.0, index=prices.index)
    invested.loc[buy_dates] = per_buy
    cum_invested = invested.cumsum()
    dca_equity = cum_units * prices + (total - cum_invested)

    equity = pd.DataFrame({"aporte_unico": lump_equity, "dca": dca_equity, "investido_dca": cum_invested})

    def stats_row(name: str, eq: pd.Series, n_units: float, avg_price: float, n_buys: int) -> dict:
        final = float(eq.iloc[-1] - n_units * prices.iloc[-1] * (1 - exit_cost))  # custo de venda
        dd = (eq / eq.cummax() - 1).min() * 100
        return {
            "Estratégia": name,
            "Valor final": final,
            "Resultado": final - total,
            "Retorno (%)": (final / total - 1) * 100,
            "Preço médio": avg_price,
            "Compras": n_buys,
            "Pior queda do patrimônio (%)": float(dd),
        }

    dca_units = float(cum_units.iloc[-1])
    summary = pd.DataFrame(
        [
            stats_row("Aporte único", lump_equity, lump_units, float(prices.iloc[0]), 1),
            stats_row(
                "Aportes periódicos (DCA)", dca_equity, dca_units, float(total * cost / dca_units) if dca_units else np.nan, len(buy_dates)
            ),
        ]
    ).set_index("Estratégia")
    purchases = pd.DataFrame(
        {
            "Data": buy_dates,
            "Preço": prices.loc[buy_dates].to_numpy(),
            "Valor aplicado": per_buy,
            "Unidades": units.loc[buy_dates].to_numpy(),
        }
    )
    return HistoricalComparison(equity, summary, purchases)


def forecast_scenarios(
    current_price: float,
    forecast,  # forecast_model.ModelForecast
    investment: float,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> pd.DataFrame:
    """Cenários a partir de uma previsão: limite inferior 95%, mediana e limite superior 95%."""
    cost = (1 - fee) * (1 - slippage)
    units = investment * cost / current_price
    rows = []
    for label, series in (
        ("Pessimista (limite inferior 95%)", forecast.lower95),
        ("Moderado (inferior 80%)", forecast.lower80),
        ("Base (mediana prevista)", forecast.mean),
        ("Moderado (superior 80%)", forecast.upper80),
        ("Otimista (limite superior 95%)", forecast.upper95),
    ):
        price = float(series.iloc[-1])
        value = units * price * cost
        rows.append(
            {
                "Cenário": label,
                "Preço no fim": price,
                "Valor final": value,
                "Resultado": value - investment,
                "Retorno (%)": (value / investment - 1) * 100,
            }
        )
    return pd.DataFrame(rows).set_index("Cenário")


class InvestmentSimulator:
    """Wrapper orientado a objeto (compatibilidade com a versão anterior)."""

    def __init__(self, fee: float = DEFAULT_FEE, slippage: float = DEFAULT_SLIPPAGE):
        self.fee, self.slippage = fee, slippage

    def monte_carlo(self, close: pd.Series, investment: float, horizon: int, **kw) -> MonteCarloResult:
        return monte_carlo(close, investment, horizon, fee=self.fee, slippage=self.slippage, **kw)

    def lump_sum_vs_dca(self, close: pd.Series, total: float, frequency_days: int = 7) -> HistoricalComparison:
        return lump_sum_vs_dca(close, total, frequency_days, self.fee, self.slippage)
