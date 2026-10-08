"""Análise multiativo: desempenho comparado, correlação, otimização de carteira
(média-variância com covariância encolhida de Ledoit-Wolf) e backtest com rebalanceamento."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import DEFAULT_FEE, DEFAULT_SLIPPAGE, PERIODS_PER_YEAR
from .risk_analyzer import annualized_return, sharpe_ratio, sortino_ratio

N = PERIODS_PER_YEAR

EQUAL = "Pesos iguais"
MIN_VAR = "Mínima variância"
MAX_SHARPE = "Máximo Sharpe"
RISK_PARITY = "Paridade de risco"
MANUAL = "Manual"
METHODS = (EQUAL, MIN_VAR, MAX_SHARPE, RISK_PARITY, MANUAL)

METHOD_HELP = {
    EQUAL: "Divide o capital igualmente. Simples e difícil de bater fora da amostra.",
    MIN_VAR: "Busca a carteira de menor volatilidade possível, ignorando retornos esperados.",
    MAX_SHARPE: "Maximiza retorno/risco usando os retornos passados — o mais sujeito a sobreajuste.",
    RISK_PARITY: "Cada ativo contribui com a mesma parcela do risco total.",
    MANUAL: "Você define os pesos.",
}


def align_prices(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Une as séries de fechamento no período em comum."""
    df = pd.DataFrame(prices).sort_index()
    return df.dropna(how="any")


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="any")


def asset_stats(prices: pd.DataFrame, risk_free: float = 0.0) -> pd.DataFrame:
    rets = daily_returns(prices)
    rows = {}
    for col in prices.columns:
        r = rets[col]
        eq = (1 + r).cumprod()
        rows[col] = {
            "Retorno anualizado (%)": annualized_return(r) * 100,
            "Volatilidade anual (%)": float(r.std(ddof=1) * np.sqrt(N) * 100),
            "Sharpe": sharpe_ratio(r, risk_free),
            "Drawdown máximo (%)": float((eq / eq.cummax() - 1).min() * 100),
            "Retorno no período (%)": float((prices[col].iloc[-1] / prices[col].iloc[0] - 1) * 100),
        }
    return pd.DataFrame(rows).T


def covariance(rets: pd.DataFrame, shrink: bool = True) -> np.ndarray:
    """Covariância anualizada. Com ``shrink`` usa Ledoit-Wolf (mais estável com poucos dados)."""
    if shrink and len(rets) > len(rets.columns) + 2:
        from sklearn.covariance import LedoitWolf

        return LedoitWolf().fit(rets.to_numpy()).covariance_ * N
    return rets.cov().to_numpy() * N


def _expected_returns(rets: pd.DataFrame) -> np.ndarray:
    """Taxa de crescimento contínua anual (média dos retornos log × 365)."""
    return np.log1p(rets).mean().to_numpy() * N


def _simple(log_growth):
    """Converte crescimento log anual em retorno anual simples — para uma moeda isolada é
    exatamente o retorno anualizado (CAGR) mostrado nas outras abas."""
    return np.expm1(log_growth)


def optimize(prices: pd.DataFrame, method: str, max_weight: float = 1.0, risk_free: float = 0.0, manual: dict | None = None) -> pd.Series:
    cols = list(prices.columns)
    n = len(cols)
    if n == 0:
        return pd.Series(dtype=float)
    max_weight = max(max_weight, 1.0 / n + 1e-9)
    if method == EQUAL or n == 1:
        return pd.Series(np.full(n, 1 / n), index=cols)
    if method == MANUAL:
        w = np.array([max(float((manual or {}).get(c, 0.0)), 0.0) for c in cols])
        w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)
        return pd.Series(w, index=cols)

    rets = daily_returns(prices)
    cov = covariance(rets)
    mu = _expected_returns(rets)
    x0 = np.full(n, 1 / n)
    bounds = [(0.0, max_weight)] * n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)

    if method == MIN_VAR:
        obj = lambda w: w @ cov @ w  # noqa: E731
    elif method == MAX_SHARPE:

        def obj(w):
            vol = np.sqrt(max(w @ cov @ w, 1e-12))
            return -(_simple(w @ mu) - risk_free) / vol
    elif method == RISK_PARITY:

        def obj(w):
            port_var = w @ cov @ w
            rc = w * (cov @ w) / max(port_var, 1e-12)
            return float(((rc - 1 / n) ** 2).sum())
    else:
        raise ValueError(f"Método desconhecido: {method}")

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x if res.success else x0, 0, None)
    w = w / w.sum()
    w[w < 1e-4] = 0.0
    return pd.Series(w / w.sum(), index=cols)


def portfolio_point(weights: pd.Series, prices: pd.DataFrame, risk_free: float = 0.0) -> dict:
    rets = daily_returns(prices)
    cov = covariance(rets)
    mu = _expected_returns(rets)
    w = weights.reindex(prices.columns).fillna(0).to_numpy()
    vol = float(np.sqrt(w @ cov @ w))
    ret = float(_simple(w @ mu))
    return {"retorno": ret * 100, "vol": vol * 100, "sharpe": (ret - risk_free) / vol if vol > 0 else np.nan}


def random_portfolios(prices: pd.DataFrame, n: int = 3000, risk_free: float = 0.0, seed: int = 7) -> pd.DataFrame:
    """Carteiras aleatórias (Dirichlet) para desenhar a fronteira eficiente."""
    rets = daily_returns(prices)
    cov = covariance(rets)
    mu = _expected_returns(rets)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.full(len(prices.columns), 0.7), size=n)
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    ret = _simple(W @ mu)
    return pd.DataFrame({"vol": vol * 100, "retorno": ret * 100, "sharpe": (ret - risk_free) / vol})


def risk_contributions(weights: pd.Series, prices: pd.DataFrame) -> pd.Series:
    cov = covariance(daily_returns(prices))
    w = weights.reindex(prices.columns).fillna(0).to_numpy()
    var = w @ cov @ w
    return pd.Series(w * (cov @ w) / var * 100 if var > 0 else np.zeros(len(w)), index=prices.columns)


REBALANCE_RULES = {"Nunca (comprar e segurar)": None, "Mensal": "MS", "Trimestral": "QS"}


@dataclass
class PortfolioBacktest:
    equity: pd.Series
    weights: pd.DataFrame  # pesos efetivos ao longo do tempo
    metrics: dict
    turnover_cost: float


def backtest_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    rebalance: str | None = "MS",
    initial: float = 1000.0,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    risk_free: float = 0.0,
) -> PortfolioBacktest:
    """Simula a carteira dia a dia, rebalanceando no primeiro dia de cada período."""
    cols = list(prices.columns)
    target = weights.reindex(cols).fillna(0).to_numpy()
    rets = prices.pct_change().fillna(0.0).to_numpy()
    dates = prices.index
    if rebalance:
        periods = dates.to_period("M" if rebalance == "MS" else "Q")
        rebal_flags = np.r_[False, periods[1:] != periods[:-1]]
    else:
        rebal_flags = np.zeros(len(dates), dtype=bool)

    cost_rate = fee + slippage
    holdings = target * initial * (1 - cost_rate)  # compra inicial
    total_cost = initial * cost_rate
    equity = np.empty(len(dates))
    w_hist = np.empty((len(dates), len(cols)))
    for i in range(len(dates)):
        if i > 0:
            holdings = holdings * (1 + rets[i])
        value = holdings.sum()
        if rebal_flags[i] and value > 0:
            desired = target * value
            trade = np.abs(desired - holdings).sum()
            cost = trade * cost_rate
            total_cost += cost
            holdings = target * (value - cost)
            value = holdings.sum()
        equity[i] = value
        w_hist[i] = holdings / value if value > 0 else 0
    eq = pd.Series(equity, index=dates)
    r = eq.pct_change().dropna()
    dd = (eq / eq.cummax() - 1).min() * 100
    metrics = {
        "Retorno total (%)": float((eq.iloc[-1] / initial - 1) * 100),
        "Retorno anualizado (%)": annualized_return(r) * 100,
        "Volatilidade anual (%)": float(r.std(ddof=1) * np.sqrt(N) * 100),
        "Sharpe": sharpe_ratio(r, risk_free),
        "Sortino": sortino_ratio(r, risk_free),
        "Drawdown máximo (%)": float(dd),
        "Custos totais": float(total_cost),
    }
    return PortfolioBacktest(eq, pd.DataFrame(w_hist, index=dates, columns=cols), metrics, float(total_cost))
