"""Métricas de risco: volatilidade, VaR/CVaR, drawdown, índices de desempenho,
risco de cauda, beta/correlação e um score de risco explicável."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .config import PERIODS_PER_YEAR

N = PERIODS_PER_YEAR


# ---------------------------------------------------------------------------
# Métricas básicas
# ---------------------------------------------------------------------------


def annualized_return(returns: pd.Series) -> float:
    """CAGR a partir de retornos simples diários."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = float((1 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (N / len(r)) - 1


def annualized_volatility(returns: pd.Series) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(N)) if len(r) > 1 else float("nan")


def volatility_windows(returns: pd.Series, windows=(7, 30, 90, 365)) -> dict[int, float]:
    """Volatilidade anualizada (%) dos últimos ``w`` dias para cada janela."""
    r = returns.dropna()
    return {w: float(r.tail(w).std(ddof=1) * np.sqrt(N) * 100) if len(r) >= w else float("nan") for w in windows}


def rolling_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * np.sqrt(N) * 100


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidade EWMA (RiskMetrics), anualizada em %. Reage mais rápido a choques."""
    r = returns.dropna()
    var = (r**2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * N) * 100


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free / N
    sd = excess.std(ddof=1)
    # tolerância: o desvio de uma série constante em ponto flutuante não é exatamente zero
    return float(excess.mean() / sd * np.sqrt(N)) if sd > 1e-12 else float("nan")


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Sortino com desvio de perdas correto: sqrt(média(min(r - rf, 0)²))."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free / N
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    return float(excess.mean() / downside * np.sqrt(N)) if downside > 1e-12 else float("nan")


@dataclass
class Drawdown:
    max_drawdown: float  # % (negativo)
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    recovery_date: pd.Timestamp | None  # None = ainda não recuperou
    duration_days: int  # do pico até a recuperação (ou até hoje)
    current_drawdown: float  # %
    series: pd.Series = field(repr=False)  # % ao longo do tempo


def drawdown_analysis(prices: pd.Series) -> Drawdown:
    p = prices.dropna()
    running_max = p.cummax()
    dd = (p / running_max - 1) * 100
    trough = dd.idxmin()
    mdd = float(dd.min())
    if mdd >= 0:
        return Drawdown(0.0, None, None, None, 0, float(dd.iloc[-1]), dd)
    before = p.loc[:trough]
    peak = before[before == before.max()].index[-1]  # último toque no topo antes do fundo
    after = p.loc[trough:]
    recovered = after[after >= p.loc[peak]]
    recovery = recovered.index[0] if len(recovered) else None
    end = recovery if recovery is not None else p.index[-1]
    return Drawdown(mdd, peak, trough, recovery, int((end - peak).days), float(dd.iloc[-1]), dd)


def calmar_ratio(returns: pd.Series, max_drawdown_pct: float) -> float:
    if not max_drawdown_pct or np.isnan(max_drawdown_pct):
        return float("nan")
    return annualized_return(returns) / abs(max_drawdown_pct / 100)


def var_cvar(returns: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    """VaR/CVaR diários em % (valores negativos = perda).

    - histórico: quantil empírico;
    - paramétrico (normal);
    - Cornish-Fisher: ajusta o quantil normal pela assimetria e curtose observadas.
    """
    r = returns.dropna()
    alpha = 1 - confidence
    hist_var = float(np.quantile(r, alpha))
    tail = r[r <= hist_var]
    hist_cvar = float(tail.mean()) if len(tail) else hist_var
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = stats.norm.ppf(alpha)
    s, k = float(stats.skew(r)), float(stats.kurtosis(r))  # curtose em excesso
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return {
        "historical_var": hist_var * 100,
        "historical_cvar": hist_cvar * 100,
        "normal_var": (mu + z * sd) * 100,
        "cornish_fisher_var": (mu + z_cf * sd) * 100,
    }


def tail_risk(returns: pd.Series) -> dict:
    r = returns.dropna()
    jb = stats.jarque_bera(r)
    sd = r.std(ddof=1)
    extreme = r[(r - r.mean()).abs() > 3 * sd]
    expected_normal = 2 * stats.norm.sf(3) * 100
    return {
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "jarque_bera_pvalue": float(jb.pvalue),
        "is_normal": bool(jb.pvalue > 0.05),
        "extreme_days": int(len(extreme)),
        "extreme_freq_pct": float(len(extreme) / len(r) * 100) if len(r) else float("nan"),
        "extreme_freq_normal_pct": float(expected_normal),
        "best_day": float(r.max() * 100),
        "best_day_date": r.idxmax(),
        "worst_day": float(r.min() * 100),
        "worst_day_date": r.idxmin(),
        "positive_days_pct": float((r > 0).mean() * 100),
    }


def beta_correlation(asset_returns: pd.Series, market_returns: pd.Series, window: int = 90) -> dict:
    data = pd.concat([asset_returns, market_returns], axis=1, keys=["a", "m"], sort=True).dropna()
    if len(data) < 30:
        return {"beta": float("nan"), "correlation": float("nan"), "r_squared": float("nan"), "rolling_corr": pd.Series(dtype=float)}
    cov = np.cov(data["a"], data["m"], ddof=1)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else float("nan")
    corr = float(data["a"].corr(data["m"]))
    rolling = data["a"].rolling(window, min_periods=window).corr(data["m"])
    return {"beta": float(beta), "correlation": corr, "r_squared": corr**2, "rolling_corr": rolling}


# ---------------------------------------------------------------------------
# Score de risco explicável
# ---------------------------------------------------------------------------

# (nome, peso, valor "sem risco", valor "risco máximo")
SCORE_COMPONENTS = (
    ("Volatilidade 30d (anual.)", 0.30, 0.0, 120.0),
    ("Drawdown máximo", 0.25, 0.0, 80.0),
    ("VaR 95% diário", 0.25, 0.0, 8.0),
    ("Sharpe (retorno/risco)", 0.20, 2.0, -1.0),
)


def risk_score(vol_30d: float, max_dd: float, var_95: float, sharpe: float) -> dict:
    """Score 0-10 (escala calibrada para criptoativos) com a contribuição de cada fator."""
    raw = {
        "Volatilidade 30d (anual.)": vol_30d,
        "Drawdown máximo": abs(max_dd),
        "VaR 95% diário": abs(min(var_95, 0.0)),
        "Sharpe (retorno/risco)": sharpe,
    }
    parts = []
    total = 0.0
    for name, weight, best, worst in SCORE_COMPONENTS:
        value = raw[name]
        if value is None or np.isnan(value):
            sub = 5.0
        else:
            sub = float(np.clip((value - best) / (worst - best) * 10, 0, 10))
        total += weight * sub
        parts.append({"Fator": name, "Valor": value, "Nota (0-10)": sub, "Peso": weight, "Contribuição": weight * sub})
    return {"score": total, "level": risk_level(total), "components": pd.DataFrame(parts)}


def risk_level(score: float) -> str:
    if score <= 2.5:
        return "Baixo"
    if score <= 5.0:
        return "Moderado"
    if score <= 7.5:
        return "Alto"
    return "Muito alto"


# ---------------------------------------------------------------------------
# Relatório completo
# ---------------------------------------------------------------------------


@dataclass
class RiskReport:
    volatility: dict[int, float]
    ewma_vol: float
    var95: dict[str, float]
    var99: dict[str, float]
    drawdown: Drawdown
    sharpe: float
    sortino: float
    calmar: float
    cagr: float
    ann_vol: float
    tail: dict
    score: dict
    risk_free: float


def analyze(prices: pd.Series, risk_free: float = 0.0) -> RiskReport:
    returns = prices.pct_change().dropna()
    vol = volatility_windows(returns)
    dd = drawdown_analysis(prices)
    v95 = var_cvar(returns, 0.95)
    sharpe = sharpe_ratio(returns, risk_free)
    score = risk_score(vol.get(30, np.nan), dd.max_drawdown, v95["historical_var"], sharpe)
    ewma = ewma_volatility(returns)
    return RiskReport(
        volatility=vol,
        ewma_vol=float(ewma.iloc[-1]) if len(ewma) else float("nan"),
        var95=v95,
        var99=var_cvar(returns, 0.99),
        drawdown=dd,
        sharpe=sharpe,
        sortino=sortino_ratio(returns, risk_free),
        calmar=calmar_ratio(returns, dd.max_drawdown),
        cagr=annualized_return(returns),
        ann_vol=annualized_volatility(returns),
        tail=tail_risk(returns),
        score=score,
        risk_free=risk_free,
    )


def position_size(capital: float, risk_per_trade_pct: float, stop_distance_pct: float) -> dict:
    """Dimensionamento por risco fixo: quanto alocar para que, se o stop for
    atingido, a perda seja ``risk_per_trade_pct`` do capital."""
    if stop_distance_pct <= 0 or capital <= 0:
        return {"position_value": 0.0, "position_pct": 0.0, "max_loss": 0.0}
    max_loss = capital * risk_per_trade_pct / 100
    value = min(capital, max_loss / (stop_distance_pct / 100))
    return {"position_value": value, "position_pct": value / capital * 100, "max_loss": value * stop_distance_pct / 100}


class RiskAnalyzer:
    """Wrapper orientado a objeto (compatibilidade com a versão anterior)."""

    def __init__(self, risk_free: float = 0.0):
        self.risk_free = risk_free

    def calculate_risk_metrics(self, df: pd.DataFrame) -> RiskReport:
        col = "close" if "close" in df.columns else "price"
        return analyze(df[col], self.risk_free)
