"""Backtest de estratégias técnicas simples (comprado ou fora do mercado).

Regras para evitar resultados enganosos:
- o sinal é calculado no fechamento do dia *t* e a posição só passa a valer no
  retorno do dia *t+1* (nada de olhar para o futuro);
- cada troca de posição paga taxa + slippage;
- todas as estratégias são comparadas a partir da mesma data (após o aquecimento
  dos indicadores), inclusive o "comprar e segurar".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import technical_indicators as ti
from .config import DEFAULT_FEE, DEFAULT_SLIPPAGE, PERIODS_PER_YEAR
from .risk_analyzer import annualized_return, sharpe_ratio, sortino_ratio

N = PERIODS_PER_YEAR


def _state_machine(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Posição 1 após um sinal de entrada, 0 após um sinal de saída."""
    state = pd.Series(np.nan, index=enter.index)
    state[exit_.fillna(False).astype(bool)] = 0.0
    state[enter.fillna(False).astype(bool)] = 1.0
    return state.ffill().fillna(0.0)


# --- Estratégias: recebem DataFrame com OHLC e retornam posição desejada (0/1) e
#     a quantidade de dias de aquecimento necessária.


def strat_buy_hold(df: pd.DataFrame, **_) -> tuple[pd.Series, int]:
    return pd.Series(1.0, index=df.index), 0


def strat_sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50, **_) -> tuple[pd.Series, int]:
    c = df["close"]
    f, s = ti.sma(c, fast), ti.sma(c, slow)
    return (f > s).astype(float).where(s.notna(), 0.0), slow


def strat_trend_filter(df: pd.DataFrame, period: int = 100, **_) -> tuple[pd.Series, int]:
    c = df["close"]
    m = ti.sma(c, period)
    return (c > m).astype(float).where(m.notna(), 0.0), period


def strat_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **_) -> tuple[pd.Series, int]:
    m = ti.macd(df["close"], fast, slow, signal)
    return (m["macd"] > m["macd_signal"]).astype(float).where(m["macd"].notna(), 0.0), slow + signal


def strat_rsi(df: pd.DataFrame, period: int = 14, lower: float = 30, upper: float = 70, **_) -> tuple[pd.Series, int]:
    r = ti.rsi(df["close"], period)
    return _state_machine(r < lower, r > upper), period + 1


def strat_bollinger(df: pd.DataFrame, period: int = 20, n_std: float = 2.0, **_) -> tuple[pd.Series, int]:
    bb = ti.bollinger(df["close"], period, n_std)
    c = df["close"]
    return _state_machine(c < bb["bb_lower"], c > bb["bb_middle"]), period


def strat_donchian(df: pd.DataFrame, entry: int = 20, exit: int = 10, **_) -> tuple[pd.Series, int]:
    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    c = df["close"]
    upper = high.rolling(entry).max().shift(1)
    lower = low.rolling(exit).min().shift(1)
    return _state_machine(c > upper, c < lower), entry + 1


@dataclass(frozen=True)
class StrategySpec:
    key: str
    name: str
    description: str
    func: Callable[..., tuple[pd.Series, int]]
    params: dict = field(default_factory=dict)


STRATEGIES: dict[str, StrategySpec] = {
    s.key: s
    for s in (
        StrategySpec("buy_hold", "Comprar e segurar", "Compra no início e mantém até o fim.", strat_buy_hold),
        StrategySpec(
            "sma_cross",
            "Cruzamento de médias",
            "Comprado quando a média rápida está acima da lenta.",
            strat_sma_cross,
            {"fast": 20, "slow": 50},
        ),
        StrategySpec(
            "trend_filter",
            "Filtro de tendência",
            "Comprado apenas quando o preço está acima da média de N dias.",
            strat_trend_filter,
            {"period": 100},
        ),
        StrategySpec(
            "macd", "MACD", "Comprado quando o MACD está acima da linha de sinal.", strat_macd, {"fast": 12, "slow": 26, "signal": 9}
        ),
        StrategySpec(
            "rsi",
            "Reversão pelo RSI",
            "Compra quando o RSI cai abaixo do limite inferior; vende acima do superior.",
            strat_rsi,
            {"period": 14, "lower": 30, "upper": 70},
        ),
        StrategySpec(
            "bollinger",
            "Reversão nas Bandas de Bollinger",
            "Compra abaixo da banda inferior; vende ao voltar à média.",
            strat_bollinger,
            {"period": 20, "n_std": 2.0},
        ),
        StrategySpec(
            "donchian",
            "Rompimento (canal de Donchian)",
            "Compra ao romper a máxima de N dias; vende ao perder a mínima de M dias.",
            strat_donchian,
            {"entry": 20, "exit": 10},
        ),
    )
}


@dataclass
class StrategyResult:
    key: str
    name: str
    equity: pd.Series
    position: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict


def _trades(position: pd.Series, close_prev: pd.Series, cost: float) -> pd.DataFrame:
    """Lista de operações.

    ``close_prev`` tem um elemento a mais que ``position``: ``close_prev.iloc[i]`` é o
    fechamento do dia anterior a ``position.iloc[i]`` — o preço em que a ordem
    que gerou aquela posição foi executada.
    """
    pos = position.to_numpy()
    entries, exits = [], []
    prev = 0.0
    for i, p in enumerate(pos):
        if p > prev:
            entries.append(i)
        elif p < prev:
            exits.append(i)
        prev = p
    rows = []
    for k, e in enumerate(entries):
        x = exits[k] if k < len(exits) else None
        entry_date, entry_price = close_prev.index[e], float(close_prev.iloc[e])
        exit_date = close_prev.index[x] if x is not None else close_prev.index[-1]
        exit_price = float(close_prev.iloc[x]) if x is not None else float(close_prev.iloc[-1])
        ret = (exit_price / entry_price) * (1 - cost) ** 2 - 1
        rows.append(
            {
                "Entrada": entry_date,
                "Saída": exit_date if x is not None else pd.NaT,
                "Preço de entrada": entry_price,
                "Preço de saída": exit_price,
                "Retorno (%)": ret * 100,
                "Dias": int((exit_date - entry_date).days),
                "Aberta": x is None,
            }
        )
    return pd.DataFrame(rows, columns=["Entrada", "Saída", "Preço de entrada", "Preço de saída", "Retorno (%)", "Dias", "Aberta"])


def _metrics(rets: pd.Series, equity: pd.Series, position: pd.Series, trades: pd.DataFrame, risk_free: float) -> dict:
    dd = (equity / equity.cummax() - 1) * 100
    mdd = float(dd.min())
    cagr = annualized_return(rets)
    closed = trades[~trades["Aberta"]] if not trades.empty else trades
    return {
        "Retorno total (%)": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100),
        "Retorno anualizado (%)": cagr * 100,
        "Volatilidade anual (%)": float(rets.std(ddof=1) * np.sqrt(N) * 100),
        "Sharpe": sharpe_ratio(rets, risk_free),
        "Sortino": sortino_ratio(rets, risk_free),
        "Drawdown máximo (%)": mdd,
        "Calmar": cagr / abs(mdd / 100) if mdd < 0 else np.nan,
        "Operações": int(len(trades)),
        "Taxa de acerto (%)": float((closed["Retorno (%)"] > 0).mean() * 100) if len(closed) else np.nan,
        "Tempo posicionado (%)": float(position.mean() * 100),
    }


def run_backtest(
    df: pd.DataFrame,
    strategies: dict[str, dict] | list[str],
    initial_capital: float = 1000.0,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    risk_free: float = 0.0,
) -> dict[str, StrategyResult]:
    """Roda as estratégias pedidas. ``strategies`` = {chave: parâmetros} ou lista de chaves."""
    if isinstance(strategies, (list, tuple)):
        strategies = {k: {} for k in strategies}
    strategies = {"buy_hold": {}, **strategies}  # referência sempre presente

    close = df["close"]
    rets = close.pct_change().fillna(0.0)
    targets, warmups = {}, []
    for key, params in strategies.items():
        spec = STRATEGIES[key]
        target, warmup = spec.func(df, **{**spec.params, **(params or {})})
        targets[key] = target
        warmups.append(warmup)
    start = max(max(warmups), 1)
    if start >= len(df) - 30:
        raise ValueError("Histórico insuficiente para o aquecimento dos indicadores escolhidos.")

    cost = fee + slippage
    close_prev = close.iloc[start - 1 :]  # fechamento em que cada posição foi montada
    results = {}
    for key, target in targets.items():
        # posição de hoje = sinal de ontem (executado no fechamento de ontem)
        position = target.shift(1).fillna(0.0).iloc[start:].copy()
        if key == "buy_hold":
            position[:] = 1.0
        r = rets.iloc[start:]
        turnover = position.diff().abs().fillna(position.iloc[0])  # 1º dia paga a entrada
        strat_r = position * r - turnover * cost
        equity = initial_capital * (1 + strat_r).cumprod()
        equity = pd.concat([pd.Series([initial_capital], index=close_prev.index[:1]), equity])  # parte do capital inicial
        trades = _trades(position, close_prev, cost)
        spec = STRATEGIES[key]
        results[key] = StrategyResult(
            key, spec.name, equity, position, strat_r, trades, _metrics(strat_r, equity, position, trades, risk_free)
        )
    return results


def parameter_grid(
    df: pd.DataFrame,
    fast_values=(5, 10, 15, 20, 30, 40),
    slow_values=(30, 50, 75, 100, 150, 200),
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
    metric: str = "Sharpe",
    risk_free: float = 0.0,
) -> pd.DataFrame:
    """Sharpe (ou outra métrica) do cruzamento de médias para cada combinação de parâmetros.

    Útil para ilustrar o risco de *overfitting*: o melhor ponto da grade quase
    nunca se repete fora da amostra.
    """
    close = df["close"]
    rets = close.pct_change().fillna(0.0)
    max_slow = max(s for s in slow_values if s < len(df) - 30) if any(s < len(df) - 30 for s in slow_values) else None
    if max_slow is None:
        return pd.DataFrame()
    start = max_slow
    cost = fee + slippage
    table = pd.DataFrame(index=list(fast_values), columns=list(slow_values), dtype=float)
    for f in fast_values:
        for s in slow_values:
            if f >= s or s > max_slow:
                continue
            target = (ti.sma(close, f) > ti.sma(close, s)).astype(float)
            pos = target.shift(1).fillna(0.0).iloc[start:]
            r = rets.iloc[start:]
            strat_r = pos * r - pos.diff().abs().fillna(pos.iloc[0]) * cost
            if metric == "Sharpe":
                table.loc[f, s] = sharpe_ratio(strat_r, risk_free)
            else:
                table.loc[f, s] = float(((1 + strat_r).prod() - 1) * 100)
    table.index.name = "Média rápida"
    table.columns.name = "Média lenta"
    table.attrs["start"] = df.index[start]
    return table
