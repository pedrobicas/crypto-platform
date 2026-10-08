"""Indicadores técnicos, níveis de suporte/resistência e resumo técnico.

Todas as funções são vetorizadas e não olham para o futuro (exceto a detecção de
pivôs de suporte/resistência, que exige confirmação de ``window`` dias à frente —
por isso pivôs muito recentes não são considerados).

Quando a fonte não fornece máxima/mínima (CoinGecko), os indicadores que dependem
delas usam o fechamento como aproximação; a interface avisa o usuário.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicadores individuais
# ---------------------------------------------------------------------------


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Média de Wilder (RMA) semeada com a média simples dos primeiros ``period``
    valores — mesma convenção do TA-Lib/TradingView para RSI, ATR e ADX."""
    s = series.astype(float)
    valid = np.flatnonzero(s.notna().to_numpy())
    if len(valid) < period:
        return pd.Series(np.nan, index=s.index)
    seed_pos = valid[period - 1]
    seeded = s.copy()
    seeded.iloc[:seed_pos] = np.nan
    seeded.iloc[seed_pos] = s.iloc[valid[0] : seed_pos + 1].mean()
    return seeded.ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder (padrão de mercado)."""
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), period)
    loss = wilder(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0)  # só altas no período => RSI 100
    out = out.where((gain != 0) | (loss != 0), 50.0)  # preço parado => neutro
    return out.where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    line = line.where(close.expanding().count() >= slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + n_std * std, mid - n_std * std
    width = upper - lower
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": mid,
            "bb_lower": lower,
            "bb_pct_b": (close - lower) / width.replace(0, np.nan),
            "bb_width": width / mid * 100,
        }
    )


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    rng = (highest - lowest).replace(0, np.nan)
    k = (100 * (close - lowest) / rng).fillna(50.0).where(lowest.notna())
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(d_period, min_periods=d_period).mean()})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    rng = (highest - lowest).replace(0, np.nan)
    return (-100 * (highest - close) / rng).fillna(-50.0).where(highest.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.where(prev.notna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """ADX/DMI de Wilder."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=close.index)
    tr_smooth = wilder(true_range(high, low, close), period).replace(0, np.nan)
    di_plus = 100 * wilder(plus_dm.where(up.notna()), period) / tr_smooth
    di_minus = 100 * wilder(minus_dm.where(up.notna()), period) / tr_smooth
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, period), "di_plus": di_plus, "di_minus": di_minus})


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    window = period + 1
    days_since_high = high.rolling(window, min_periods=window).apply(lambda x: period - int(np.argmax(x)), raw=True)
    days_since_low = low.rolling(window, min_periods=window).apply(lambda x: period - int(np.argmin(x)), raw=True)
    up = 100 * (period - days_since_high) / period
    down = 100 * (period - days_since_low) / period
    return pd.DataFrame({"aroon_up": up, "aroon_down": down, "aroon_osc": up - down})


def roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period) * 100


# ---------------------------------------------------------------------------
# Conjunto completo
# ---------------------------------------------------------------------------


def _hlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    return high, low, close


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna uma cópia do DataFrame com todos os indicadores como colunas."""
    out = df.copy()
    high, low, close = _hlc(df)

    for p in (20, 50, 200):
        out[f"sma_{p}"] = sma(close, p)
    for p in (12, 26, 50):
        out[f"ema_{p}"] = ema(close, p)

    out["rsi"] = rsi(close)
    out = out.join(macd(close))
    out = out.join(bollinger(close))
    out = out.join(stochastic(high, low, close))
    out["williams_r"] = williams_r(high, low, close)
    out["atr"] = atr(high, low, close)
    out["atr_pct"] = out["atr"] / close * 100
    out = out.join(adx(high, low, close))
    out = out.join(aroon(high, low))
    out["roc_10"] = roc(close, 10)
    out["roc_20"] = roc(close, 20)

    if "volume" in df.columns:
        out["obv"] = obv(close, df["volume"])
        out["obv_sma"] = sma(out["obv"], 20)
        out["volume_sma"] = sma(df["volume"], 20)
        out["volume_ratio"] = df["volume"] / out["volume_sma"].replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Resumo técnico (votação de indicadores)
# ---------------------------------------------------------------------------

OSCILLATORS = "Osciladores"
TREND = "Médias e tendência"


@dataclass(frozen=True)
class Rule:
    name: str
    group: str


RULES = (
    Rule("RSI (14)", OSCILLATORS),
    Rule("Estocástico (14, 3)", OSCILLATORS),
    Rule("Bandas de Bollinger (20, 2)", OSCILLATORS),
    Rule("MACD (12, 26, 9)", OSCILLATORS),
    Rule("Momento (ROC 20)", OSCILLATORS),
    Rule("Preço × média de 20 dias", TREND),
    Rule("Preço × média de 50 dias", TREND),
    Rule("Preço × média de 200 dias", TREND),
    Rule("Média 50 × média 200", TREND),
    Rule("ADX / DMI (14)", TREND),
    Rule("Aroon (25)", TREND),
    Rule("OBV × média de 20 dias", TREND),
)


def _vote(cond_up: pd.Series, cond_down: pd.Series, valid: pd.Series) -> pd.Series:
    v = pd.Series(0.0, index=valid.index)
    v[cond_up.fillna(False).astype(bool)] = 1.0
    v[cond_down.fillna(False).astype(bool)] = -1.0
    return v.where(valid.fillna(False).astype(bool))


def signal_votes(ind: pd.DataFrame) -> pd.DataFrame:
    """Voto de cada regra em cada dia: +1 (alta), -1 (baixa), 0 (neutro), NaN (sem dados)."""
    c = ind["close"]
    votes = {
        "RSI (14)": _vote(ind["rsi"] < 30, ind["rsi"] > 70, ind["rsi"].notna()),
        "Estocástico (14, 3)": _vote(ind["stoch_k"] < 20, ind["stoch_k"] > 80, ind["stoch_k"].notna()),
        "Bandas de Bollinger (20, 2)": _vote(ind["bb_pct_b"] < 0, ind["bb_pct_b"] > 1, ind["bb_pct_b"].notna()),
        "MACD (12, 26, 9)": _vote(ind["macd"] > ind["macd_signal"], ind["macd"] < ind["macd_signal"], ind["macd"].notna()),
        "Momento (ROC 20)": _vote(ind["roc_20"] > 0, ind["roc_20"] < 0, ind["roc_20"].notna()),
        "Preço × média de 20 dias": _vote(c > ind["sma_20"], c < ind["sma_20"], ind["sma_20"].notna()),
        "Preço × média de 50 dias": _vote(c > ind["sma_50"], c < ind["sma_50"], ind["sma_50"].notna()),
        "Preço × média de 200 dias": _vote(c > ind["sma_200"], c < ind["sma_200"], ind["sma_200"].notna()),
        "Média 50 × média 200": _vote(ind["sma_50"] > ind["sma_200"], ind["sma_50"] < ind["sma_200"], ind["sma_200"].notna()),
        "ADX / DMI (14)": _vote(
            (ind["adx"] >= 20) & (ind["di_plus"] > ind["di_minus"]),
            (ind["adx"] >= 20) & (ind["di_plus"] < ind["di_minus"]),
            ind["adx"].notna(),
        ),
        "Aroon (25)": _vote(ind["aroon_osc"] > 50, ind["aroon_osc"] < -50, ind["aroon_osc"].notna()),
    }
    if "obv" in ind.columns:
        votes["OBV × média de 20 dias"] = _vote(ind["obv"] > ind["obv_sma"], ind["obv"] < ind["obv_sma"], ind["obv_sma"].notna())
    return pd.DataFrame(votes, index=ind.index)


def technical_score(ind: pd.DataFrame) -> pd.Series:
    """Score técnico diário em [-1, 1] (média dos votos disponíveis)."""
    votes = signal_votes(ind)
    enough = votes.notna().sum(axis=1) >= 5
    return votes.mean(axis=1).where(enough)


def score_label(score: float) -> str:
    if score is None or np.isnan(score):
        return "Indefinido"
    if score <= -0.5:
        return "Viés de baixa forte"
    if score <= -0.1:
        return "Viés de baixa"
    if score < 0.1:
        return "Neutro"
    if score < 0.5:
        return "Viés de alta"
    return "Viés de alta forte"


def _describe(rule: str, row: pd.Series, vote: float) -> tuple[str, str]:
    """(valor exibido, explicação) para a última leitura de cada regra."""
    from .formatting import fmt_number

    c = row["close"]
    if rule == "RSI (14)":
        txt = "sobrevendido (abaixo de 30)" if vote > 0 else "sobrecomprado (acima de 70)" if vote < 0 else "zona neutra (30 a 70)"
        return fmt_number(row["rsi"], 1), txt
    if rule == "Estocástico (14, 3)":
        txt = "sobrevendido (abaixo de 20)" if vote > 0 else "sobrecomprado (acima de 80)" if vote < 0 else "zona neutra"
        return fmt_number(row["stoch_k"], 1), txt
    if rule == "Bandas de Bollinger (20, 2)":
        txt = "abaixo da banda inferior" if vote > 0 else "acima da banda superior" if vote < 0 else "dentro das bandas"
        return f"%B {fmt_number(row['bb_pct_b'], 2)}", txt
    if rule == "MACD (12, 26, 9)":
        txt = "MACD acima da linha de sinal" if vote > 0 else "MACD abaixo da linha de sinal"
        return fmt_number(row["macd_hist"], 4 if abs(row["macd_hist"]) < 1 else 2), txt
    if rule == "Momento (ROC 20)":
        return f"{fmt_number(row['roc_20'], 2)}%", "preço maior que há 20 dias" if vote > 0 else "preço menor que há 20 dias"
    if rule.startswith("Preço × média de"):
        p = rule.split()[-2]
        ma = row[f"sma_{p}"]
        dist = (c / ma - 1) * 100
        return f"{fmt_number(dist, 2)}%", f"preço {'acima' if vote > 0 else 'abaixo'} da média de {p} dias"
    if rule == "Média 50 × média 200":
        return "", "cruz dourada (média 50 acima da 200)" if vote > 0 else "cruz da morte (média 50 abaixo da 200)"
    if rule == "ADX / DMI (14)":
        if vote == 0:
            return fmt_number(row["adx"], 1), "sem tendência definida (ADX abaixo de 20)"
        return fmt_number(row["adx"], 1), f"tendência de {'alta' if vote > 0 else 'baixa'} com força"
    if rule == "Aroon (25)":
        txt = "máximas recentes dominam" if vote > 0 else "mínimas recentes dominam" if vote < 0 else "sem direção clara"
        return fmt_number(row["aroon_osc"], 0), txt
    if rule == "OBV × média de 20 dias":
        return "", "volume confirma compradores" if vote > 0 else "volume confirma vendedores"
    return "", ""


def technical_summary(ind: pd.DataFrame) -> dict:
    """Leitura atual de cada regra + score geral e por grupo."""
    votes = signal_votes(ind)
    last_votes = votes.iloc[-1]
    row = ind.iloc[-1]
    rows = []
    group_of = {r.name: r.group for r in RULES}
    for rule, vote in last_votes.items():
        if np.isnan(vote):
            continue
        value, explanation = _describe(rule, row, vote)
        rows.append(
            {
                "Indicador": rule,
                "Grupo": group_of.get(rule, TREND),
                "Leitura": value,
                "Sinal": "Alta" if vote > 0 else "Baixa" if vote < 0 else "Neutro",
                "Interpretação": explanation,
                "voto": vote,
            }
        )
    table = pd.DataFrame(rows, columns=["Indicador", "Grupo", "Leitura", "Sinal", "Interpretação", "voto"])
    score = float(table["voto"].mean()) if not table.empty else float("nan")
    groups = {}
    for g in (OSCILLATORS, TREND):
        sub = table[table["Grupo"] == g] if not table.empty else table
        groups[g] = float(sub["voto"].mean()) if len(sub) else float("nan")
    return {
        "score": score,
        "label": score_label(score),
        "groups": groups,
        "table": table,
        "counts": {
            "alta": int((table["voto"] > 0).sum()) if not table.empty else 0,
            "neutro": int((table["voto"] == 0).sum()) if not table.empty else 0,
            "baixa": int((table["voto"] < 0).sum()) if not table.empty else 0,
        },
    }


# ---------------------------------------------------------------------------
# Suporte, resistência e Fibonacci
# ---------------------------------------------------------------------------


def support_resistance(df: pd.DataFrame, window: int = 7, tolerance_pct: float | None = None, max_levels: int = 3) -> dict:
    """Detecta pivôs confirmados e agrupa níveis próximos.

    Retorna os níveis mais próximos do preço atual: suportes abaixo e
    resistências acima, cada um com número de toques e distância percentual.
    """
    high, low, close = _hlc(df)
    span = 2 * window + 1
    is_peak = high == high.rolling(span, center=True, min_periods=span).max()
    is_valley = low == low.rolling(span, center=True, min_periods=span).min()
    pivots = pd.concat([high[is_peak], low[is_valley]]).sort_index()
    price = float(close.iloc[-1])
    if pivots.empty:
        return {"supports": [], "resistances": [], "price": price}

    if tolerance_pct is None:
        atr_pct = (atr(high, low, close) / close).median()
        tolerance_pct = float(np.clip(atr_pct if np.isfinite(atr_pct) else 0.02, 0.01, 0.06))

    values = pivots.sort_values()
    clusters: list[list[tuple[pd.Timestamp, float]]] = []
    for date, value in values.items():
        if clusters and value <= np.mean([v for _, v in clusters[-1]]) * (1 + tolerance_pct):
            clusters[-1].append((date, value))
        else:
            clusters.append([(date, value)])

    levels = []
    for cl in clusters:
        level = float(np.mean([v for _, v in cl]))
        levels.append(
            {
                "price": level,
                "touches": len(cl),
                "last_touch": max(d for d, _ in cl),
                "distance_pct": (level / price - 1) * 100,
            }
        )
    supports = sorted([lv for lv in levels if lv["price"] < price], key=lambda x: -x["price"])[:max_levels]
    resistances = sorted([lv for lv in levels if lv["price"] > price], key=lambda x: x["price"])[:max_levels]
    return {"supports": supports, "resistances": resistances, "price": price}


FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)


def fibonacci_levels(df: pd.DataFrame) -> dict:
    """Retrações de Fibonacci entre a mínima e a máxima do período."""
    high, low, _ = _hlc(df)
    hi_date, lo_date = high.idxmax(), low.idxmin()
    hi, lo = float(high.max()), float(low.min())
    uptrend = lo_date < hi_date  # a perna mais recente foi de alta: retrações a partir do topo
    levels = {}
    for r in FIB_RATIOS:
        levels[r] = hi - r * (hi - lo) if uptrend else lo + r * (hi - lo)
    return {"levels": levels, "uptrend": uptrend, "high": hi, "low": lo}


# ---------------------------------------------------------------------------
# Compatibilidade com a API da versão anterior
# ---------------------------------------------------------------------------


class TechnicalIndicators:
    """Wrapper orientado a objeto (mantido por compatibilidade)."""

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns and "price" in df.columns:
            df = df.rename(columns={"price": "close"})
        out = add_all_indicators(df)
        sr = support_resistance(out)
        out.attrs["support_levels"] = [lv["price"] for lv in sr["supports"]]
        out.attrs["resistance_levels"] = [lv["price"] for lv in sr["resistances"]]
        return out

    def get_summary(self, df_with_indicators: pd.DataFrame) -> dict:
        return technical_summary(df_with_indicators)
