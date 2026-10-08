import numpy as np
import pandas as pd
import pytest

from crypto_platform import technical_indicators as ti


def _reference_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI de Wilder em laço explícito (mesma convenção do TA-Lib)."""
    d = np.diff(close)
    g, l = np.where(d > 0, d, 0), np.where(d < 0, -d, 0)
    ag, al = g[:n].mean(), l[:n].mean()
    out = [np.nan] * n + [100 - 100 / (1 + ag / al)]
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + l[i]) / n
        out.append(100 - 100 / (1 + ag / al))
    return np.array(out)


def test_rsi_matches_wilder_reference(btc_close):
    ours = ti.rsi(btc_close).to_numpy()
    ref = _reference_rsi(btc_close.to_numpy())
    np.testing.assert_allclose(ours[14:], ref[14:], atol=1e-9)


def test_rsi_bounds_and_only_gains():
    up = pd.Series(np.arange(1.0, 60.0))
    assert ti.rsi(up).dropna().eq(100).all()
    r = ti.rsi(pd.Series(np.random.default_rng(0).normal(100, 1, 300)).abs())
    assert r.dropna().between(0, 100).all()


def test_macd_is_difference_of_emas(btc_close):
    m = ti.macd(btc_close)
    expected = btc_close.ewm(span=12, adjust=False).mean() - btc_close.ewm(span=26, adjust=False).mean()
    pd.testing.assert_series_equal(m["macd"].iloc[30:], expected.iloc[30:], check_names=False)
    pd.testing.assert_series_equal(m["macd_hist"], m["macd"] - m["macd_signal"], check_names=False)


def test_bollinger_middle_is_sma(btc_close):
    bb = ti.bollinger(btc_close)
    pd.testing.assert_series_equal(bb["bb_middle"], btc_close.rolling(20).mean(), check_names=False)
    assert (bb["bb_upper"].dropna() >= bb["bb_lower"].dropna()).all()


def test_atr_constant_range():
    n = 60
    close = pd.Series(np.full(n, 100.0))
    atr = ti.atr(close + 5, close - 5, close)
    assert atr.dropna().round(10).eq(10).all()


def test_adx_range(btc_df):
    a = ti.adx(btc_df["high"], btc_df["low"], btc_df["close"]).dropna()
    assert a["adx"].between(0, 100).all() and len(a) > 600


@pytest.mark.parametrize("col", ["rsi", "macd", "bb_upper", "stoch_k", "adx", "atr", "aroon_osc", "obv"])
def test_indicators_do_not_look_ahead(btc_df, col):
    full = ti.add_all_indicators(btc_df)
    past = ti.add_all_indicators(btc_df.iloc[:500])
    pd.testing.assert_series_equal(full[col].iloc[:500], past[col], check_names=False, check_freq=False)


def test_summary_structure(btc_df):
    ind = ti.add_all_indicators(btc_df)
    s = ti.technical_summary(ind)
    assert -1 <= s["score"] <= 1
    assert s["label"] in {"Viés de baixa forte", "Viés de baixa", "Neutro", "Viés de alta", "Viés de alta forte"}
    assert sum(s["counts"].values()) == len(s["table"]) == 12
    score = ti.technical_score(ind)
    assert score.dropna().between(-1, 1).all()
    assert score.iloc[-1] == pytest.approx(s["score"])


def test_close_only_data_still_works(btc_df):
    ind = ti.add_all_indicators(btc_df[["close", "volume"]])
    assert ind[["stoch_k", "adx", "atr", "williams_r"]].iloc[-1].notna().all()


def test_support_below_resistance_above(btc_df):
    sr = ti.support_resistance(btc_df)
    assert all(lv["price"] < sr["price"] for lv in sr["supports"])
    assert all(lv["price"] > sr["price"] for lv in sr["resistances"])
    assert all(lv["touches"] >= 1 for lv in sr["supports"] + sr["resistances"])


def test_fibonacci_levels_span_range(btc_df):
    fib = ti.fibonacci_levels(btc_df)
    vals = list(fib["levels"].values())
    assert min(vals) == pytest.approx(fib["low"]) and max(vals) == pytest.approx(fib["high"])
