import numpy as np
import pandas as pd
import pytest

from crypto_platform import risk_analyzer as ra


def test_drawdown_known_path():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    p = pd.Series([100, 120, 60, 90, 130], index=idx, dtype=float)
    dd = ra.drawdown_analysis(p)
    assert dd.max_drawdown == pytest.approx(-50)
    assert dd.peak_date == idx[1] and dd.trough_date == idx[2] and dd.recovery_date == idx[4]
    assert dd.duration_days == 3 and dd.current_drawdown == 0


def test_unrecovered_drawdown():
    p = pd.Series([100, 80, 90], index=pd.date_range("2024-01-01", periods=3), dtype=float)
    dd = ra.drawdown_analysis(p)
    assert dd.recovery_date is None and dd.current_drawdown == pytest.approx(-10)


def test_volatility_annualization():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.02, 20000))
    assert ra.annualized_volatility(r) == pytest.approx(0.02 * np.sqrt(365), rel=0.02)


def test_sortino_uses_full_downside_deviation():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert ra.sortino_ratio(r) == pytest.approx(r.mean() / downside * np.sqrt(365))


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(ra.sharpe_ratio(pd.Series([0.01] * 10)))


def test_var_cvar_ordering(btc_close):
    v = ra.var_cvar(btc_close.pct_change().dropna(), 0.95)
    assert v["historical_cvar"] <= v["historical_var"] < 0
    v99 = ra.var_cvar(btc_close.pct_change().dropna(), 0.99)
    assert v99["historical_var"] <= v["historical_var"]


def test_cagr():
    r = pd.Series([0.0] * 364 + [0.10])
    assert ra.annualized_return(r) == pytest.approx(0.10)


def test_risk_score_bounds_and_levels():
    low = ra.risk_score(5, -5, -0.5, 2.5)
    high = ra.risk_score(150, -90, -12, -2)
    assert low["score"] < 1 and low["level"] == "Baixo"
    assert high["score"] == pytest.approx(10) and high["level"] == "Muito alto"
    assert low["components"]["Peso"].sum() == pytest.approx(1)


def test_full_report(btc_close):
    rep = ra.analyze(btc_close, 0.04)
    assert 0 <= rep.score["score"] <= 10
    assert rep.drawdown.max_drawdown < 0
    assert set(rep.volatility) == {7, 30, 90, 365}
    assert rep.tail["extreme_freq_pct"] >= 0


def test_beta_of_scaled_series():
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0, 0.03, 500))
    a = 1.5 * m + pd.Series(rng.normal(0, 0.001, 500))
    bc = ra.beta_correlation(a, m)
    assert bc["beta"] == pytest.approx(1.5, rel=0.02) and bc["correlation"] > 0.99


def test_position_size():
    res = ra.position_size(10_000, 1, 8)
    assert res["position_value"] == pytest.approx(1250) and res["max_loss"] == pytest.approx(100)
    assert ra.position_size(10_000, 5, 1)["position_value"] == 10_000  # nunca maior que o capital
