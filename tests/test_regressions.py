"""Regressões de problemas encontrados em revisão de código."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from crypto_platform import forecast_model as fm
from crypto_platform import investment_simulator as sim
from crypto_platform import portfolio as pf
from crypto_platform import risk_analyzer as ra
from crypto_platform import technical_indicators as ti
from crypto_platform.config import COINS
from crypto_platform.data_collector import (
    SOURCE_COINGECKO,
    SOURCE_YAHOO,
    CryptoDataCollector,
    SyntheticProvider,
    normalize_ohlcv,
)


def test_neutral_monte_carlo_is_centered_even_with_short_history():
    """Bootstrap circular: com tendência removida, a mediana fica perto do valor investido."""
    medians = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2025-01-01", periods=180, freq="D")
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.004, 0.04, 180))), index=idx)
        mc = sim.monte_carlo(close, 1000, 365, 4000, sim.BOOTSTRAP, sim.DRIFT_ZERO, fee=0, slippage=0, seed=seed)
        medians.append(mc.median_value)
    assert 900 < np.median(medians) < 1100
    assert max(abs(m - 1000) for m in medians) < 150


def test_ml_does_not_inherit_random_walk_scores():
    idx = pd.date_range("2025-01-01", periods=181, freq="D")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, 181))), index=idx)
    with pytest.raises(ValueError):
        fm.fc_ml(np.log(close.to_numpy()[:100]), 30)
    rep = fm.run_forecast(close, 30, fm.DEFAULT_MODELS, folds=6)
    total = rep.metrics.attrs["total_windows"]
    if fm.ML in rep.metrics.index:
        assert rep.metrics.loc[fm.ML, "Janelas"] <= total
        if rep.metrics.loc[fm.ML, "Janelas"] < fm.MIN_WINDOWS:
            assert fm.ML not in rep.beats_naive and rep.best_model != fm.ML
    assert any(fm.ML in w for w in rep.warnings) or rep.metrics.loc[fm.ML, "Janelas"] == total


def test_skill_is_measured_on_the_same_windows():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))), index=pd.date_range("2024-01-01", periods=400))
    rep = fm.run_forecast(close, 14, (fm.NAIVE, fm.DRIFT), folds=5)
    d = rep.details
    naive = d[d["modelo"] == fm.NAIVE]["mape"].mean()
    drift = d[d["modelo"] == fm.DRIFT]["mape"].mean()
    assert rep.metrics.loc[fm.DRIFT, "Skill"] == pytest.approx((1 - drift / naive) * 100)


def test_long_gap_keeps_daily_index_continuous():
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    raw = pd.DataFrame({"close": np.linspace(100, 200, 200)}, index=idx).drop(idx[50:70])  # 20 dias sem cotação
    df, q = normalize_ohlcv(raw)
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert q.longest_gap == 20 and q.trimmed_start == "2025-03-12"  # usa só o trecho contínuo recente
    # sem histórico suficiente depois da lacuna: preenche tudo, mas mantém o índice contínuo
    raw2 = pd.DataFrame({"close": np.linspace(100, 200, 80)}, index=idx[:80]).drop(idx[40:60])
    df2, q2 = normalize_ohlcv(raw2)
    assert q2.trimmed_start is None and len(df2) == 80
    assert (df2.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_coingecko_midnight_points_are_previous_day_close(tmp_path):
    days = pd.date_range("2026-10-01", "2026-10-07", freq="D", tz="UTC")
    ts = [int(d.timestamp() * 1000) for d in days] + [int((days[-1] + pd.Timedelta(hours=15)).timestamp() * 1000)]
    prices = [[t, float(i + 1)] for i, t in enumerate(ts)]  # 1..7 nos pontos diários, 8 ao vivo
    body = {"prices": prices, "total_volumes": [[t, 1.0] for t in ts], "market_caps": [[t, 1.0] for t in ts]}
    session = MagicMock()
    session.headers = {}
    resp = MagicMock(status_code=200, headers={})
    resp.json.return_value = body
    session.get.return_value = resp
    md = CryptoDataCollector(cache_dir=str(tmp_path), session=session, demo=False).get_market_data(COINS[0], 7, "usd", SOURCE_COINGECKO)
    assert md.close.loc["2026-10-06"] == 7.0  # ponto de 07/10 00:00 = fechamento de 06/10
    assert md.close.iloc[-1] == 8.0 and md.df.index[-1] == pd.Timestamp("2026-10-07")
    assert md.df["returns"].iloc[-1] == pytest.approx(8 / 7 - 1)  # variação do dia = ao vivo vs. fechamento de ontem


class LondonFxTicker:
    """Câmbio diário com índice no fuso de Londres (horário de verão) e valor = dia do mês."""

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        if self.symbol.endswith("=X"):
            idx = pd.date_range("2026-06-01", "2026-06-30", freq="D", tz="Europe/London")
            return pd.DataFrame({"Close": [float(d.day) for d in idx]}, index=idx)
        idx = pd.date_range("2026-06-10", "2026-06-20", freq="D", tz="UTC")
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)


def test_fx_conversion_uses_same_calendar_day(tmp_path):
    c = CryptoDataCollector(cache_dir=str(tmp_path), demo=False, yahoo_ticker_factory=LondonFxTicker)
    md = c.get_market_data(COINS[1], 30, "brl", SOURCE_YAHOO)
    assert (md.close.to_numpy() == md.df.index.day.to_numpy()).all()


def test_frontier_uses_one_return_measure():
    prov = SyntheticProvider()
    prices = pf.align_prices({c.symbol: normalize_ohlcv(prov.history(c, 365, "usd"))[0]["close"] for c in COINS[:3]})
    stats = pf.asset_stats(prices)
    for sym in prices.columns:
        only = pd.Series({c: 1.0 if c == sym else 0.0 for c in prices.columns})
        assert pf.portfolio_point(only, prices)["retorno"] == pytest.approx(stats.loc[sym, "Retorno anualizado (%)"], rel=1e-6)


def test_short_history_summary_has_columns():
    df = pd.DataFrame({"close": np.linspace(1, 2, 10)}, index=pd.date_range("2026-01-01", periods=10))
    s = ti.technical_summary(ti.add_all_indicators(df))
    assert list(s["table"].columns)[:2] == ["Indicador", "Grupo"]


def test_flat_price_rsi_is_neutral():
    assert ti.rsi(pd.Series(np.full(40, 5.0))).dropna().eq(50).all()


def test_positive_var_does_not_add_risk():
    a = ra.risk_score(30, -5, 0.5, 1.0)["components"].set_index("Fator")
    assert a.loc["VaR 95% diário", "Nota (0-10)"] == 0


def test_tiny_prices_keep_significant_digits():
    from crypto_platform.formatting import fmt_price

    assert fmt_price(0.0000123456) == "US$ 0,00001235"
    assert fmt_price(1.5e-9) == "US$ 0,000000001500"
