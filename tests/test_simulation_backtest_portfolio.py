import numpy as np
import pandas as pd
import pytest

from crypto_platform import backtesting as bt
from crypto_platform import investment_simulator as sim
from crypto_platform import portfolio as pf
from crypto_platform.config import COINS
from crypto_platform.data_collector import SyntheticProvider, normalize_ohlcv

# --- Monte Carlo ----------------------------------------------------------------


def test_monte_carlo_neutral_median_close_to_investment(btc_close):
    mc = sim.monte_carlo(btc_close, 1000, 60, 20000, sim.BOOTSTRAP, sim.DRIFT_ZERO, fee=0, slippage=0, seed=1)
    assert mc.percentiles.shape == (61, 5)
    assert mc.median_value == pytest.approx(1000, rel=0.03)
    assert mc.expected_value > mc.median_value  # assimetria log-normal
    assert 0 <= mc.prob_loss <= 100 and mc.var95 > 0 and mc.cvar95 >= mc.var95


def test_monte_carlo_is_reproducible(btc_close):
    a = sim.monte_carlo(btc_close, 1000, 30, 2000, sim.GBM, seed=9)
    b = sim.monte_carlo(btc_close, 1000, 30, 2000, sim.GBM, seed=9)
    np.testing.assert_array_equal(a.final_values, b.final_values)


def test_fees_reduce_outcome(btc_close):
    no_fee = sim.monte_carlo(btc_close, 1000, 30, 2000, fee=0, slippage=0, seed=2)
    fee = sim.monte_carlo(btc_close, 1000, 30, 2000, fee=0.01, slippage=0, seed=2)
    assert fee.median_value < no_fee.median_value


def test_dca_vs_lump_sum():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    flat = pd.Series(100.0, index=idx)
    res = sim.lump_sum_vs_dca(flat, 1000, 7, fee=0, slippage=0)
    assert res.summary["Valor final"].tolist() == pytest.approx([1000, 1000])
    rising = pd.Series(np.linspace(100, 200, 100), index=idx)
    res = sim.lump_sum_vs_dca(rising, 1000, 7, fee=0, slippage=0)
    assert res.summary.loc["Aporte único", "Valor final"] == pytest.approx(2000)
    assert res.summary.loc["Aporte único", "Valor final"] > res.summary.loc["Aportes periódicos (DCA)", "Valor final"]
    assert res.purchases["Valor aplicado"].sum() == pytest.approx(1000)


# --- Backtest -----------------------------------------------------------------------


def test_buy_and_hold_matches_price_change(btc_df):
    res = bt.run_backtest(btc_df, ["sma_cross"], 1000, fee=0, slippage=0)
    bh = res["buy_hold"]
    start = bh.equity.index[0]
    expected = btc_df["close"].iloc[-1] / btc_df["close"].loc[start] * 1000
    assert bh.equity.iloc[-1] == pytest.approx(expected)
    assert bh.equity.iloc[0] == 1000


def test_backtest_has_no_lookahead(btc_df):
    """Alterar o último preço não pode mudar nenhuma posição anterior."""
    base = bt.run_backtest(btc_df, list(bt.STRATEGIES), 1000)
    shocked = btc_df.copy()
    shocked.iloc[-1, shocked.columns.get_loc("close")] *= 3
    after = bt.run_backtest(shocked, list(bt.STRATEGIES), 1000)
    for key in base:
        pd.testing.assert_series_equal(base[key].position, after[key].position)


def test_costs_are_charged_per_trade(btc_df):
    free = bt.run_backtest(btc_df, ["macd"], 1000, fee=0, slippage=0)["macd"]
    paid = bt.run_backtest(btc_df, ["macd"], 1000, fee=0.005, slippage=0)["macd"]
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert free.metrics["Operações"] == len(free.trades) > 0


def test_trades_are_consistent(btc_df):
    r = bt.run_backtest(btc_df, ["sma_cross"], 1000, fee=0, slippage=0)["sma_cross"]
    closed = r.trades[~r.trades["Aberta"]]
    assert (closed["Saída"] > closed["Entrada"]).all()
    growth = np.prod(1 + r.trades["Retorno (%)"] / 100)
    assert r.equity.iloc[-1] / r.equity.iloc[0] == pytest.approx(growth, rel=1e-9)


def test_parameter_grid(btc_df):
    g = bt.parameter_grid(btc_df)
    assert g.shape == (6, 6)
    assert np.isnan(g.loc[30, 30])  # rápida >= lenta não é testada


# --- Portfólio --------------------------------------------------------------------


@pytest.fixture(scope="module")
def prices():
    prov = SyntheticProvider()
    data = {c.symbol: normalize_ohlcv(prov.history(c, 365, "usd"))[0]["close"] for c in COINS[:4]}
    return pf.align_prices(data)


@pytest.mark.parametrize("method", [pf.EQUAL, pf.MIN_VAR, pf.MAX_SHARPE, pf.RISK_PARITY])
def test_weights_are_valid(prices, method):
    w = pf.optimize(prices, method, max_weight=0.6)
    assert w.sum() == pytest.approx(1) and (w >= 0).all() and (w <= 0.6 + 1e-6).all()


def test_min_variance_has_lowest_vol(prices):
    mv = pf.portfolio_point(pf.optimize(prices, pf.MIN_VAR), prices)
    eq = pf.portfolio_point(pf.optimize(prices, pf.EQUAL), prices)
    rnd = pf.random_portfolios(prices, 500)
    assert mv["vol"] <= eq["vol"] + 1e-9 and mv["vol"] <= rnd["vol"].min() + 1e-6


def test_risk_parity_equalizes_contributions(prices):
    rc = pf.risk_contributions(pf.optimize(prices, pf.RISK_PARITY), prices)
    assert rc.max() - rc.min() < 1.0  # pontos percentuais


def test_buy_and_hold_portfolio(prices):
    w = pd.Series(0.25, index=prices.columns)
    res = pf.backtest_portfolio(prices, w, rebalance=None, initial=1000, fee=0, slippage=0)
    expected = (1000 * w * prices.iloc[-1] / prices.iloc[0]).sum()
    assert res.equity.iloc[-1] == pytest.approx(expected)
    monthly = pf.backtest_portfolio(prices, w, rebalance="MS", initial=1000, fee=0.001, slippage=0)
    assert monthly.turnover_cost > 1000 * 0.001
