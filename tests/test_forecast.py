import numpy as np
import pandas as pd
import pytest

from crypto_platform import forecast_model as fm


def test_naive_is_flat_with_widening_bands(random_walk):
    y = np.log(random_walk.to_numpy())
    fc = fm.fc_naive(y, 10)
    assert np.allclose(fc.mean, y[-1])
    width = fc.hi95 - fc.lo95
    assert np.all(np.diff(width) > 0)
    assert np.all(fc.lo95 < fc.lo80) and np.all(fc.hi80 < fc.hi95)


@pytest.mark.parametrize("name", fm.ALL_MODELS)
def test_each_model_produces_valid_forecast(random_walk, name):
    y = np.log(random_walk.to_numpy())
    fc = fm.MODEL_FUNCS[name](y, 14)
    for arr in (fc.mean, fc.lo80, fc.hi80, fc.lo95, fc.hi95):
        assert arr.shape == (14,) and np.isfinite(arr).all()
    assert np.all(fc.lo95 <= fc.mean + 1e-9) and np.all(fc.mean <= fc.hi95 + 1e-9)


def test_arima_order_selection_uses_d1(random_walk):
    spec = fm.select_arima_order(np.log(random_walk.to_numpy()))
    assert spec["order"][1] == 1 and spec["tested"] > 0 and np.isfinite(spec["aicc"])


def test_run_forecast_report(random_walk):
    rep = fm.run_forecast(random_walk, 14, fm.DEFAULT_MODELS, folds=4)
    assert set(fm.DEFAULT_MODELS) | {fm.ENSEMBLE} == set(rep.forecasts)
    assert rep.metrics.loc[fm.NAIVE, "Skill"] == pytest.approx(0)
    assert rep.forecasts[fm.NAIVE].mean.iloc[-1] == pytest.approx(random_walk.iloc[-1])
    assert rep.forecasts[fm.ENSEMBLE].mean.index[0] == random_walk.index[-1] + pd.Timedelta(days=1)
    assert np.isnan(rep.metrics.loc[fm.NAIVE, "Direcao"])  # previsão sem mudança não tem direção
    assert rep.best_model in rep.forecasts


def test_walk_forward_has_no_leakage(random_walk):
    """A 1ª janela de teste não pode mudar se alterarmos dados posteriores a ela."""
    y = np.log(random_walk.to_numpy())
    h, folds = 14, 4
    _, d1, _ = fm.walk_forward(y, h, [fm.NAIVE, fm.DRIFT, fm.ARIMA, fm.ETS], folds)
    first_origin = d1["origin_idx"].min()
    y2 = y.copy()
    y2[first_origin + h :] += 0.5  # muda o "futuro" após a 1ª janela
    _, d2, _ = fm.walk_forward(y2, h, [fm.NAIVE, fm.DRIFT, fm.ARIMA, fm.ETS], folds)
    a = d1[d1["origin_idx"] == first_origin].set_index("modelo")["previsto_final"]
    b = d2[d2["origin_idx"] == first_origin].set_index("modelo")["previsto_final"]
    pd.testing.assert_series_equal(a, b)


def test_drift_beats_naive_on_trending_series():
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    rng = np.random.default_rng(5)
    s = pd.Series(np.exp(np.log(100) + 0.01 * np.arange(400) + rng.normal(0, 0.005, 400)), index=idx)
    rep = fm.run_forecast(s, 14, (fm.NAIVE, fm.DRIFT), folds=4)
    assert rep.metrics.loc[fm.DRIFT, "Skill"] > 50
    assert fm.DRIFT in rep.beats_naive
    assert rep.metrics.loc[fm.DRIFT, "Direcao"] == 100


def test_insufficient_history_raises():
    s = pd.Series(np.linspace(1, 2, 50), index=pd.date_range("2024-01-01", periods=50))
    with pytest.raises(ValueError):
        fm.run_forecast(s, 30)
