"""Previsão de preços com vários modelos e validação walk-forward.

Todos os modelos trabalham sobre o log do preço (variância mais estável e preços
sempre positivos). O valor central exibido é a mediana prevista
(exp da média em log) e os intervalos são de 80% e 95%.

Princípio central: um modelo só tem valor se superar o **passeio aleatório**
(amanhã = hoje) em dados que ele não viu. Por isso todo modelo passa por um
backtest com origem móvel (*walk-forward*) antes de ser exibido.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

NAIVE = "Passeio aleatório"
DRIFT = "Passeio aleatório c/ tendência"
ARIMA = "ARIMA automático"
ETS = "Suavização exponencial (ETS)"
THETA = "Theta"
ML = "Gradient Boosting (ML)"
ENSEMBLE = "Combinação (média)"

ALL_MODELS = (NAIVE, DRIFT, ARIMA, ETS, THETA, ML)
SHORT_NAMES = {
    NAIVE: "Aleatório",
    DRIFT: "Tendência",
    ARIMA: "ARIMA",
    ETS: "ETS",
    THETA: "Theta",
    ML: "ML (boosting)",
    ENSEMBLE: "Combinação",
}
DEFAULT_MODELS = (NAIVE, DRIFT, ARIMA, ETS, THETA, ML)

MODEL_DESCRIPTIONS = {
    NAIVE: "O preço de amanhã é igual ao de hoje. É a referência que qualquer modelo precisa vencer.",
    DRIFT: "Passeio aleatório mais a tendência média do histórico (extrapola a direção passada).",
    ARIMA: "ARIMA(p,1,q) com ordem escolhida por AICc; captura autocorrelação de curto prazo.",
    ETS: "Suavização exponencial com tendência amortecida (Holt); dá mais peso ao passado recente.",
    THETA: "Método Theta (vencedor da competição M3): combina tendência linear e suavização.",
    ML: "Gradient Boosting com retornos defasados, volatilidade, RSI e distância às médias.",
    ENSEMBLE: "Média, em log, das previsões de todos os modelos selecionados.",
}

# Mínimo de janelas de teste para afirmar que um modelo supera a referência
MIN_WINDOWS = 3

Z80 = stats.norm.ppf(0.90)
Z95 = stats.norm.ppf(0.975)


@dataclass
class LogForecast:
    mean: np.ndarray
    lo80: np.ndarray
    hi80: np.ndarray
    lo95: np.ndarray
    hi95: np.ndarray
    info: dict = field(default_factory=dict)


def _gaussian_bands(y_last: float, path: np.ndarray, sigma_steps: np.ndarray, info: dict | None = None) -> LogForecast:
    return LogForecast(
        mean=path,
        lo80=path - Z80 * sigma_steps,
        hi80=path + Z80 * sigma_steps,
        lo95=path - Z95 * sigma_steps,
        hi95=path + Z95 * sigma_steps,
        info=info or {},
    )


# ---------------------------------------------------------------------------
# Modelos (entrada: log-preço; saída: LogForecast com h passos)
# ---------------------------------------------------------------------------


def fc_naive(y: np.ndarray, h: int, **_) -> LogForecast:
    r = np.diff(y)
    sigma = r.std(ddof=1)
    steps = np.arange(1, h + 1)
    return _gaussian_bands(y[-1], np.full(h, y[-1]), sigma * np.sqrt(steps), {"sigma_diario": sigma})


def fc_drift(y: np.ndarray, h: int, **_) -> LogForecast:
    r = np.diff(y)
    mu, sigma, T = r.mean(), r.std(ddof=1), len(r)
    steps = np.arange(1, h + 1)
    path = y[-1] + mu * steps
    se = sigma * np.sqrt(steps * (1 + steps / T))
    return _gaussian_bands(y[-1], path, se, {"tendencia_diaria_pct": mu * 100})


def select_arima_order(y: np.ndarray, max_p: int = 2, max_q: int = 2) -> dict:
    """Busca em grade ARIMA(p,1,q) com e sem drift, escolhendo o menor AICc.

    Todas as combinações usam d=1 (log-preço é integrado de ordem 1), então os
    critérios de informação são comparáveis entre si.
    """
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    best = {"order": (0, 1, 0), "trend": "n", "aicc": np.inf}
    tested = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q, trend in itertools.product(range(max_p + 1), range(max_q + 1), ("n", "t")):
            try:
                res = _ARIMA(y, order=(p, 1, q), trend=trend).fit()
            except Exception:
                continue
            tested += 1
            if np.isfinite(res.aicc) and res.aicc < best["aicc"]:
                best = {"order": (p, 1, q), "trend": trend, "aicc": float(res.aicc)}
    best["tested"] = tested
    return best


def fc_arima(y: np.ndarray, h: int, arima_spec: dict | None = None, **_) -> LogForecast:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA

    spec = arima_spec or select_arima_order(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _ARIMA(y, order=spec["order"], trend=spec["trend"]).fit()
        fc = res.get_forecast(h)
        ci80 = np.asarray(fc.conf_int(alpha=0.20))
        ci95 = np.asarray(fc.conf_int(alpha=0.05))
    info = {"ordem": spec["order"], "drift": spec["trend"] == "t", "aicc": float(res.aicc)}
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(res.resid[1:], lags=[10])
        info["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        pass
    return LogForecast(np.asarray(fc.predicted_mean), ci80[:, 0], ci80[:, 1], ci95[:, 0], ci95[:, 1], info)


def fc_ets(y: np.ndarray, h: int, **_) -> LogForecast:
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # pd.Series (e não ndarray): o get_prediction do ETS exige índice
        res = ETSModel(pd.Series(y), error="add", trend="add", damped_trend=True).fit(disp=False)
        pred = res.get_prediction(start=len(y), end=len(y) + h - 1)
        f80 = pred.summary_frame(alpha=0.20)
        f95 = pred.summary_frame(alpha=0.05)
    info = {"alpha": float(res.smoothing_level), "amortecimento": float(res.damping_trend), "aic": float(res.aic)}
    return LogForecast(
        f95["mean"].to_numpy(),
        f80["pi_lower"].to_numpy(),
        f80["pi_upper"].to_numpy(),
        f95["pi_lower"].to_numpy(),
        f95["pi_upper"].to_numpy(),
        info,
    )


def fc_theta(y: np.ndarray, h: int, **_) -> LogForecast:
    from statsmodels.tsa.forecasting.theta import ThetaModel

    series = pd.Series(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ThetaModel(series, deseasonalize=False).fit()
        mean = res.forecast(h).to_numpy()
        pi80 = res.prediction_intervals(h, alpha=0.20).to_numpy()
        pi95 = res.prediction_intervals(h, alpha=0.05).to_numpy()
    return LogForecast(mean, pi80[:, 0], pi80[:, 1], pi95[:, 0], pi95[:, 1], {"b0": float(res.params.get("b0", np.nan))})


def _ml_features(y: np.ndarray) -> pd.DataFrame:
    s = pd.Series(y)
    r = s.diff()
    feats = {f"ret_{k}": s - s.shift(k) for k in (1, 3, 7, 14, 30)}
    feats["vol_7"] = r.rolling(7).std()
    feats["vol_30"] = r.rolling(30).std()
    price = np.exp(s)
    feats["dist_sma20"] = s - np.log(price.rolling(20).mean())
    feats["dist_sma50"] = s - np.log(price.rolling(50).mean())
    gain = r.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-r.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    feats["rsi"] = 1 - 1 / (1 + gain / loss.replace(0, np.nan))
    return pd.DataFrame(feats)


def fc_ml(y: np.ndarray, h: int, **_) -> LogForecast:
    """Modelo direto: prevê o retorno acumulado de h dias a partir das features de hoje."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = _ml_features(y)
    target = pd.Series(y).shift(-h) - pd.Series(y)
    train = X.assign(target=target).dropna()
    if len(train) < 60:
        # falha explícita: devolver o passeio aleatório aqui faria o ML "herdar" as notas dele no backtest
        raise ValueError("histórico insuficiente para treinar o modelo de ML")
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(train.drop(columns="target"), train["target"])
    x_last = X.iloc[[-1]].fillna(X.median())
    pred_total = float(model.predict(x_last)[0])
    steps = np.arange(1, h + 1)
    path = y[-1] + pred_total * steps / h
    sigma = np.diff(y).std(ddof=1)
    return _gaussian_bands(y[-1], path, sigma * np.sqrt(steps), {"retorno_previsto_pct": (np.exp(pred_total) - 1) * 100})


MODEL_FUNCS: dict[str, Callable[..., LogForecast]] = {
    NAIVE: fc_naive,
    DRIFT: fc_drift,
    ARIMA: fc_arima,
    ETS: fc_ets,
    THETA: fc_theta,
    ML: fc_ml,
}


def combine(forecasts: list[LogForecast]) -> LogForecast:
    stack = lambda attr: np.mean([getattr(f, attr) for f in forecasts], axis=0)  # noqa: E731
    return LogForecast(stack("mean"), stack("lo80"), stack("hi80"), stack("lo95"), stack("hi95"), {"modelos": len(forecasts)})


def _safe_forecast(name: str, y: np.ndarray, h: int, ctx: dict) -> tuple[LogForecast | None, str | None]:
    try:
        fc = MODEL_FUNCS[name](y, h, **ctx)
    except Exception as exc:
        return None, f"{name}: falhou ({exc.__class__.__name__})"
    if not np.all(np.isfinite(fc.mean)):
        return None, f"{name}: previsão inválida"
    return fc, None


# ---------------------------------------------------------------------------
# Backtest walk-forward
# ---------------------------------------------------------------------------


def walk_forward(
    y: np.ndarray, horizon: int, models: list[str], folds: int = 6, min_train: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Avalia cada modelo em ``folds`` origens móveis, sempre treinando apenas no passado.

    Retorna (métricas por modelo, detalhes por origem, especificação ARIMA usada).
    """
    T = len(y)
    min_train = min_train or max(120, 2 * horizon)
    step = max(horizon // 2, 5)
    origins = sorted(o for o in (T - horizon - step * i for i in range(folds)) if o >= min_train)
    if not origins:
        raise ValueError(
            f"Histórico insuficiente para validar uma previsão de {horizon} dias. Use um período de histórico maior ou um horizonte menor."
        )

    ctx: dict = {}
    if ARIMA in models:
        ctx["arima_spec"] = select_arima_order(y[: origins[0]])  # sem vazamento: só dados anteriores ao 1º teste

    records = []
    for o in origins:
        train, actual_log = y[:o], y[o : o + horizon]
        actual = np.exp(actual_log)
        base = np.exp(train[-1])
        fold_fc: dict[str, LogForecast] = {}
        for name in models:
            fc, _ = _safe_forecast(name, train, horizon, ctx)
            if fc is not None:
                fold_fc[name] = fc
        if len(fold_fc) > 1:
            fold_fc[ENSEMBLE] = combine(list(fold_fc.values()))
        for name, fc in fold_fc.items():
            pred = np.exp(fc.mean)
            inside = (actual_log >= fc.lo95) & (actual_log <= fc.hi95)
            records.append(
                {
                    "origem": pd.NaT,
                    "origin_idx": o,
                    "modelo": name,
                    "mape": float(np.mean(np.abs(pred - actual) / actual) * 100),
                    "mape_final": float(abs(pred[-1] - actual[-1]) / actual[-1] * 100),
                    "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
                    "mae": float(np.mean(np.abs(pred - actual))),
                    # previsão "sem mudança" (passeio aleatório, ARIMA(0,1,0)) não indica direção
                    "acertou_direcao": (
                        float(np.sign(pred[-1] - base) == np.sign(actual[-1] - base)) if abs(pred[-1] / base - 1) > 1e-9 else np.nan
                    ),
                    "cobertura95": float(inside.mean() * 100),
                    "previsto_final": float(pred[-1]),
                    "real_final": float(actual[-1]),
                }
            )
    details = pd.DataFrame(records)
    agg = details.groupby("modelo").agg(
        MAPE=("mape", "mean"),
        MAPE_final=("mape_final", "mean"),
        RMSE=("rmse", "mean"),
        MAE=("mae", "mean"),
        Direcao=("acertou_direcao", lambda s: float(pd.to_numeric(s, errors="coerce").mean() * 100) if s.notna().any() else np.nan),
        Cobertura95=("cobertura95", "mean"),
        Janelas=("mape", "size"),
    )
    # Ganho sobre o passeio aleatório calculado nas MESMAS janelas em que o modelo foi
    # avaliado (um modelo que falhou em alguma janela não é comparado a um conjunto diferente).
    naive_by_origin = details[details["modelo"] == NAIVE].set_index("origin_idx")["mape"]
    skills = {}
    for name, grp in details.groupby("modelo"):
        ref = naive_by_origin.reindex(grp["origin_idx"]).mean()
        skills[name] = (1 - grp["mape"].mean() / ref) * 100 if ref and np.isfinite(ref) else np.nan
    agg["Skill"] = pd.Series(skills)
    # Ordena pelo ganho; em empate a referência vem primeiro — um modelo mais
    # complexo só "ganha" se for estritamente melhor.
    agg["_tie"] = [0 if m == NAIVE else 1 for m in agg.index]
    agg["_skill_r"] = agg["Skill"].round(6)
    agg = agg.sort_values(["_skill_r", "_tie"], ascending=[False, True]).drop(columns=["_tie", "_skill_r"])
    agg.attrs["total_windows"] = len(origins)
    return agg, details, ctx.get("arima_spec", {})


# ---------------------------------------------------------------------------
# Relatório final
# ---------------------------------------------------------------------------


@dataclass
class ModelForecast:
    name: str
    mean: pd.Series
    lower80: pd.Series
    upper80: pd.Series
    lower95: pd.Series
    upper95: pd.Series
    info: dict = field(default_factory=dict)


@dataclass
class ForecastReport:
    horizon: int
    last_date: pd.Timestamp
    last_price: float
    forecasts: dict[str, ModelForecast]
    metrics: pd.DataFrame
    details: pd.DataFrame
    diagnostics: dict
    best_model: str
    beats_naive: list[str]
    warnings: list[str]


def _to_price_series(fc: LogForecast, index: pd.DatetimeIndex, name: str) -> ModelForecast:
    s = lambda arr: pd.Series(np.exp(arr), index=index)  # noqa: E731
    return ModelForecast(name, s(fc.mean), s(fc.lo80), s(fc.hi80), s(fc.lo95), s(fc.hi95), fc.info)


def diagnostics(y: np.ndarray) -> dict:
    from statsmodels.tsa.stattools import adfuller

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out["adf_log_price_p"] = float(adfuller(y, autolag="AIC")[1])
            out["adf_returns_p"] = float(adfuller(np.diff(y), autolag="AIC")[1])
        except Exception:
            pass
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            out["ljung_box_returns_p"] = float(acorr_ljungbox(np.diff(y), lags=[10])["lb_pvalue"].iloc[0])
        except Exception:
            pass
    return out


def run_forecast(
    close: pd.Series, horizon: int = 30, models: list[str] | tuple[str, ...] = DEFAULT_MODELS, folds: int = 6
) -> ForecastReport:
    """Executa backtest + previsão final para os modelos escolhidos."""
    close = close.dropna()
    close = close[close > 0]
    models = [m for m in models if m in MODEL_FUNCS]
    if NAIVE not in models:
        models = [NAIVE, *models]  # a referência é sempre avaliada
    if len(close) < 60 + horizon:
        raise ValueError("Histórico insuficiente para gerar previsões (mínimo de ~60 dias + horizonte).")

    y = np.log(close.to_numpy(dtype=float))
    warn: list[str] = []

    metrics, details, arima_spec_bt = walk_forward(y, horizon, models, folds=folds)
    if not details.empty:
        origin_dates = close.index[details["origin_idx"].to_numpy() - 1]
        details["origem"] = origin_dates
        details = details.drop(columns="origin_idx")

    ctx = {"arima_spec": select_arima_order(y)} if ARIMA in models else {}
    future_index = pd.date_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecasts: dict[str, ModelForecast] = {}
    raw: list[LogForecast] = []
    for name in models:
        fc, err = _safe_forecast(name, y, horizon, ctx)
        if err:
            warn.append(err)
            continue
        raw.append(fc)
        forecasts[name] = _to_price_series(fc, future_index, name)
    if len(raw) > 1:
        forecasts[ENSEMBLE] = _to_price_series(combine(raw), future_index, ENSEMBLE)

    diag = diagnostics(y)
    if ARIMA in forecasts:
        diag["arima"] = forecasts[ARIMA].info
    diag["arima_backtest_spec"] = arima_spec_bt

    total = metrics.attrs.get("total_windows", int(metrics["Janelas"].max()) if len(metrics) else 0)
    for name in metrics.index:
        n = int(metrics.loc[name, "Janelas"])
        if n < total:
            warn.append(f"{name}: avaliado em {n} de {total} janelas (histórico curto para treinar nas primeiras).")
    reliable = [m for m in metrics.index if m in forecasts and metrics.loc[m, "Janelas"] >= MIN_WINDOWS]
    best = reliable[0] if reliable else NAIVE
    beats = [m for m in reliable if m != NAIVE and metrics.loc[m, "Skill"] > 0]
    return ForecastReport(
        horizon=horizon,
        last_date=close.index[-1],
        last_price=float(close.iloc[-1]),
        forecasts=forecasts,
        metrics=metrics,
        details=details,
        diagnostics=diag,
        best_model=best,
        beats_naive=beats,
        warnings=warn,
    )


class CryptoForecaster:
    """Wrapper orientado a objeto (compatibilidade com a versão anterior)."""

    def forecast(self, df: pd.DataFrame, forecast_days: int = 30, models=DEFAULT_MODELS) -> ForecastReport:
        col = "close" if "close" in df.columns else "price"
        return run_forecast(df[col], forecast_days, models)
