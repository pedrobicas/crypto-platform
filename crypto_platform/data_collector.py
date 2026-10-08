"""Coleta de dados de mercado com múltiplas fontes, cache com validade e tolerância a falhas.

Fontes:
- **CoinGecko** (padrão): preço de fechamento diário, volume e market cap. A API
  gratuita (com ou sem chave demo) limita o histórico a 365 dias.
- **Yahoo Finance** (via ``yfinance``): candles OHLCV diários com histórico longo.
  Permite indicadores que dependem de máxima/mínima (ATR, Estocástico, ADX reais).
- **Sintética**: dados gerados, apenas para testes/demonstração offline.

Em modo automático a coleta tenta as fontes em ordem e, se todas falharem, usa o
último dado salvo em disco (marcado como desatualizado).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    CACHE_FRESH_SECONDS,
    CACHE_STALE_MAX_SECONDS,
    COINGECKO_FREE_MAX_DAYS,
    CURRENCIES,
    DEMO_MODE,
    Coin,
)

logger = logging.getLogger(__name__)

SOURCE_AUTO = "auto"
SOURCE_COINGECKO = "coingecko"
SOURCE_YAHOO = "yahoo"
SOURCE_SYNTHETIC = "synthetic"

SOURCE_LABELS = {
    SOURCE_AUTO: "Automático",
    SOURCE_COINGECKO: "CoinGecko",
    SOURCE_YAHOO: "Yahoo Finance",
    SOURCE_SYNTHETIC: "Dados sintéticos (demo)",
}

USER_AGENT = "CryptoAnalysisPlatform/2.0 (+https://github.com/)"


class DataSourceError(RuntimeError):
    """Falha ao obter dados de uma fonte (mensagem já pensada para o usuário)."""


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------


@dataclass
class DataQuality:
    points: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    filled_days: int = 0
    dropped_rows: int = 0
    outlier_dates: list[str] = field(default_factory=list)
    longest_gap: int = 0  # maior sequência de dias sem cotação na fonte
    trimmed_start: str | None = None  # se o início foi descartado por causa de uma lacuna longa

    @property
    def is_clean(self) -> bool:
        return self.filled_days == 0 and self.dropped_rows == 0 and not self.outlier_dates


@dataclass
class MarketData:
    coin: Coin
    currency: str
    df: pd.DataFrame
    source: str
    has_ohlc: bool
    fetched_at: datetime
    from_cache: bool = False
    stale: bool = False
    quality: DataQuality | None = None
    notices: list[str] = field(default_factory=list)

    @property
    def close(self) -> pd.Series:
        return self.df["close"]

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)


# ---------------------------------------------------------------------------
# Normalização e qualidade dos dados
# ---------------------------------------------------------------------------


def normalize_ohlcv(raw: pd.DataFrame, max_ffill_days: int = 5) -> tuple[pd.DataFrame, DataQuality]:
    """Padroniza um DataFrame de preços em frequência diária.

    - índice diário sem fuso (UTC), um registro por dia (o último do dia prevalece —
      na CoinGecko é o preço "ao vivo" do dia corrente);
    - remove preços nulos/não positivos;
    - preenche dias faltantes repetindo o último preço (volume zero), sem nunca
      reescalar a série — o preço atual exibido é sempre o preço real;
    - calcula retornos simples e logarítmicos;
    - apenas *sinaliza* movimentos extremos (não altera dados reais).
    """
    if raw is None or raw.empty or "close" not in raw.columns:
        raise DataSourceError("A fonte não retornou preços.")

    df = raw.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    numeric_cols = [c for c in ("open", "high", "low", "close", "volume", "market_cap") if c in df.columns]
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    before = len(df)
    df = df[df["close"].notna() & (df["close"] > 0)]
    dropped = before - len(df)
    if len(df) < 2:
        raise DataSourceError("Dados insuficientes retornados pela fonte.")

    # Lacunas: uma lacuna longa (> max_ffill_days) no meio da série distorceria
    # volatilidade e retornos; se sobrar histórico suficiente depois dela, a análise
    # usa só o trecho contínuo mais recente. Lacunas curtas repetem o último preço.
    gaps = (df.index.to_series().diff().dt.days - 1).fillna(0).astype(int)
    longest_gap = int(gaps.max()) if len(gaps) else 0
    trimmed_start = None
    long_gaps = gaps[gaps > max_ffill_days]
    if len(long_gaps):
        cut = long_gaps.index[-1]
        if (df.index >= cut).sum() >= 60:
            df = df[df.index >= cut]
            trimmed_start = cut.strftime("%Y-%m-%d")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full_index.difference(df.index)
    df = df.reindex(full_index)
    if len(missing):
        df["close"] = df["close"].ffill()
        for col in ("open", "high", "low"):
            if col in df.columns:
                df[col] = df[col].fillna(df["close"])
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)
        if "market_cap" in df.columns:
            df["market_cap"] = df["market_cap"].ffill()

    # Garante consistência OHLC (máxima >= mínima, etc.)
    if {"open", "high", "low"}.issubset(df.columns):
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"]).diff()
    df.index.name = "date"
    df.index.freq = None

    lr = df["log_returns"].dropna()
    outliers: list[str] = []
    if len(lr) > 30:
        mad = (lr - lr.median()).abs().median() * 1.4826
        threshold = max(0.25, 8 * mad) if mad > 0 else 0.25
        outliers = [d.strftime("%Y-%m-%d") for d in lr[lr.abs() > threshold].index]

    quality = DataQuality(
        points=len(df),
        start=df.index.min(),
        end=df.index.max(),
        filled_days=int(len(missing)),
        dropped_rows=int(dropped),
        outlier_dates=outliers,
        longest_gap=longest_gap,
        trimmed_start=trimmed_start,
    )
    return df, quality


# ---------------------------------------------------------------------------
# Cache em disco com validade
# ---------------------------------------------------------------------------


class DiskCache:
    """Cache simples em Parquet + metadados JSON, com escrita atômica."""

    def __init__(self, directory: str | os.PathLike | None):
        self.directory = os.fspath(directory) if directory else None
        if self.directory:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except OSError:
                self.directory = None

    def _paths(self, key: str) -> tuple[str, str]:
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)[:80]
        base = os.path.join(self.directory or "", f"{safe}_{digest}")
        return base + ".parquet", base + ".json"

    def load(self, key: str) -> tuple[pd.DataFrame, dict] | None:
        if not self.directory:
            return None
        data_path, meta_path = self._paths(key)
        if not (os.path.exists(data_path) and os.path.exists(meta_path)):
            return None
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            df = pd.read_parquet(data_path)
            return df, meta
        except Exception as exc:  # cache corrompido não deve derrubar o app
            logger.warning("Cache ilegível (%s): %s", key, exc)
            return None

    def save(self, key: str, df: pd.DataFrame, meta: dict) -> None:
        if not self.directory:
            return
        data_path, meta_path = self._paths(key)
        try:
            df.to_parquet(data_path + ".tmp")
            os.replace(data_path + ".tmp", data_path)
            with open(meta_path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(meta, fh)
            os.replace(meta_path + ".tmp", meta_path)
        except Exception as exc:
            logger.warning("Não foi possível salvar cache (%s): %s", key, exc)

    @staticmethod
    def age_seconds(meta: dict) -> float:
        try:
            return time.time() - float(meta["saved_at"])
        except (KeyError, TypeError, ValueError):
            return float("inf")


# ---------------------------------------------------------------------------
# Fontes de dados
# ---------------------------------------------------------------------------


class CoinGeckoProvider:
    name = SOURCE_COINGECKO

    def __init__(
        self,
        session: requests.Session | None = None,
        demo_api_key: str | None = None,
        pro_api_key: str | None = None,
        timeout: float = 20,
        max_retries: int = 3,
        max_wait: float = 10,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.session.headers.setdefault("Accept", "application/json")
        self.demo_api_key = demo_api_key or None
        self.pro_api_key = pro_api_key or None
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._sleep = sleep

    @property
    def base_url(self) -> str:
        if self.pro_api_key:
            return "https://pro-api.coingecko.com/api/v3"
        return "https://api.coingecko.com/api/v3"

    @property
    def max_days(self) -> int | None:
        return None if self.pro_api_key else COINGECKO_FREE_MAX_DAYS

    def _headers(self) -> dict:
        if self.pro_api_key:
            return {"x-cg-pro-api-key": self.pro_api_key}
        if self.demo_api_key:
            return {"x-cg-demo-api-key": self.demo_api_key}
        return {}

    def _get_json(self, path: str, params: dict | None = None):
        url = f"{self.base_url}{path}"
        last_error = "erro desconhecido"
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = f"falha de conexão ({exc.__class__.__name__})"
                if attempt < self.max_retries - 1:
                    self._sleep(min(2**attempt, self.max_wait))
                continue

            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise DataSourceError("CoinGecko retornou uma resposta inválida.") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = (
                    "limite de requisições atingido (HTTP 429)"
                    if resp.status_code == 429
                    else f"erro no servidor (HTTP {resp.status_code})"
                )
                if attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = int(retry_after)
                    self._sleep(min(wait, self.max_wait))
                continue

            detail = _extract_error_message(resp)
            if resp.status_code == 404:
                raise DataSourceError(f"Moeda não encontrada na CoinGecko ({detail}). Confira o ID.")
            if resp.status_code in (401, 403):
                raise DataSourceError(
                    f"A CoinGecko recusou a requisição (HTTP {resp.status_code}: {detail}). "
                    "Verifique a chave de API ou reduza o período de histórico."
                )
            raise DataSourceError(f"CoinGecko respondeu HTTP {resp.status_code}: {detail}")
        raise DataSourceError(f"CoinGecko indisponível: {last_error}.")

    def history(self, coin: Coin, days: int, currency: str) -> pd.DataFrame:
        payload = self._get_json(
            f"/coins/{coin.coingecko_id}/market_chart",
            {"vs_currency": currency, "days": int(days), "interval": "daily"},
        )
        prices = payload.get("prices") or []
        if not prices:
            raise DataSourceError("A CoinGecko não retornou preços para esta moeda.")

        def to_series(key: str) -> pd.Series:
            rows = payload.get(key) or []
            if not rows:
                return pd.Series(dtype=float)
            arr = np.asarray(rows, dtype=float)
            return pd.Series(arr[:, 1], index=pd.to_datetime(arr[:, 0], unit="ms", utc=True))

        close = to_series("prices")
        df = pd.DataFrame({"close": close})
        df["volume"] = to_series("total_volumes").reindex(df.index)
        df["market_cap"] = to_series("market_caps").reindex(df.index)
        # Os pontos diários da CoinGecko ficam às 00:00 UTC e representam o fechamento do
        # dia anterior; o último ponto (fora da meia-noite) é o preço ao vivo de hoje.
        idx = pd.DatetimeIndex(df.index)
        at_midnight = (idx - idx.normalize()) < pd.Timedelta(minutes=30)
        df.index = idx.where(~at_midnight, idx.normalize() - pd.Timedelta(days=1))
        return df

    def markets(self, coin_ids: Iterable[str], currency: str) -> list[dict]:
        ids = ",".join(coin_ids)
        return self._get_json(
            "/coins/markets",
            {
                "vs_currency": currency,
                "ids": ids,
                "order": "market_cap_desc",
                "price_change_percentage": "24h,7d,30d,1y",
                "sparkline": "false",
            },
        )


def _extract_error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:120] or resp.reason or "sem detalhes"
    if isinstance(body, dict):
        if isinstance(body.get("error"), str):
            return body["error"]
        status = body.get("status")
        if isinstance(status, dict) and status.get("error_message"):
            return str(status["error_message"])
    return str(body)[:120]


class YahooProvider:
    name = SOURCE_YAHOO
    max_days = None

    def __init__(self, ticker_factory: Callable | None = None):
        self._ticker_factory = ticker_factory

    def _ticker(self, symbol: str):
        if self._ticker_factory is not None:
            return self._ticker_factory(symbol)
        import yfinance as yf  # import tardio: só carrega se for usado

        return yf.Ticker(symbol)

    def _download(self, symbol: str, start: datetime) -> pd.DataFrame:
        try:
            hist = self._ticker(symbol).history(
                start=start.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                actions=False,
                raise_errors=True,
            )
        except Exception as exc:
            name = exc.__class__.__name__
            if "RateLimit" in name:
                raise DataSourceError("Yahoo Finance: limite de requisições atingido.") from exc
            raise DataSourceError(f"Yahoo Finance indisponível ({name}).") from exc
        if hist is None or hist.empty:
            raise DataSourceError(f"Yahoo Finance não retornou dados para {symbol}.")
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist

    def history(self, coin: Coin, days: int, currency: str) -> pd.DataFrame:
        if not coin.yahoo_ticker:
            raise DataSourceError(f"{coin.name} não tem ticker conhecido no Yahoo Finance.")
        start = datetime.now(timezone.utc) - timedelta(days=int(days) + 1)
        hist = self._download(coin.yahoo_ticker, start)
        df = pd.DataFrame(
            {
                "open": hist["Open"],
                "high": hist["High"],
                "low": hist["Low"],
                "close": hist["Close"],
                "volume": hist["Volume"],
            }
        )
        if currency != "usd":
            fx = self._download(f"{currency.upper()}=X", start - timedelta(days=10))["Close"]
            fx_idx = pd.DatetimeIndex(fx.index)
            if fx_idx.tz is not None:
                # mantém a data do calendário local (converter para UTC jogaria a meia-noite
                # de Londres no horário de verão para o dia anterior)
                fx_idx = fx_idx.tz_localize(None)
            fx.index = fx_idx.normalize()
            fx = fx[~fx.index.duplicated(keep="last")].sort_index()
            target = pd.DatetimeIndex(df.index)
            if target.tz is not None:
                target = target.tz_convert("UTC").tz_localize(None)
            rate = fx.reindex(fx.index.union(target.normalize())).ffill().bfill().reindex(target.normalize())
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].to_numpy() * rate.to_numpy()
        return df


class SyntheticProvider:
    """Gera séries realistas (volatilidade em clusters, caudas pesadas) para testes/demonstração."""

    name = SOURCE_SYNTHETIC
    max_days = None

    BASE_PRICES = {
        "bitcoin": 65000.0,
        "ethereum": 3200.0,
        "solana": 150.0,
        "ripple": 0.6,
        "binancecoin": 580.0,
        "cardano": 0.45,
        "dogecoin": 0.12,
        "tron": 0.12,
        "chainlink": 14.0,
        "avalanche-2": 28.0,
        "polkadot": 6.0,
        "litecoin": 75.0,
        "bitcoin-cash": 380.0,
        "stellar": 0.11,
        "polygon-ecosystem-token": 0.4,
    }
    SUPPLY = {
        "bitcoin": 19.9e6,
        "ethereum": 120e6,
        "solana": 520e6,
        "ripple": 59e9,
        "binancecoin": 140e6,
        "cardano": 36e9,
        "dogecoin": 150e9,
        "tron": 95e9,
        "chainlink": 680e6,
        "avalanche-2": 420e6,
        "polkadot": 1.6e9,
        "litecoin": 76e6,
        "bitcoin-cash": 19.9e6,
        "stellar": 31e9,
        "polygon-ecosystem-token": 10.5e9,
    }
    FX = {"usd": 1.0, "brl": 5.4, "eur": 0.92}

    @staticmethod
    def _seed(text: str) -> int:
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

    def history(self, coin: Coin, days: int, currency: str) -> pd.DataFrame:
        n = int(days) + 1
        end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        idx = pd.date_range(end=end, periods=n, freq="D")

        # Fator de mercado comum (BTC) + componente idiossincrático => correlações realistas
        market = self._garch_returns(n, self._seed("market"), 0.028)
        own = self._garch_returns(n, self._seed(coin.coingecko_id), 0.02)
        beta = 1.0 if coin.coingecko_id == "bitcoin" else 1.2
        idio = 0.0 if coin.coingecko_id == "bitcoin" else 1.0
        rets = beta * market + idio * own

        base = self.BASE_PRICES.get(coin.coingecko_id, 1.0) * self.FX.get(currency, 1.0)
        log_price = np.log(base) + np.cumsum(rets) - np.cumsum(rets)[-1]  # termina perto do preço base
        close = np.exp(log_price)

        rng = np.random.default_rng(self._seed(coin.coingecko_id + "ohlc"))
        open_ = np.empty(n)
        open_[0] = close[0]
        open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.002, n - 1))
        spread = np.abs(rets) + np.abs(rng.normal(0, 0.012, n))
        high = np.maximum(open_, close) * np.exp(spread * rng.uniform(0.2, 0.7, n))
        low = np.minimum(open_, close) * np.exp(-spread * rng.uniform(0.2, 0.7, n))
        supply = self.SUPPLY.get(coin.coingecko_id, 1e9)
        volume = close * supply * 0.03 * np.exp(rng.normal(0, 0.35, n)) * (1 + 8 * np.abs(rets))
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "market_cap": close * supply},
            index=idx,
        )

    @staticmethod
    def _garch_returns(n: int, seed: int, base_vol: float) -> np.ndarray:
        rng = np.random.default_rng(seed)
        omega, alpha, beta = base_vol**2 * 0.06, 0.09, 0.85
        var = base_vol**2
        out = np.empty(n)
        shocks = rng.standard_t(df=4, size=n) / np.sqrt(2.0)
        regime = np.repeat(rng.normal(0.0004, 0.0025, size=n // 60 + 1), 60)[:n]
        for i in range(n):
            out[i] = regime[i] + np.sqrt(var) * shocks[i]
            var = omega + alpha * (out[i] - regime[i]) ** 2 + beta * var
        return out

    def markets(self, coin_ids: Iterable[str], currency: str) -> list[dict]:
        rows = []
        from .config import find_coin

        for rank, cid in enumerate(coin_ids, start=1):
            coin = find_coin(cid)
            df = self.history(coin, 400, currency)
            close = df["close"]
            price = float(close.iloc[-1])
            ath = float(df["high"].max())

            def chg(days: int, close: pd.Series = close) -> float:
                return float((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)

            rows.append(
                {
                    "id": cid,
                    "symbol": coin.symbol.lower(),
                    "name": coin.name,
                    "image": None,
                    "current_price": price,
                    "market_cap": float(df["market_cap"].iloc[-1]),
                    "market_cap_rank": rank,
                    "total_volume": float(df["volume"].iloc[-1]),
                    "high_24h": float(df["high"].iloc[-1]),
                    "low_24h": float(df["low"].iloc[-1]),
                    "price_change_percentage_24h": chg(1),
                    "price_change_percentage_7d_in_currency": chg(7),
                    "price_change_percentage_30d_in_currency": chg(30),
                    "price_change_percentage_1y_in_currency": chg(365),
                    "circulating_supply": float(df["market_cap"].iloc[-1] / price),
                    "max_supply": None,
                    "ath": ath,
                    "ath_change_percentage": (price / ath - 1) * 100,
                    "ath_date": df["high"].idxmax().strftime("%Y-%m-%dT00:00:00.000Z"),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# Fachada
# ---------------------------------------------------------------------------


FEAR_GREED_PT = {
    "Extreme Fear": "Medo extremo",
    "Fear": "Medo",
    "Neutral": "Neutro",
    "Greed": "Ganância",
    "Extreme Greed": "Ganância extrema",
}


class CryptoDataCollector:
    """Ponto único de acesso aos dados de mercado."""

    def __init__(
        self,
        cache_dir: str | None = CACHE_DIR,
        coingecko_demo_key: str | None = None,
        coingecko_pro_key: str | None = None,
        session: requests.Session | None = None,
        demo: bool = DEMO_MODE,
        sleep: Callable[[float], None] = time.sleep,
        yahoo_ticker_factory: Callable | None = None,
    ):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.cache = DiskCache(cache_dir)
        self.demo = demo
        self.coingecko = CoinGeckoProvider(
            self.session,
            demo_api_key=coingecko_demo_key or os.environ.get("COINGECKO_DEMO_API_KEY"),
            pro_api_key=coingecko_pro_key or os.environ.get("COINGECKO_PRO_API_KEY"),
            sleep=sleep,
        )
        self.yahoo = YahooProvider(yahoo_ticker_factory)
        self.synthetic = SyntheticProvider()

    # -- histórico -------------------------------------------------------

    def _provider_order(self, coin: Coin, days: int, source: str) -> list:
        if self.demo or source == SOURCE_SYNTHETIC:
            return [self.synthetic]
        if source == SOURCE_COINGECKO:
            return [self.coingecko]
        if source == SOURCE_YAHOO:
            return [self.yahoo]
        cg_max = self.coingecko.max_days
        order = [self.coingecko, self.yahoo]
        if cg_max is not None and days > cg_max and coin.yahoo_ticker:
            order = [self.yahoo, self.coingecko]
        if not coin.yahoo_ticker:
            order = [self.coingecko]
        return order

    def get_market_data(
        self,
        coin: Coin,
        days: int = 365,
        currency: str = "usd",
        source: str = SOURCE_AUTO,
        min_saved_at: float | None = None,
    ) -> MarketData:
        """Histórico diário da moeda.

        ``min_saved_at`` (timestamp) força nova busca se o cache em disco for
        anterior a esse instante — usado pelo botão "Atualizar dados".
        """
        currency = currency.lower()
        if currency not in CURRENCIES:
            raise DataSourceError(f"Moeda de cotação não suportada: {currency}")
        days = int(days)
        cache_key = f"{source if not self.demo else SOURCE_SYNTHETIC}_{coin.coingecko_id}_{currency}_{days}"

        cached = self.cache.load(cache_key)
        if cached is not None and DiskCache.age_seconds(cached[1]) < CACHE_FRESH_SECONDS:
            df, meta = cached
            if min_saved_at is None or float(meta.get("saved_at", 0)) >= min_saved_at:
                return self._build(coin, currency, df, meta, from_cache=True, stale=False)

        errors: list[str] = []
        for provider in self._provider_order(coin, days, source):
            notices: list[str] = []
            request_days = days
            max_days = provider.max_days
            if max_days is not None and days > max_days:
                request_days = max_days
                notices.append(
                    f"A {SOURCE_LABELS[provider.name]} gratuita fornece no máximo {max_days} dias; o histórico foi limitado a esse período."
                )
            try:
                raw = provider.history(coin, request_days, currency)
            except DataSourceError as exc:
                errors.append(f"{SOURCE_LABELS[provider.name]}: {exc}")
                continue
            except Exception as exc:  # erro inesperado de biblioteca externa
                logger.exception("Falha inesperada em %s", provider.name)
                errors.append(f"{SOURCE_LABELS[provider.name]}: erro inesperado ({exc.__class__.__name__}).")
                continue

            df, quality = normalize_ohlcv(raw)
            if errors:
                notices.insert(0, "Fonte principal indisponível; usando fonte alternativa. " + " | ".join(errors))
            if quality.trimmed_start:
                notices.append(
                    f"A fonte ficou {quality.longest_gap} dias sem cotação; a análise usa apenas o período contínuo "
                    f"a partir de {pd.Timestamp(quality.trimmed_start):%d/%m/%Y}."
                )
            elif quality.longest_gap > 5:
                notices.append(f"Lacuna de {quality.longest_gap} dias sem cotação preenchida com o último preço.")
            meta = {
                "source": provider.name,
                "saved_at": time.time(),
                "has_ohlc": bool({"open", "high", "low"}.issubset(df.columns)),
                "notices": notices,
                "quality": {
                    "filled_days": quality.filled_days,
                    "dropped_rows": quality.dropped_rows,
                    "longest_gap": quality.longest_gap,
                    "trimmed_start": quality.trimmed_start,
                },
            }
            self.cache.save(cache_key, df, meta)
            return self._build(coin, currency, df, meta, from_cache=False, stale=False)

        if cached is not None and DiskCache.age_seconds(cached[1]) < CACHE_STALE_MAX_SECONDS:
            df, meta = cached
            age_h = DiskCache.age_seconds(meta) / 3600
            meta = dict(meta)
            meta["notices"] = [f"Todas as fontes falharam; exibindo dados salvos há {age_h:.1f} h. Detalhes: " + " | ".join(errors)]
            return self._build(coin, currency, df, meta, from_cache=True, stale=True)

        raise DataSourceError(" | ".join(errors) or "Nenhuma fonte de dados disponível.")

    def _build(self, coin: Coin, currency: str, df: pd.DataFrame, meta: dict, from_cache: bool, stale: bool) -> MarketData:
        df, quality = normalize_ohlcv(df)
        saved_quality = meta.get("quality") or {}
        quality.filled_days = int(saved_quality.get("filled_days", quality.filled_days))
        quality.dropped_rows = int(saved_quality.get("dropped_rows", quality.dropped_rows))
        quality.longest_gap = int(saved_quality.get("longest_gap", quality.longest_gap))
        quality.trimmed_start = saved_quality.get("trimmed_start", quality.trimmed_start)
        return MarketData(
            coin=coin,
            currency=currency,
            df=df,
            source=meta.get("source", "?"),
            has_ohlc=bool(meta.get("has_ohlc")) and {"open", "high", "low"}.issubset(df.columns),
            fetched_at=datetime.fromtimestamp(float(meta.get("saved_at", time.time())), tz=timezone.utc),
            from_cache=from_cache,
            stale=stale,
            quality=quality,
            notices=list(meta.get("notices", [])),
        )

    # -- dados complementares (falham silenciosamente) --------------------

    def get_market_snapshot(self, coin_ids: Iterable[str], currency: str = "usd") -> pd.DataFrame:
        """Preço atual, ranking, máxima histórica (ATH) etc. Retorna DataFrame vazio se falhar."""
        ids = list(coin_ids)
        try:
            rows = (self.synthetic if self.demo else self.coingecko).markets(ids, currency)
        except Exception as exc:
            logger.info("Snapshot indisponível: %s", exc)
            return pd.DataFrame()
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("id")

    def get_fear_greed(self, limit: int = 90) -> pd.DataFrame:
        """Índice Fear & Greed (alternative.me). Colunas: value, classification."""
        if self.demo:
            idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=limit, freq="D")
            rng = np.random.default_rng(7)
            values = np.empty(limit)
            values[0] = 50.0
            for i in range(1, limit):  # AR(1) em torno de 50
                values[i] = 50 + 0.92 * (values[i - 1] - 50) + rng.normal(0, 6)
            values = np.clip(values, 3, 97).round()
            return pd.DataFrame({"value": values, "classification": [_fng_label(v) for v in values]}, index=idx)
        try:
            resp = self.session.get("https://api.alternative.me/fng/", params={"limit": limit}, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data") or []
        except Exception as exc:
            logger.info("Fear & Greed indisponível: %s", exc)
            return pd.DataFrame()
        rows = []
        for item in data:
            try:
                rows.append(
                    {
                        "date": pd.to_datetime(int(item["timestamp"]), unit="s").normalize(),
                        "value": float(item["value"]),
                        "classification": FEAR_GREED_PT.get(item.get("value_classification"), _fng_label(float(item["value"]))),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()


def _fng_label(value: float) -> str:
    if value <= 24:
        return "Medo extremo"
    if value <= 44:
        return "Medo"
    if value <= 55:
        return "Neutro"
    if value <= 75:
        return "Ganância"
    return "Ganância extrema"
