import json
import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from crypto_platform.config import COINS, find_coin
from crypto_platform.data_collector import (
    SOURCE_AUTO,
    SOURCE_COINGECKO,
    SOURCE_YAHOO,
    CryptoDataCollector,
    DataSourceError,
    normalize_ohlcv,
)

BTC, ETH, POL = COINS[0], COINS[1], find_coin("polygon-ecosystem-token")


def _resp(code, body=None, headers=None):
    r = MagicMock()
    r.status_code = code
    r.json.return_value = body if body is not None else {}
    r.headers = headers or {}
    r.text = json.dumps(body or {})
    r.reason = ""
    return r


def _market_chart(days=365, live_hour=18, skip=None):
    """Resposta no formato da CoinGecko: pontos diários 00:00 UTC + ponto 'ao vivo' de hoje."""
    end = pd.Timestamp("2026-10-07", tz="UTC")
    dates = pd.date_range(end=end, periods=days, freq="D")
    if skip is not None:
        dates = dates.delete(skip)
    ts = [int(d.timestamp() * 1000) for d in dates] + [int((end + pd.Timedelta(hours=live_hour)).timestamp() * 1000)]
    prices = [[t, 60000 + i * 10] for i, t in enumerate(ts)]
    return {
        "prices": prices,
        "total_volumes": [[t, 3e10] for t in ts],
        "market_caps": [[t, 1.2e12] for t in ts],
    }


def _collector(tmp_path, responses, **kw):
    session = MagicMock()
    session.headers = {}
    session.get.side_effect = responses
    waits = []
    c = CryptoDataCollector(cache_dir=str(tmp_path), session=session, demo=False, sleep=waits.append, **kw)
    return c, session, waits


# --- normalização -------------------------------------------------------------


def test_normalize_keeps_live_price_and_fills_gaps():
    idx = pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00", "2026-01-04 00:00", "2026-01-04 15:30"], utc=True)
    raw = pd.DataFrame({"close": [100.0, 110.0, 120.0, 125.0], "volume": [1, 1, 1, 1]}, index=idx)
    df, q = normalize_ohlcv(raw)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert df["close"].iloc[-1] == 125.0  # o último ponto do dia (ao vivo) prevalece
    assert df.loc["2026-01-03", "close"] == 110.0  # dia faltante = último preço
    assert df.loc["2026-01-03", "volume"] == 0.0
    assert q.filled_days == 1


def test_normalize_never_rescales_prices_and_only_flags_outliers():
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    close = np.full(60, 100.0)
    close[40:] = 180.0  # salto de +80% em um dia
    df, q = normalize_ohlcv(pd.DataFrame({"close": close}, index=idx))
    assert df["close"].iloc[-1] == 180.0
    assert q.outlier_dates == ["2026-02-10"]


def test_normalize_drops_invalid_prices():
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    df, q = normalize_ohlcv(pd.DataFrame({"close": [100, -1, 0, np.nan, 105]}, index=idx))
    assert (df["close"] > 0).all()
    assert q.dropped_rows == 3


# --- CoinGecko ----------------------------------------------------------------


def test_coingecko_retries_on_429_with_capped_wait(tmp_path):
    c, session, waits = _collector(tmp_path, [_resp(429, headers={"Retry-After": "60"}), _resp(200, _market_chart())])
    md = c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    assert md.source == SOURCE_COINGECKO
    assert waits == [10]  # Retry-After respeitado, mas limitado
    assert md.close.iloc[-1] == 60000 + 365 * 10  # preço ao vivo é o último
    assert not md.has_ohlc


def test_coingecko_gives_up_after_max_retries(tmp_path):
    c, _, waits = _collector(tmp_path, [_resp(429)] * 3)
    with pytest.raises(DataSourceError, match="429"):
        c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    assert len(waits) == 2


def test_coingecko_404_has_friendly_message(tmp_path):
    c, _, _ = _collector(tmp_path, [_resp(404, {"error": "coin not found"})])
    with pytest.raises(DataSourceError, match="não encontrada"):
        c.get_market_data(find_coin("moeda-que-nao-existe"), 365, "usd", SOURCE_COINGECKO)


def test_coingecko_sends_demo_key_header(tmp_path):
    c, session, _ = _collector(tmp_path, [_resp(200, _market_chart())], coingecko_demo_key="CG-teste")
    c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    assert session.get.call_args.kwargs["headers"] == {"x-cg-demo-api-key": "CG-teste"}


def test_coingecko_free_history_is_capped_at_365_days(tmp_path):
    c, session, _ = _collector(tmp_path, [_resp(200, _market_chart())])
    md = c.get_market_data(BTC, 730, "usd", SOURCE_COINGECKO)
    assert session.get.call_args.kwargs["params"]["days"] == 365
    assert any("365" in n for n in md.notices)


# --- cache ----------------------------------------------------------------------


def test_cache_is_reused_while_fresh(tmp_path):
    c, session, _ = _collector(tmp_path, [_resp(200, _market_chart())])
    c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    md2 = c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    assert md2.from_cache and session.get.call_count == 1


def test_refresh_bypasses_fresh_cache(tmp_path):
    c, session, _ = _collector(tmp_path, [_resp(200, _market_chart()), _resp(200, _market_chart())])
    c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    md = c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO, min_saved_at=time.time() + 1)
    assert not md.from_cache and session.get.call_count == 2


def test_stale_cache_is_last_resort(tmp_path, monkeypatch):
    c, session, _ = _collector(tmp_path, [_resp(200, _market_chart())] + [_resp(429)] * 3)
    c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    import crypto_platform.data_collector as dc

    monkeypatch.setattr(dc, "CACHE_FRESH_SECONDS", -1)  # tudo vira "antigo"
    md = c.get_market_data(BTC, 365, "usd", SOURCE_COINGECKO)
    assert md.stale and md.from_cache
    assert "falharam" in md.notices[0]


# --- Yahoo e fallback -----------------------------------------------------------


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=50, freq="D")
        if self.symbol.endswith("=X"):
            return pd.DataFrame({"Close": np.full(50, 5.0)}, index=idx.tz_convert("Europe/London"))
        close = np.linspace(100, 149, 50)
        return pd.DataFrame({"Open": close, "High": close * 1.02, "Low": close * 0.98, "Close": close, "Volume": 1e6}, index=idx)


def test_auto_falls_back_to_yahoo_and_converts_currency(tmp_path):
    c, _, _ = _collector(tmp_path, [_resp(500)] * 3, yahoo_ticker_factory=FakeTicker)
    md = c.get_market_data(ETH, 49, "brl", SOURCE_AUTO)
    assert md.source == SOURCE_YAHOO and md.has_ohlc
    assert md.close.iloc[-1] == pytest.approx(149 * 5.0)  # USD × câmbio
    assert "alternativa" in md.notices[0]


def test_yahoo_unavailable_for_coin_without_ticker(tmp_path):
    c, _, _ = _collector(tmp_path, [], yahoo_ticker_factory=FakeTicker)
    with pytest.raises(DataSourceError, match="ticker"):
        c.get_market_data(POL, 365, "usd", SOURCE_YAHOO)


def test_long_history_prefers_yahoo_in_auto_mode(tmp_path):
    c, session, _ = _collector(tmp_path, [], yahoo_ticker_factory=FakeTicker)
    md = c.get_market_data(BTC, 1095, "usd", SOURCE_AUTO)
    assert md.source == SOURCE_YAHOO and session.get.call_count == 0


# --- extras ---------------------------------------------------------------------


def test_fear_greed_parsing(tmp_path):
    body = {
        "data": [
            {"value": "72", "value_classification": "Greed", "timestamp": "1790000000"},
            {"value": "20", "value_classification": "Extreme Fear", "timestamp": "1789913600"},
        ]
    }
    c, _, _ = _collector(tmp_path, [_resp(200, body)])
    fng = c.get_fear_greed(2)
    assert list(fng["classification"]) == ["Medo extremo", "Ganância"]  # ordenado por data
    assert fng["value"].iloc[-1] == 72


def test_extras_fail_silently(tmp_path):
    c, _, _ = _collector(tmp_path, [_resp(500)] * 3 + [RuntimeError("sem rede")])
    assert c.get_market_snapshot(["bitcoin"]).empty
    assert c.get_fear_greed().empty


def test_demo_mode_generates_consistent_data(tmp_path):
    c = CryptoDataCollector(cache_dir=str(tmp_path), demo=True)
    a = c.get_market_data(BTC, 365)
    b = CryptoDataCollector(cache_dir=None, demo=True).get_market_data(BTC, 365)
    assert len(a.df) == 366 and a.has_ohlc
    pd.testing.assert_series_equal(a.close, b.close, check_freq=False)
