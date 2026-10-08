"""Configuração dos testes: força dados sintéticos e cache temporário (nenhum acesso à internet)."""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CRYPTO_PLATFORM_DEMO", "1")
os.environ.setdefault("CRYPTO_PLATFORM_CACHE_DIR", tempfile.mkdtemp(prefix="crypto_cache_"))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from crypto_platform.config import COINS  # noqa: E402
from crypto_platform.data_collector import SyntheticProvider, normalize_ohlcv  # noqa: E402


@pytest.fixture(scope="session")
def btc_df() -> pd.DataFrame:
    raw = SyntheticProvider().history(COINS[0], 730, "usd")
    df, _ = normalize_ohlcv(raw)
    return df


@pytest.fixture(scope="session")
def btc_close(btc_df) -> pd.Series:
    return btc_df["close"]


@pytest.fixture
def random_walk() -> pd.Series:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400))), index=idx)
