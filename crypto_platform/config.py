"""Configurações centrais: catálogo de moedas, moedas de cotação e parâmetros padrão."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Dias por ano usados para anualizar métricas. Cripto negocia 24/7, então 365.
PERIODS_PER_YEAR = 365


@dataclass(frozen=True)
class Coin:
    """Uma criptomoeda e seus identificadores em cada fonte de dados."""

    name: str
    symbol: str
    coingecko_id: str
    yahoo_ticker: str | None = None  # ticker em USD no Yahoo Finance (ex.: BTC-USD)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.symbol})"


COINS: tuple[Coin, ...] = (
    Coin("Bitcoin", "BTC", "bitcoin", "BTC-USD"),
    Coin("Ethereum", "ETH", "ethereum", "ETH-USD"),
    Coin("Solana", "SOL", "solana", "SOL-USD"),
    Coin("XRP", "XRP", "ripple", "XRP-USD"),
    Coin("BNB", "BNB", "binancecoin", "BNB-USD"),
    Coin("Cardano", "ADA", "cardano", "ADA-USD"),
    Coin("Dogecoin", "DOGE", "dogecoin", "DOGE-USD"),
    Coin("TRON", "TRX", "tron", "TRX-USD"),
    Coin("Chainlink", "LINK", "chainlink", "LINK-USD"),
    Coin("Avalanche", "AVAX", "avalanche-2", "AVAX-USD"),
    Coin("Polkadot", "DOT", "polkadot", "DOT-USD"),
    Coin("Litecoin", "LTC", "litecoin", "LTC-USD"),
    Coin("Bitcoin Cash", "BCH", "bitcoin-cash", "BCH-USD"),
    Coin("Stellar", "XLM", "stellar", "XLM-USD"),
    # MATIC migrou para POL em 2024; o ID antigo (matic-network) parou de ser atualizado.
    Coin("POL (ex-MATIC)", "POL", "polygon-ecosystem-token", None),
)

COINS_BY_ID: dict[str, Coin] = {c.coingecko_id: c for c in COINS}
BENCHMARK = COINS[0]  # Bitcoin, usado para beta/correlação


@dataclass(frozen=True)
class Currency:
    code: str  # código usado pela CoinGecko (minúsculo)
    symbol: str
    name: str
    default_risk_free: float  # taxa livre de risco anual aproximada (editável na interface)


CURRENCIES: dict[str, Currency] = {
    "usd": Currency("usd", "US$", "Dólar americano", 0.04),
    "brl": Currency("brl", "R$", "Real brasileiro", 0.14),
    "eur": Currency("eur", "€", "Euro", 0.02),
}

# Opções de histórico (dias). Acima de 365 a CoinGecko gratuita não atende;
# a camada de dados troca automaticamente para o Yahoo Finance.
HISTORY_OPTIONS = (180, 365, 730, 1095, 1825)
DEFAULT_HISTORY_DAYS = 365
COINGECKO_FREE_MAX_DAYS = 365

DEFAULT_FEE = 0.001  # 0,1% por operação
DEFAULT_SLIPPAGE = 0.001  # 0,1%

# Cache em disco: dados "frescos" por 1 hora; dados antigos (até 7 dias) só
# são usados como último recurso quando todas as fontes falham.
CACHE_DIR = os.environ.get("CRYPTO_PLATFORM_CACHE_DIR", ".cache")
CACHE_FRESH_SECONDS = 60 * 60
CACHE_STALE_MAX_SECONDS = 7 * 24 * 60 * 60

# Modo demonstração (dados sintéticos) — usado em testes automatizados.
DEMO_MODE = os.environ.get("CRYPTO_PLATFORM_DEMO", "").lower() in {"1", "true", "yes"}


def find_coin(coingecko_id: str) -> Coin:
    """Retorna a moeda do catálogo ou cria uma entrada genérica para IDs personalizados."""
    coin = COINS_BY_ID.get(coingecko_id)
    if coin is not None:
        return coin
    clean = coingecko_id.strip().lower()
    return Coin(clean.replace("-", " ").title(), clean.split("-")[0].upper()[:6], clean, None)
