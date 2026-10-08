"""Formatação de números no padrão brasileiro (1.234,56)."""

from __future__ import annotations

import math

import numpy as np

from .config import CURRENCIES


def _is_missing(value) -> bool:
    try:
        return value is None or math.isnan(float(value)) or math.isinf(float(value))
    except (TypeError, ValueError):
        return True


def fmt_number(value, decimals: int = 2) -> str:
    """Formata com separador de milhar '.' e decimal ','."""
    if _is_missing(value):
        return "—"
    value = float(value)
    if round(value, decimals) == 0:
        value = 0.0  # evita "-0,00"
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def price_decimals(value) -> int:
    """Casas decimais adequadas ao tamanho do preço (moedas baratas precisam de mais)."""
    if _is_missing(value):
        return 2
    v = abs(float(value))
    if v >= 1000:
        return 2
    if v >= 1:
        return 3 if v < 10 else 2
    if v >= 0.01:
        return 4
    if v >= 0.0001:
        return 6
    if v == 0:
        return 2
    return int(min(12, -np.floor(np.log10(v)) + 3))  # ~4 algarismos significativos


def fmt_price(value, currency: str = "usd") -> str:
    symbol = CURRENCIES.get(currency, CURRENCIES["usd"]).symbol
    if _is_missing(value):
        return "—"
    return f"{symbol} {fmt_number(value, price_decimals(value))}"


def fmt_money(value, currency: str = "usd", decimals: int = 2) -> str:
    symbol = CURRENCIES.get(currency, CURRENCIES["usd"]).symbol
    if _is_missing(value):
        return "—"
    sign = "-" if float(value) < 0 else ""
    return f"{sign}{symbol} {fmt_number(abs(float(value)), decimals)}"


def fmt_compact(value, currency: str | None = None) -> str:
    """Números grandes de forma compacta: 1,23 tri / 45,6 bi / 7,8 mi / 12,3 mil."""
    if _is_missing(value):
        return "—"
    v = float(value)
    sign = "-" if v < 0 else ""
    v = abs(v)
    for limit, suffix in ((1e12, "tri"), (1e9, "bi"), (1e6, "mi"), (1e3, "mil")):
        if v >= limit:
            body = f"{fmt_number(v / limit, 2)} {suffix}"
            break
    else:
        body = fmt_number(v, 2)
    if currency:
        symbol = CURRENCIES.get(currency, CURRENCIES["usd"]).symbol
        return f"{sign}{symbol} {body}"
    return f"{sign}{body}"


def fmt_pct(value, decimals: int = 2, signed: bool = False) -> str:
    """Formata percentuais já em escala 0-100 (ex.: 12.5 -> '12,50%')."""
    if _is_missing(value):
        return "—"
    text = fmt_number(value, decimals) + "%"
    if signed and round(float(value), decimals) > 0:
        text = "+" + text
    return text


def fmt_ratio(value, decimals: int = 2) -> str:
    if _is_missing(value):
        return "—"
    return fmt_number(value, decimals)


def plotly_tickformat(value) -> str:
    """Formato d3 para eixos de preço, coerente com price_decimals."""
    return f",.{price_decimals(value)}f"
