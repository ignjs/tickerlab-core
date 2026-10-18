from __future__ import annotations

from datetime import date, timedelta
import hashlib

import numpy as np
import polars as pl

REQUIRED_OHLCV = ["ts", "open", "high", "low", "close", "volume"]


def normalize_bars(df: pl.DataFrame, symbol: str, timeframe: str) -> pl.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    aliases = {
        "date": "ts",
        "datetime": "ts",
        "timestamp": "ts",
        "adj close": "adj_close",
        "adjclose": "adj_close",
    }
    renames: dict[str, str] = {}
    for lower, original in cols.items():
        if lower in aliases:
            renames[original] = aliases[lower]
        elif lower != original:
            renames[original] = lower
    if renames:
        df = df.rename(renames)

    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    if "adj_close" not in df.columns:
        df = df.with_columns(pl.col("close").alias("adj_close"))

    return (
        df.select(["ts", "open", "high", "low", "close", "adj_close", "volume"])
        .with_columns(
            pl.col("ts").cast(pl.Datetime),
            pl.lit(symbol).alias("symbol"),
            pl.lit(timeframe).alias("timeframe"),
        )
        .sort("ts")
    )


def load_from_csv(path: str, symbol: str, timeframe: str) -> pl.DataFrame:
    df = pl.read_csv(path, try_parse_dates=True)
    return normalize_bars(df, symbol=symbol, timeframe=timeframe)


def load_synthetic(symbol: str, timeframe: str, bars: int = 800, seed: int | None = None) -> pl.DataFrame:
    if seed is None:
        seed = int(hashlib.sha256(symbol.encode("utf-8")).hexdigest(), 16) % (2**32)
    np.random.seed(seed)
    start = date(2015, 1, 1)
    ts = [start + timedelta(days=i) for i in range(bars)]
    rets = np.random.normal(0.0004, 0.015, size=bars)
    close = 100 * np.cumprod(1 + rets)
    open_ = close * (1 + np.random.normal(0, 0.003, size=bars))
    spread = np.maximum(0.001, np.random.normal(0.01, 0.004, size=bars))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = np.random.lognormal(14, 0.2, size=bars)
    raw = pl.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return normalize_bars(raw, symbol=symbol, timeframe=timeframe)


def load_from_yahoo(symbol: str, timeframe: str = "1d", period: str = "max") -> pl.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance no está instalado. Ejecuta: pip install yfinance")

    interval = timeframe.lower()
    ticker = yf.Ticker(symbol)
    pdf = ticker.history(period=period, interval=interval)
    if pdf.empty:
        raise ValueError(
            f"No se pudieron descargar datos de Yahoo Finance para el símbolo: '{symbol}' "
            f"(timeframe: '{timeframe}', period: '{period}')"
        )
    pdf = pdf.reset_index()
    pl_df = pl.from_pandas(pdf)
    return normalize_bars(pl_df, symbol=symbol, timeframe=timeframe)


def get_realtime_quote(symbol: str) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance no está instalado. Ejecuta: pip install yfinance")

    ticker = yf.Ticker(symbol)
    try:
        fi = ticker.fast_info
        last_price = getattr(fi, "last_price", None)
    except Exception as exc:
        raise ValueError(f"No se pudo consultar el símbolo '{symbol}': {exc}")

    if last_price is None or (isinstance(last_price, float) and last_price != last_price):
        raise ValueError(f"No se encontraron datos en tiempo real para el símbolo '{symbol}'.")

    prev_close = getattr(fi, "previous_close", None)
    open_price = getattr(fi, "open", None)
    day_high = getattr(fi, "day_high", None)
    day_low = getattr(fi, "day_low", None)
    volume = getattr(fi, "last_volume", None)
    currency = getattr(fi, "currency", "USD") or "USD"
    tz = getattr(fi, "timezone", "UTC") or "UTC"

    change = None
    change_pct = None
    if prev_close and prev_close != 0 and last_price is not None:
        change = float(last_price - prev_close)
        change_pct = float((change / prev_close) * 100)

    from datetime import datetime

    return {
        "symbol": symbol.upper(),
        "price": float(last_price),
        "change": change,
        "change_pct": change_pct,
        "prev_close": float(prev_close) if prev_close is not None else None,
        "open": float(open_price) if open_price is not None else None,
        "day_high": float(day_high) if day_high is not None else None,
        "day_low": float(day_low) if day_low is not None else None,
        "volume": int(volume) if volume is not None else None,
        "currency": str(currency),
        "timezone": str(tz),
        "timestamp": datetime.now().isoformat(),
    }
