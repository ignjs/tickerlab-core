from datetime import datetime

import polars as pl

from tickerlab.ingest.service import normalize_bars


def test_normalize_bars_accepts_yahoo_intraday_datetime_column():
    bars = pl.DataFrame(
        {
            "Datetime": [datetime(2026, 10, 18, 13, 30)],
            "Open": [10.0],
            "High": [11.0],
            "Low": [9.0],
            "Close": [10.5],
            "Volume": [100],
        }
    )

    normalized = normalize_bars(bars, symbol="TEST", timeframe="5m")

    assert normalized["ts"][0] == datetime(2026, 10, 18, 13, 30)
    assert normalized["symbol"][0] == "TEST"
    assert normalized["timeframe"][0] == "5m"
