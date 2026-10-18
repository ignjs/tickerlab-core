import pytest
import polars as pl
import numpy as np
from tickerlab.pivots.detector import detect_pivots

def test_detect_pivots_basic():
    # Create a simple price series with a clear peak and valley
    # Indices: 0 1 2 3 4
    # Values:  10 20 10 5 10
    # Peak at index 1 (20 > 10 and 20 > 10)
    # Valley at index 3 (5 < 10 and 5 < 10)
    bars = pl.DataFrame({
        "ts": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "symbol": ["AAPL"] * 5,
        "close": [10.0, 20.0, 10.0, 5.0, 10.0]
    })
    
    pivots = detect_pivots(bars, window=1)
    
    # Expected pivots: High at 20.0, Low at 5.0
    assert len(pivots) == 2
    assert "high" in pivots["pivot_type"].to_list()
    assert "low" in pivots["pivot_type"].to_list()
    assert pivots.filter(pl.col("pivot_type") == "high")["price"][0] == 20.0
    assert pivots.filter(pl.col("pivot_type") == "low")["price"][0] == 5.0

def test_detect_pivots_no_pivots():
    # Monotonic series has no peaks/valleys
    bars = pl.DataFrame({
        "ts": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "symbol": ["AAPL"] * 3,
        "close": [10.0, 11.0, 12.0]
    })
    
    pivots = detect_pivots(bars, window=1)
    assert len(pivots) == 0
    assert "pivot_type" in pivots.columns
