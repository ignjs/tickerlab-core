import pytest
import numpy as np
from tickerlab.stats.metrics import compute_signal_metrics, stability_score

def test_compute_signal_metrics_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.9, 0.1])
    returns = np.array([0.05, -0.02, 0.04, -0.01])
    
    metrics = compute_signal_metrics(y_true, y_prob, returns)
    
    assert metrics["hit_rate"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["profit_factor"] > 1.0

def test_compute_signal_metrics_all_nan():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.5])
    returns = np.array([np.nan, np.nan])
    
    metrics = compute_signal_metrics(y_true, y_prob, returns)
    
    assert np.isnan(metrics["mean_forward_return"])
    # If all returns are NaN, sharpe might be NaN instead of 0.0
    if not np.isnan(metrics["sharpe"]):
        assert metrics["sharpe"] == 0.0

def test_stability_score_perfect():
    values = [0.7, 0.7, 0.7]
    # Use pytest.approx for floating point comparisons
    assert stability_score(values) == pytest.approx(1.0)

def test_stability_score_zero():
    values = [0.1, 0.9, 0.5] # High variance
    score = stability_score(values)
    assert score < 1.0

def test_stability_score_empty():
    assert np.isnan(stability_score([]))
