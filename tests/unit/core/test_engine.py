import pytest
from unittest.mock import MagicMock, patch
from tickerlab.analysis.engine import analyze_ticker

def test_analyze_ticker_basic_flow():
    # We mock the core engine's dependencies to avoid calling actual Polars/YFinance code
    with patch('tickerlab.analysis.engine.normalize_symbol') as mock_norm, \
         patch('tickerlab.analysis.engine.validate_yahoo_symbol') as mock_val, \
         patch('tickerlab.analysis.engine.get_realtime_quote') as mock_quote, \
         patch('tickerlab.analysis.engine.load_from_yahoo') as mock_load, \
         patch('tickerlab.analysis.engine.save_frame') as mock_save, \
         patch('tickerlab.analysis.engine.build_features') as mock_feats, \
         patch('tickerlab.analysis.engine.detect_pivots') as mock_pivots, \
         patch('tickerlab.analysis.engine.scan_structural') as mock_scan, \
         patch('tickerlab.analysis.engine.build_bar_labels') as mock_blab, \
         patch('tickerlab.analysis.engine.build_event_labels') as mock_elab, \
         patch('tickerlab.analysis.engine.event_dataset') as mock_ds, \
         patch('tickerlab.analysis.engine.as_xy') as mock_xy, \
         patch('tickerlab.analysis.engine.fit_logistic') as mock_fit, \
         patch('tickerlab.analysis.engine.compute_signal_metrics') as mock_metrics:
        
        # Setup mocks
        mock_norm.return_value = ("AAPL", None)
        mock_val.return_value = (True, None)
        mock_quote.return_value = {"price": 150.0}
        mock_load.return_value = MagicMock() # Mock Polars DF
        mock_feats.return_value = MagicMock()
        mock_pivots.return_value = MagicMock()
        mock_scan.return_value = MagicMock()
        mock_blab.return_value = MagicMock()
        mock_elab.return_value = MagicMock()
        mock_ds.return_value = MagicMock()
        mock_xy.return_value = (np.array([1]), np.array([1]), np.array([1]), MagicMock())
        mock_fit.return_value = MagicMock()
        mock_fit.return_value.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_metrics.return_value = {"auc": 0.7}
        
        # Execute
        from tickerlab.analysis.engine import analyze_ticker
        result = analyze_ticker("AAPL")
        
        # Assert
        assert result["symbol"] == "AAPL"
        assert "summary" in result
        assert result["summary"]["auc"] == 0.7
