"""
Tests for OptionsFlowAnalyzer module.

Tests options flow analysis, unusual activity detection,
put/call ratio calculation, and options sentiment aggregation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch


# Import will be created when module exists
try:
    from stanley.analytics.options_flow import OptionsFlowAnalyzer
except ImportError:
    OptionsFlowAnalyzer = None


# Skip all tests if module not yet implemented
pytestmark = pytest.mark.skipif(
    OptionsFlowAnalyzer is None,
    reason="OptionsFlowAnalyzer module not yet implemented"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_options_data():
    """Sample options flow data."""
    dates = pd.date_range(end=datetime.now(), periods=20, freq="D")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "symbol": ["AAPL"] * 20,
        "call_volume": np.random.randint(50000, 200000, 20),
        "put_volume": np.random.randint(30000, 150000, 20),
        "call_premium": np.random.uniform(10_000_000, 50_000_000, 20),
        "put_premium": np.random.uniform(5_000_000, 30_000_000, 20),
        "call_oi": np.random.randint(100000, 500000, 20),
        "put_oi": np.random.randint(80000, 400000, 20),
    })


@pytest.fixture
def sample_unusual_activity():
    """Sample unusual options activity data."""
    return pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=10, freq="h"),
        "symbol": ["AAPL"] * 10,
        "option_type": ["call", "put", "call", "call", "put", "call", "put", "call", "call", "put"],
        "strike": [150, 145, 155, 160, 140, 150, 148, 152, 155, 145],
        "expiry": pd.date_range(start=datetime.now() + timedelta(days=30), periods=10, freq="7D"),
        "volume": [50000, 30000, 45000, 60000, 25000, 55000, 35000, 40000, 70000, 20000],
        "open_interest": [5000, 3000, 4000, 6000, 2000, 5500, 3200, 4500, 8000, 1800],
        "premium": [2_500_000, 1_500_000, 2_200_000, 3_000_000, 1_200_000, 2_700_000, 1_700_000, 2_000_000, 3_500_000, 1_000_000],
        "unusual_score": [8.5, 7.2, 8.0, 9.1, 6.5, 8.3, 7.0, 7.8, 9.5, 6.0],
    })


@pytest.fixture
def sample_large_trades():
    """Sample large options trades."""
    return pd.DataFrame({
        "timestamp": pd.date_range(end=datetime.now(), periods=5, freq="2h"),
        "symbol": ["AAPL"] * 5,
        "option_type": ["call", "call", "put", "call", "put"],
        "strike": [150, 155, 145, 160, 140],
        "expiry": pd.date_range(start=datetime.now() + timedelta(days=45), periods=5, freq="14D"),
        "size": [10000, 8000, 12000, 15000, 9000],
        "premium": [5_000_000, 4_000_000, 6_000_000, 7_500_000, 4_500_000],
        "trade_type": ["sweep", "block", "sweep", "sweep", "block"],
        "sentiment": ["bullish", "bullish", "bearish", "bullish", "bearish"],
    })


@pytest.fixture
def empty_options_data():
    """Empty options data DataFrame."""
    return pd.DataFrame(columns=[
        "date", "symbol", "call_volume", "put_volume",
        "call_premium", "put_premium", "call_oi", "put_oi"
    ])


@pytest.fixture
def mock_data_manager_for_options():
    """Mock DataManager for options flow."""
    mock = Mock()
    mock.get_options_flow = AsyncMock(return_value=pd.DataFrame({
        "call_volume": [100000],
        "put_volume": [80000],
        "call_premium": [25_000_000],
        "put_premium": [15_000_000],
    }))
    mock.get_unusual_options_activity = AsyncMock(return_value=pd.DataFrame())
    return mock


# =============================================================================
# Initialization Tests
# =============================================================================


class TestOptionsFlowAnalyzerInit:
    """Tests for OptionsFlowAnalyzer initialization."""

    def test_init_without_data_manager(self):
        """Test initialization without data_manager."""
        analyzer = OptionsFlowAnalyzer()
        assert analyzer is not None
        assert analyzer.data_manager is None

    def test_init_with_data_manager(self, mock_data_manager_for_options):
        """Test initialization with mock data_manager."""
        analyzer = OptionsFlowAnalyzer(data_manager=mock_data_manager_for_options)
        assert analyzer.data_manager is mock_data_manager_for_options

    def test_default_unusual_threshold(self):
        """Test default unusual activity threshold."""
        analyzer = OptionsFlowAnalyzer()
        assert analyzer.unusual_threshold == 2.0  # 2x average volume

    def test_custom_unusual_threshold(self):
        """Test custom unusual activity threshold."""
        analyzer = OptionsFlowAnalyzer(unusual_threshold=3.0)
        assert analyzer.unusual_threshold == 3.0


# =============================================================================
# Unusual Activity Detection Tests
# =============================================================================


class TestUnusualActivityDetection:
    """Tests for unusual_activity detection."""

    def test_returns_dataframe(self):
        """Test that method returns a DataFrame."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.detect_unusual_activity("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        """Test that DataFrame has expected columns."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.detect_unusual_activity("AAPL")
        expected_cols = [
            "symbol", "option_type", "strike", "expiry",
            "volume", "open_interest", "unusual_score"
        ]
        if not result.empty:
            for col in expected_cols:
                assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_unusual_score(self):
        """Test that result is sorted by unusual_score descending."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.detect_unusual_activity("AAPL")
        if len(result) > 1 and "unusual_score" in result.columns:
            for i in range(len(result) - 1):
                assert result["unusual_score"].iloc[i] >= result["unusual_score"].iloc[i + 1]

    def test_threshold_filtering(self, sample_unusual_activity):
        """Test that threshold filters correctly."""
        analyzer = OptionsFlowAnalyzer(unusual_threshold=8.0)
        with patch.object(analyzer, '_get_options_activity', return_value=sample_unusual_activity):
            result = analyzer.detect_unusual_activity("AAPL")
            # All results should have score >= threshold
            if not result.empty and "unusual_score" in result.columns:
                assert all(result["unusual_score"] >= 8.0)

    def test_min_premium_filter(self):
        """Test minimum premium filter."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.detect_unusual_activity("AAPL", min_premium=1_000_000)
        assert isinstance(result, pd.DataFrame)

    def test_lookback_days_parameter(self):
        """Test lookback_days parameter."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.detect_unusual_activity("AAPL", lookback_days=5)
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# Put/Call Ratio Tests
# =============================================================================


class TestPutCallRatio:
    """Tests for put_call_ratio calculation."""

    def test_returns_dict(self):
        """Test that method returns a dictionary."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.calculate_put_call_ratio("AAPL")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected keys."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.calculate_put_call_ratio("AAPL")
        expected_keys = [
            "symbol",
            "volume_ratio",
            "premium_ratio",
            "oi_ratio",
            "ratio_percentile",
            "sentiment_signal",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_volume_ratio_positive(self):
        """Test that volume_ratio is positive."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.calculate_put_call_ratio("AAPL")
        assert result["volume_ratio"] >= 0

    def test_ratio_percentile_bounded(self):
        """Test that ratio_percentile is in [0, 100] range."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.calculate_put_call_ratio("AAPL")
        assert 0 <= result["ratio_percentile"] <= 100

    def test_sentiment_signal_valid_values(self):
        """Test that sentiment_signal is valid."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.calculate_put_call_ratio("AAPL")
        valid_signals = ["bullish", "bearish", "neutral", "extreme_bullish", "extreme_bearish"]
        assert result["sentiment_signal"] in valid_signals

    def test_high_put_call_ratio_bearish(self, sample_options_data):
        """Test that high put/call ratio signals bearish."""
        analyzer = OptionsFlowAnalyzer()
        sample_options_data["put_volume"] = sample_options_data["call_volume"] * 2
        with patch.object(analyzer, '_get_options_data', return_value=sample_options_data):
            result = analyzer.calculate_put_call_ratio("AAPL")
            assert result["volume_ratio"] > 1.0

    def test_low_put_call_ratio_bullish(self, sample_options_data):
        """Test that low put/call ratio signals bullish."""
        analyzer = OptionsFlowAnalyzer()
        sample_options_data["put_volume"] = sample_options_data["call_volume"] // 2
        with patch.object(analyzer, '_get_options_data', return_value=sample_options_data):
            result = analyzer.calculate_put_call_ratio("AAPL")
            assert result["volume_ratio"] < 1.0


# =============================================================================
# Large Trades Filtering Tests
# =============================================================================


class TestLargeTradesFiltering:
    """Tests for large_trades filtering."""

    def test_returns_dataframe(self):
        """Test that method returns a DataFrame."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.get_large_trades("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        """Test that DataFrame has expected columns."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.get_large_trades("AAPL")
        expected_cols = ["symbol", "option_type", "strike", "size", "premium"]
        if not result.empty:
            for col in expected_cols:
                assert col in result.columns, f"Missing column: {col}"

    def test_min_size_filter(self, sample_large_trades):
        """Test minimum size filter."""
        analyzer = OptionsFlowAnalyzer()
        with patch.object(analyzer, '_get_large_trades_data', return_value=sample_large_trades):
            result = analyzer.get_large_trades("AAPL", min_size=10000)
            if not result.empty:
                assert all(result["size"] >= 10000)

    def test_min_premium_filter(self, sample_large_trades):
        """Test minimum premium filter."""
        analyzer = OptionsFlowAnalyzer()
        with patch.object(analyzer, '_get_large_trades_data', return_value=sample_large_trades):
            result = analyzer.get_large_trades("AAPL", min_premium=5_000_000)
            if not result.empty:
                assert all(result["premium"] >= 5_000_000)

    def test_trade_type_filter(self, sample_large_trades):
        """Test trade type filter (sweep vs block)."""
        analyzer = OptionsFlowAnalyzer()
        with patch.object(analyzer, '_get_large_trades_data', return_value=sample_large_trades):
            result = analyzer.get_large_trades("AAPL", trade_type="sweep")
            if not result.empty and "trade_type" in result.columns:
                assert all(result["trade_type"] == "sweep")

    def test_sorted_by_premium(self):
        """Test that result is sorted by premium descending."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.get_large_trades("AAPL")
        if len(result) > 1 and "premium" in result.columns:
            for i in range(len(result) - 1):
                assert result["premium"].iloc[i] >= result["premium"].iloc[i + 1]


# =============================================================================
# Options Sentiment Aggregation Tests
# =============================================================================


class TestOptionsSentimentAggregation:
    """Tests for options_sentiment aggregation."""

    def test_returns_dict(self):
        """Test that method returns a dictionary."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.aggregate_options_sentiment("AAPL")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected keys."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.aggregate_options_sentiment("AAPL")
        expected_keys = [
            "symbol",
            "overall_sentiment",
            "sentiment_score",
            "call_flow_strength",
            "put_flow_strength",
            "smart_money_signal",
            "retail_signal",
            "confidence",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_overall_sentiment_valid_values(self):
        """Test that overall_sentiment is valid."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.aggregate_options_sentiment("AAPL")
        valid_sentiments = ["bullish", "bearish", "neutral", "mixed"]
        assert result["overall_sentiment"] in valid_sentiments

    def test_sentiment_score_bounded(self):
        """Test that sentiment_score is in [-1, 1] range."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.aggregate_options_sentiment("AAPL")
        assert -1 <= result["sentiment_score"] <= 1

    def test_flow_strength_bounded(self):
        """Test that flow strength metrics are in [0, 1] range."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.aggregate_options_sentiment("AAPL")
        assert 0 <= result["call_flow_strength"] <= 1
        assert 0 <= result["put_flow_strength"] <= 1

    def test_confidence_bounded(self):
        """Test that confidence is in [0, 1] range."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.aggregate_options_sentiment("AAPL")
        assert 0 <= result["confidence"] <= 1

    def test_batch_sentiment(self):
        """Test batch sentiment analysis for multiple symbols."""
        analyzer = OptionsFlowAnalyzer()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = analyzer.aggregate_options_sentiment_batch(symbols)
        assert isinstance(results, dict)
        for symbol in symbols:
            assert symbol in results


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestOptionsFlowEdgeCases:
    """Edge case tests for OptionsFlowAnalyzer."""

    def test_empty_options_data(self, empty_options_data):
        """Test with empty options data."""
        analyzer = OptionsFlowAnalyzer()
        with patch.object(analyzer, '_get_options_data', return_value=empty_options_data):
            result = analyzer.calculate_put_call_ratio("AAPL")
            assert result["volume_ratio"] == 0.0 or np.isnan(result["volume_ratio"])

    def test_zero_call_volume(self, sample_options_data):
        """Test with zero call volume (division by zero)."""
        analyzer = OptionsFlowAnalyzer()
        sample_options_data["call_volume"] = 0
        with patch.object(analyzer, '_get_options_data', return_value=sample_options_data):
            result = analyzer.calculate_put_call_ratio("AAPL")
            # Should handle gracefully (inf or special value)
            assert isinstance(result["volume_ratio"], (int, float))

    def test_zero_put_volume(self, sample_options_data):
        """Test with zero put volume."""
        analyzer = OptionsFlowAnalyzer()
        sample_options_data["put_volume"] = 0
        with patch.object(analyzer, '_get_options_data', return_value=sample_options_data):
            result = analyzer.calculate_put_call_ratio("AAPL")
            assert result["volume_ratio"] == 0.0

    def test_very_high_unusual_threshold(self):
        """Test with very high unusual threshold."""
        analyzer = OptionsFlowAnalyzer(unusual_threshold=100.0)
        result = analyzer.detect_unusual_activity("AAPL")
        # May return empty DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_expiry_filter(self):
        """Test filtering by expiry date."""
        analyzer = OptionsFlowAnalyzer()
        future_date = datetime.now() + timedelta(days=60)
        result = analyzer.detect_unusual_activity("AAPL", max_expiry_days=30)
        assert isinstance(result, pd.DataFrame)

    def test_option_type_filter_calls(self):
        """Test filtering for calls only."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.detect_unusual_activity("AAPL", option_type="call")
        if not result.empty and "option_type" in result.columns:
            assert all(result["option_type"] == "call")

    def test_option_type_filter_puts(self):
        """Test filtering for puts only."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.detect_unusual_activity("AAPL", option_type="put")
        if not result.empty and "option_type" in result.columns:
            assert all(result["option_type"] == "put")

    def test_nan_handling_in_premium(self, sample_options_data):
        """Test handling of NaN values in premium data."""
        analyzer = OptionsFlowAnalyzer()
        sample_options_data.loc[0, "call_premium"] = np.nan
        sample_options_data.loc[1, "put_premium"] = np.nan
        with patch.object(analyzer, '_get_options_data', return_value=sample_options_data):
            result = analyzer.aggregate_options_sentiment("AAPL")
            # Should not crash
            assert isinstance(result, dict)


# =============================================================================
# Health Check Tests
# =============================================================================


class TestOptionsFlowHealthCheck:
    """Tests for health_check method."""

    def test_returns_true(self):
        """Test that health_check returns True."""
        analyzer = OptionsFlowAnalyzer()
        assert analyzer.health_check() is True

    def test_returns_dict_with_details(self):
        """Test that health_check can return detailed status."""
        analyzer = OptionsFlowAnalyzer()
        result = analyzer.health_check(detailed=True)
        if isinstance(result, dict):
            assert "status" in result
        else:
            assert result is True
