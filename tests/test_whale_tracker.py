"""
Tests for WhaleTracker module.

Tests whale movement tracking, accumulation/distribution detection,
and institutional whale consensus calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch


# Import will be created when module exists
# For now, we test the expected interface
try:
    from stanley.analytics.whale_tracker import WhaleTracker
except ImportError:
    WhaleTracker = None


# Skip all tests if module not yet implemented
pytestmark = pytest.mark.skipif(
    WhaleTracker is None,
    reason="WhaleTracker module not yet implemented"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_whale_holdings():
    """Sample whale holdings data."""
    return pd.DataFrame({
        "manager_name": [
            "Berkshire Hathaway",
            "Renaissance Technologies",
            "Bridgewater Associates",
            "Two Sigma",
            "Citadel Advisors",
        ],
        "manager_cik": [
            "0001067983",
            "0001037389",
            "0001350694",
            "0001450144",
            "0001423053",
        ],
        "aum": [
            700_000_000_000,  # $700B
            130_000_000_000,  # $130B
            150_000_000_000,  # $150B
            60_000_000_000,   # $60B
            50_000_000_000,   # $50B
        ],
        "shares_held": [100_000_000, 50_000_000, 30_000_000, 25_000_000, 20_000_000],
        "value_held": [
            15_000_000_000,
            7_500_000_000,
            4_500_000_000,
            3_750_000_000,
            3_000_000_000,
        ],
        "ownership_percentage": [0.08, 0.04, 0.024, 0.02, 0.016],
        "quarter_change": [5_000_000, -2_000_000, 3_000_000, 0, 1_000_000],
        "is_new_position": [False, False, False, False, False],
    })


@pytest.fixture
def sample_whale_movements():
    """Sample whale movement history."""
    dates = pd.date_range(end=datetime.now(), periods=4, freq="QE")
    return pd.DataFrame({
        "date": dates,
        "total_whale_shares": [200_000_000, 210_000_000, 205_000_000, 225_000_000],
        "net_whale_change": [0, 10_000_000, -5_000_000, 20_000_000],
        "whales_buying": [3, 4, 2, 5],
        "whales_selling": [2, 1, 3, 1],
        "whale_count": [10, 11, 10, 12],
    })


@pytest.fixture
def empty_whale_data():
    """Empty whale holdings DataFrame."""
    return pd.DataFrame(columns=[
        "manager_name", "manager_cik", "aum", "shares_held",
        "value_held", "ownership_percentage", "quarter_change", "is_new_position"
    ])


@pytest.fixture
def single_whale_data():
    """Single whale holder."""
    return pd.DataFrame({
        "manager_name": ["Berkshire Hathaway"],
        "manager_cik": ["0001067983"],
        "aum": [700_000_000_000],
        "shares_held": [100_000_000],
        "value_held": [15_000_000_000],
        "ownership_percentage": [0.08],
        "quarter_change": [5_000_000],
        "is_new_position": [False],
    })


@pytest.fixture
def mock_data_manager_for_whales():
    """Mock DataManager for whale tracking."""
    mock = Mock()
    mock.get_whale_holdings = AsyncMock(return_value=pd.DataFrame({
        "manager_name": ["Berkshire Hathaway", "Renaissance Technologies"],
        "shares_held": [100_000_000, 50_000_000],
        "value_held": [15_000_000_000, 7_500_000_000],
        "ownership_percentage": [0.08, 0.04],
        "quarter_change": [5_000_000, -2_000_000],
    }))
    mock.get_13f_filings = AsyncMock(return_value=pd.DataFrame())
    return mock


# =============================================================================
# Initialization Tests
# =============================================================================


class TestWhaleTrackerInit:
    """Tests for WhaleTracker initialization."""

    def test_init_without_data_manager(self):
        """Test initialization without data_manager."""
        tracker = WhaleTracker()
        assert tracker is not None
        assert tracker.data_manager is None

    def test_init_with_data_manager(self, mock_data_manager_for_whales):
        """Test initialization with mock data_manager."""
        tracker = WhaleTracker(data_manager=mock_data_manager_for_whales)
        assert tracker.data_manager is mock_data_manager_for_whales

    def test_default_min_aum_threshold(self):
        """Test default minimum AUM threshold for whale classification."""
        tracker = WhaleTracker()
        assert tracker.min_aum_threshold == 1_000_000_000  # $1B default

    def test_custom_min_aum_threshold(self):
        """Test custom minimum AUM threshold."""
        tracker = WhaleTracker(min_aum_threshold=10_000_000_000)
        assert tracker.min_aum_threshold == 10_000_000_000


# =============================================================================
# Track Whale Movements Tests
# =============================================================================


class TestTrackWhaleMovements:
    """Tests for track_whale_movements method."""

    def test_returns_dict(self):
        """Test that method returns a dictionary."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("AAPL")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected keys."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("AAPL")
        expected_keys = [
            "symbol",
            "total_whale_ownership",
            "whale_count",
            "net_whale_change",
            "whales_buying",
            "whales_selling",
            "whale_sentiment",
            "top_whales",
            "recent_movements",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_symbol_echoed(self):
        """Test that input symbol appears in result."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("MSFT")
        assert result["symbol"] == "MSFT"

    def test_whale_ownership_bounded(self):
        """Test that total_whale_ownership is in [0, 1] range."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("AAPL")
        assert 0 <= result["total_whale_ownership"] <= 1

    def test_whale_count_non_negative(self):
        """Test that whale_count is non-negative."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("AAPL")
        assert result["whale_count"] >= 0

    def test_whale_sentiment_valid_values(self):
        """Test that whale_sentiment is one of expected values."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("AAPL")
        valid_sentiments = ["accumulating", "distributing", "neutral", "mixed"]
        assert result["whale_sentiment"] in valid_sentiments

    def test_top_whales_is_list_or_dataframe(self):
        """Test that top_whales is a list or DataFrame."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("AAPL")
        assert isinstance(result["top_whales"], (list, pd.DataFrame))


# =============================================================================
# Get Top Whales Tests
# =============================================================================


class TestGetTopWhales:
    """Tests for get_top_whales method."""

    def test_returns_dataframe(self):
        """Test that method returns a DataFrame."""
        tracker = WhaleTracker()
        result = tracker.get_top_whales("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_value_held(self):
        """Test that result is sorted by value_held descending."""
        tracker = WhaleTracker()
        result = tracker.get_top_whales("AAPL")
        if len(result) > 1 and "value_held" in result.columns:
            for i in range(len(result) - 1):
                assert result["value_held"].iloc[i] >= result["value_held"].iloc[i + 1]

    def test_limit_parameter(self):
        """Test that limit parameter works correctly."""
        tracker = WhaleTracker()
        result = tracker.get_top_whales("AAPL", limit=3)
        assert len(result) <= 3

    def test_has_expected_columns(self):
        """Test that DataFrame has expected columns."""
        tracker = WhaleTracker()
        result = tracker.get_top_whales("AAPL")
        expected_cols = ["manager_name", "value_held", "ownership_percentage"]
        for col in expected_cols:
            if not result.empty:
                assert col in result.columns, f"Missing column: {col}"

    def test_empty_result_for_obscure_symbol(self):
        """Test handling of symbol with no whale holdings."""
        tracker = WhaleTracker()
        result = tracker.get_top_whales("OBSCURE_STOCK_XYZ")
        assert isinstance(result, pd.DataFrame)
        # May be empty or have mock data


# =============================================================================
# Accumulation/Distribution Detection Tests
# =============================================================================


class TestAccumulationDistribution:
    """Tests for accumulation/distribution detection."""

    def test_detect_accumulation_returns_dict(self):
        """Test that detect_accumulation returns a dictionary."""
        tracker = WhaleTracker()
        result = tracker.detect_accumulation("AAPL")
        assert isinstance(result, dict)

    def test_accumulation_has_expected_keys(self):
        """Test that accumulation result has expected keys."""
        tracker = WhaleTracker()
        result = tracker.detect_accumulation("AAPL")
        expected_keys = [
            "symbol",
            "pattern",
            "strength",
            "duration_quarters",
            "net_shares_added",
            "whale_conviction",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_pattern_valid_values(self):
        """Test that pattern is one of expected values."""
        tracker = WhaleTracker()
        result = tracker.detect_accumulation("AAPL")
        valid_patterns = ["accumulation", "distribution", "neutral", "mixed"]
        assert result["pattern"] in valid_patterns

    def test_strength_bounded(self):
        """Test that strength is in [-1, 1] range."""
        tracker = WhaleTracker()
        result = tracker.detect_accumulation("AAPL")
        assert -1 <= result["strength"] <= 1

    def test_duration_quarters_non_negative(self):
        """Test that duration_quarters is non-negative."""
        tracker = WhaleTracker()
        result = tracker.detect_accumulation("AAPL")
        assert result["duration_quarters"] >= 0

    def test_whale_conviction_bounded(self):
        """Test that whale_conviction is in [0, 1] range."""
        tracker = WhaleTracker()
        result = tracker.detect_accumulation("AAPL")
        assert 0 <= result["whale_conviction"] <= 1


# =============================================================================
# Whale Consensus Tests
# =============================================================================


class TestWhaleConsensus:
    """Tests for whale_consensus calculation."""

    def test_calculate_whale_consensus_returns_dict(self):
        """Test that calculate_whale_consensus returns a dictionary."""
        tracker = WhaleTracker()
        result = tracker.calculate_whale_consensus("AAPL")
        assert isinstance(result, dict)

    def test_consensus_has_expected_keys(self):
        """Test that consensus result has expected keys."""
        tracker = WhaleTracker()
        result = tracker.calculate_whale_consensus("AAPL")
        expected_keys = [
            "symbol",
            "consensus_direction",
            "consensus_strength",
            "bullish_whales",
            "bearish_whales",
            "neutral_whales",
            "agreement_score",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_consensus_direction_valid_values(self):
        """Test that consensus_direction is valid."""
        tracker = WhaleTracker()
        result = tracker.calculate_whale_consensus("AAPL")
        valid_directions = ["bullish", "bearish", "neutral", "split"]
        assert result["consensus_direction"] in valid_directions

    def test_consensus_strength_bounded(self):
        """Test that consensus_strength is in [0, 1] range."""
        tracker = WhaleTracker()
        result = tracker.calculate_whale_consensus("AAPL")
        assert 0 <= result["consensus_strength"] <= 1

    def test_whale_counts_non_negative(self):
        """Test that whale counts are non-negative."""
        tracker = WhaleTracker()
        result = tracker.calculate_whale_consensus("AAPL")
        assert result["bullish_whales"] >= 0
        assert result["bearish_whales"] >= 0
        assert result["neutral_whales"] >= 0

    def test_agreement_score_bounded(self):
        """Test that agreement_score is in [0, 1] range."""
        tracker = WhaleTracker()
        result = tracker.calculate_whale_consensus("AAPL")
        assert 0 <= result["agreement_score"] <= 1


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestWhaleTrackerEdgeCases:
    """Edge case tests for WhaleTracker."""

    def test_empty_whale_data(self, empty_whale_data):
        """Test with empty whale data."""
        tracker = WhaleTracker()
        with patch.object(tracker, '_get_whale_holdings', return_value=empty_whale_data):
            result = tracker.track_whale_movements("AAPL")
            assert result["whale_count"] == 0
            assert result["total_whale_ownership"] == 0.0

    def test_single_whale_holder(self, single_whale_data):
        """Test with single whale holder."""
        tracker = WhaleTracker()
        with patch.object(tracker, '_get_whale_holdings', return_value=single_whale_data):
            result = tracker.track_whale_movements("AAPL")
            assert result["whale_count"] == 1

    def test_very_high_aum_filter(self):
        """Test with very high AUM filter that excludes all whales."""
        tracker = WhaleTracker(min_aum_threshold=100_000_000_000_000)  # $100T
        result = tracker.track_whale_movements("AAPL")
        # Should handle gracefully
        assert isinstance(result, dict)

    def test_lookback_quarters_parameter(self):
        """Test lookback_quarters parameter."""
        tracker = WhaleTracker()
        result = tracker.track_whale_movements("AAPL", lookback_quarters=8)
        assert isinstance(result, dict)

    def test_multiple_symbols_batch(self):
        """Test batch processing of multiple symbols."""
        tracker = WhaleTracker()
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        results = tracker.track_whale_movements_batch(symbols)
        assert isinstance(results, dict)
        for symbol in symbols:
            assert symbol in results

    def test_handle_missing_quarter_change(self):
        """Test handling of missing quarter_change data."""
        tracker = WhaleTracker()
        incomplete_data = pd.DataFrame({
            "manager_name": ["Whale A"],
            "shares_held": [1_000_000],
            "value_held": [100_000_000],
            # Missing quarter_change column
        })
        with patch.object(tracker, '_get_whale_holdings', return_value=incomplete_data):
            result = tracker.track_whale_movements("AAPL")
            # Should not crash
            assert isinstance(result, dict)

    def test_negative_quarter_change(self, sample_whale_holdings):
        """Test with negative quarter changes (selling)."""
        tracker = WhaleTracker()
        sample_whale_holdings["quarter_change"] = [-10_000_000, -5_000_000, -3_000_000, -2_000_000, -1_000_000]
        with patch.object(tracker, '_get_whale_holdings', return_value=sample_whale_holdings):
            result = tracker.detect_accumulation("AAPL")
            assert result["pattern"] == "distribution"

    def test_all_positive_quarter_change(self, sample_whale_holdings):
        """Test with all positive quarter changes (buying)."""
        tracker = WhaleTracker()
        sample_whale_holdings["quarter_change"] = [10_000_000, 5_000_000, 3_000_000, 2_000_000, 1_000_000]
        with patch.object(tracker, '_get_whale_holdings', return_value=sample_whale_holdings):
            result = tracker.detect_accumulation("AAPL")
            assert result["pattern"] == "accumulation"


# =============================================================================
# Health Check Tests
# =============================================================================


class TestWhaleTrackerHealthCheck:
    """Tests for health_check method."""

    def test_returns_true(self):
        """Test that health_check returns True."""
        tracker = WhaleTracker()
        assert tracker.health_check() is True

    def test_returns_dict_with_details(self):
        """Test that health_check can return detailed status."""
        tracker = WhaleTracker()
        result = tracker.health_check(detailed=True)
        if isinstance(result, dict):
            assert "status" in result
        else:
            assert result is True
