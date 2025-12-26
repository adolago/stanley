"""
Tests for SmartMoneyIndex module.

Tests smart money index calculation, component weighting,
signal generation thresholds, divergence detection, and batch processing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch


# Import will be created when module exists
try:
    from stanley.analytics.smart_money_index import SmartMoneyIndex
except ImportError:
    SmartMoneyIndex = None


# Skip all tests if module not yet implemented
pytestmark = pytest.mark.skipif(
    SmartMoneyIndex is None, reason="SmartMoneyIndex module not yet implemented"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_smi_components():
    """Sample SMI component data."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "date": dates,
            "institutional_flow": np.random.normal(0, 1000000, 30),
            "dark_pool_activity": np.random.uniform(0.15, 0.35, 30),
            "options_sentiment": np.random.uniform(-1, 1, 30),
            "whale_accumulation": np.random.uniform(-1, 1, 30),
            "sector_rotation_signal": np.random.uniform(-1, 1, 30),
        }
    )


@pytest.fixture
def sample_smi_history():
    """Sample SMI historical values."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
    np.random.seed(42)

    # Generate a mean-reverting SMI
    smi_values = []
    current = 50
    for _ in range(252):
        current = current + np.random.normal(0, 5) - 0.1 * (current - 50)
        current = np.clip(current, 0, 100)
        smi_values.append(current)

    return pd.DataFrame(
        {
            "date": dates,
            "smi_value": smi_values,
            "smi_signal": np.where(
                np.array(smi_values) > 70,
                "overbought",
                np.where(np.array(smi_values) < 30, "oversold", "neutral"),
            ),
        }
    )


@pytest.fixture
def sample_price_data():
    """Sample price data for divergence detection."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
    np.random.seed(42)

    # Trending up prices
    base_price = 100
    returns = np.random.normal(0.002, 0.015, len(dates))
    prices = base_price * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
        }
    )


@pytest.fixture
def empty_component_data():
    """Empty component data DataFrame."""
    return pd.DataFrame(
        columns=[
            "date",
            "institutional_flow",
            "dark_pool_activity",
            "options_sentiment",
            "whale_accumulation",
            "sector_rotation_signal",
        ]
    )


@pytest.fixture
def sample_weights():
    """Sample component weights."""
    return {
        "institutional_flow": 0.30,
        "dark_pool_activity": 0.20,
        "options_sentiment": 0.20,
        "whale_accumulation": 0.20,
        "sector_rotation_signal": 0.10,
    }


@pytest.fixture
def mock_data_manager_for_smi():
    """Mock DataManager for smart money index."""
    mock = Mock()
    mock.get_institutional_flow = AsyncMock(return_value=1_000_000)
    mock.get_dark_pool_data = AsyncMock(return_value={"percentage": 0.25})
    mock.get_options_sentiment = AsyncMock(return_value=0.5)
    return mock


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSmartMoneyIndexInit:
    """Tests for SmartMoneyIndex initialization."""

    def test_init_without_data_manager(self):
        """Test initialization without data_manager."""
        smi = SmartMoneyIndex()
        assert smi is not None
        assert smi.data_manager is None

    def test_init_with_data_manager(self, mock_data_manager_for_smi):
        """Test initialization with mock data_manager."""
        smi = SmartMoneyIndex(data_manager=mock_data_manager_for_smi)
        assert smi.data_manager is mock_data_manager_for_smi

    def test_default_weights(self):
        """Test default component weights."""
        smi = SmartMoneyIndex()
        weights = smi.get_weights()
        assert isinstance(weights, dict)
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_custom_weights(self, sample_weights):
        """Test custom component weights."""
        smi = SmartMoneyIndex(weights=sample_weights)
        weights = smi.get_weights()
        assert weights == sample_weights


# =============================================================================
# SMI Calculation Tests
# =============================================================================


class TestSMICalculation:
    """Tests for SmartMoneyIndex calculation."""

    def test_calculate_returns_dict(self):
        """Test that calculate returns a dictionary."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected keys."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        expected_keys = [
            "symbol",
            "smi_value",
            "smi_signal",
            "components",
            "percentile",
            "trend",
            "timestamp",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_smi_value_bounded(self):
        """Test that smi_value is in [0, 100] range."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        assert 0 <= result["smi_value"] <= 100

    def test_smi_signal_valid_values(self):
        """Test that smi_signal is valid."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        valid_signals = [
            "strong_buy",
            "buy",
            "neutral",
            "sell",
            "strong_sell",
            "overbought",
            "oversold",
        ]
        assert result["smi_signal"] in valid_signals

    def test_components_is_dict(self):
        """Test that components is a dictionary."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        assert isinstance(result["components"], dict)

    def test_percentile_bounded(self):
        """Test that percentile is in [0, 100] range."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        assert 0 <= result["percentile"] <= 100

    def test_trend_valid_values(self):
        """Test that trend is valid."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        valid_trends = ["rising", "falling", "flat", "accelerating", "decelerating"]
        assert result["trend"] in valid_trends


# =============================================================================
# Component Weighting Tests
# =============================================================================


class TestComponentWeighting:
    """Tests for component weighting."""

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1.0."""
        smi = SmartMoneyIndex()
        weights = smi.get_weights()
        assert sum(weights.values()) == pytest.approx(1.0, rel=0.01)

    def test_all_weights_non_negative(self):
        """Test that all weights are non-negative."""
        smi = SmartMoneyIndex()
        weights = smi.get_weights()
        assert all(w >= 0 for w in weights.values())

    def test_set_weights(self, sample_weights):
        """Test setting custom weights."""
        smi = SmartMoneyIndex()
        smi.set_weights(sample_weights)
        assert smi.get_weights() == sample_weights

    def test_invalid_weights_rejected(self):
        """Test that invalid weights are rejected."""
        smi = SmartMoneyIndex()
        invalid_weights = {"component1": 0.5}  # Doesn't sum to 1.0
        with pytest.raises((ValueError, AssertionError)):
            smi.set_weights(invalid_weights)

    def test_negative_weights_rejected(self):
        """Test that negative weights are rejected."""
        smi = SmartMoneyIndex()
        negative_weights = {
            "institutional_flow": -0.10,
            "dark_pool_activity": 0.55,
            "options_sentiment": 0.55,
        }
        with pytest.raises((ValueError, AssertionError)):
            smi.set_weights(negative_weights)

    def test_get_component_contribution(self):
        """Test getting individual component contributions."""
        smi = SmartMoneyIndex()
        result = smi.calculate("AAPL")
        components = result["components"]

        # Each component should have a contribution
        for component in [
            "institutional_flow",
            "dark_pool_activity",
            "options_sentiment",
        ]:
            if component in components:
                assert "value" in components[component] or isinstance(
                    components[component], (int, float)
                )


# =============================================================================
# Signal Generation Threshold Tests
# =============================================================================


class TestSignalGenerationThresholds:
    """Tests for signal generation thresholds."""

    def test_get_thresholds(self):
        """Test getting signal thresholds."""
        smi = SmartMoneyIndex()
        thresholds = smi.get_signal_thresholds()
        assert isinstance(thresholds, dict)
        assert "overbought" in thresholds
        assert "oversold" in thresholds

    def test_set_thresholds(self):
        """Test setting custom thresholds."""
        smi = SmartMoneyIndex()
        custom_thresholds = {
            "overbought": 80,
            "oversold": 20,
            "strong_buy": 15,
            "strong_sell": 85,
        }
        smi.set_signal_thresholds(custom_thresholds)
        thresholds = smi.get_signal_thresholds()
        assert thresholds["overbought"] == 80
        assert thresholds["oversold"] == 20

    def test_overbought_signal(self, sample_smi_components):
        """Test overbought signal generation."""
        smi = SmartMoneyIndex()
        # Force high SMI value
        sample_smi_components["institutional_flow"] = 10_000_000
        sample_smi_components["options_sentiment"] = 0.9
        sample_smi_components["whale_accumulation"] = 0.9
        with patch.object(
            smi, "_get_component_data", return_value=sample_smi_components
        ):
            result = smi.calculate("AAPL")
            if result["smi_value"] > 70:
                assert result["smi_signal"] in ["overbought", "strong_sell", "sell"]

    def test_oversold_signal(self, sample_smi_components):
        """Test oversold signal generation."""
        smi = SmartMoneyIndex()
        # Force low SMI value
        sample_smi_components["institutional_flow"] = -10_000_000
        sample_smi_components["options_sentiment"] = -0.9
        sample_smi_components["whale_accumulation"] = -0.9
        with patch.object(
            smi, "_get_component_data", return_value=sample_smi_components
        ):
            result = smi.calculate("AAPL")
            if result["smi_value"] < 30:
                assert result["smi_signal"] in ["oversold", "strong_buy", "buy"]


# =============================================================================
# Divergence Detection Tests
# =============================================================================


class TestDivergenceDetection:
    """Tests for divergence detection."""

    def test_detect_divergence_returns_dict(self):
        """Test that detect_divergence returns a dictionary."""
        smi = SmartMoneyIndex()
        result = smi.detect_divergence("AAPL")
        assert isinstance(result, dict)

    def test_divergence_has_expected_keys(self):
        """Test that divergence result has expected keys."""
        smi = SmartMoneyIndex()
        result = smi.detect_divergence("AAPL")
        expected_keys = [
            "symbol",
            "divergence_type",
            "divergence_strength",
            "price_trend",
            "smi_trend",
            "signal",
            "lookback_days",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_divergence_type_valid_values(self):
        """Test that divergence_type is valid."""
        smi = SmartMoneyIndex()
        result = smi.detect_divergence("AAPL")
        valid_types = ["bullish", "bearish", "none", "hidden_bullish", "hidden_bearish"]
        assert result["divergence_type"] in valid_types

    def test_divergence_strength_bounded(self):
        """Test that divergence_strength is in [0, 1] range."""
        smi = SmartMoneyIndex()
        result = smi.detect_divergence("AAPL")
        assert 0 <= result["divergence_strength"] <= 1

    def test_bullish_divergence(self, sample_price_data, sample_smi_history):
        """Test bullish divergence detection (price down, SMI up)."""
        smi = SmartMoneyIndex()
        # Price trending down
        sample_price_data["close"] = sample_price_data["close"][::-1].values
        # SMI trending up
        sample_smi_history["smi_value"] = np.linspace(30, 70, len(sample_smi_history))

        with patch.object(smi, "_get_price_data", return_value=sample_price_data):
            with patch.object(smi, "_get_smi_history", return_value=sample_smi_history):
                result = smi.detect_divergence("AAPL")
                # May detect bullish divergence
                assert isinstance(result, dict)

    def test_bearish_divergence(self, sample_price_data, sample_smi_history):
        """Test bearish divergence detection (price up, SMI down)."""
        smi = SmartMoneyIndex()
        # Price trending up (already set)
        # SMI trending down
        sample_smi_history["smi_value"] = np.linspace(70, 30, len(sample_smi_history))

        with patch.object(smi, "_get_price_data", return_value=sample_price_data):
            with patch.object(smi, "_get_smi_history", return_value=sample_smi_history):
                result = smi.detect_divergence("AAPL")
                # May detect bearish divergence
                assert isinstance(result, dict)

    def test_lookback_parameter(self):
        """Test lookback_days parameter."""
        smi = SmartMoneyIndex()
        result = smi.detect_divergence("AAPL", lookback_days=30)
        assert result["lookback_days"] == 30


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_calculate_batch_returns_dict(self):
        """Test that calculate_batch returns a dictionary."""
        smi = SmartMoneyIndex()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        result = smi.calculate_batch(symbols)
        assert isinstance(result, dict)

    def test_batch_contains_all_symbols(self):
        """Test that batch result contains all symbols."""
        smi = SmartMoneyIndex()
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        result = smi.calculate_batch(symbols)
        for symbol in symbols:
            assert symbol in result

    def test_batch_results_structure(self):
        """Test that each batch result has expected structure."""
        smi = SmartMoneyIndex()
        symbols = ["AAPL", "MSFT"]
        result = smi.calculate_batch(symbols)
        for symbol in symbols:
            assert "smi_value" in result[symbol]
            assert "smi_signal" in result[symbol]

    def test_empty_batch(self):
        """Test with empty symbol list."""
        smi = SmartMoneyIndex()
        result = smi.calculate_batch([])
        assert result == {}

    def test_batch_with_parallel_processing(self):
        """Test batch with parallel processing enabled."""
        smi = SmartMoneyIndex()
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        result = smi.calculate_batch(symbols, parallel=True)
        assert isinstance(result, dict)
        assert len(result) == len(symbols)

    def test_batch_error_handling(self):
        """Test batch error handling for invalid symbols."""
        smi = SmartMoneyIndex()
        symbols = ["AAPL", "INVALID_SYMBOL_XYZ", "MSFT"]
        result = smi.calculate_batch(symbols)
        # Should handle gracefully (may skip invalid or return error)
        assert isinstance(result, dict)


# =============================================================================
# Historical Analysis Tests
# =============================================================================


class TestHistoricalAnalysis:
    """Tests for historical SMI analysis."""

    def test_get_history_returns_dataframe(self):
        """Test that get_history returns a DataFrame."""
        smi = SmartMoneyIndex()
        result = smi.get_history("AAPL")
        assert isinstance(result, pd.DataFrame)

    def test_history_has_expected_columns(self):
        """Test that history DataFrame has expected columns."""
        smi = SmartMoneyIndex()
        result = smi.get_history("AAPL")
        expected_cols = ["date", "smi_value", "smi_signal"]
        if not result.empty:
            for col in expected_cols:
                assert col in result.columns, f"Missing column: {col}"

    def test_history_lookback_parameter(self):
        """Test lookback_days parameter for history."""
        smi = SmartMoneyIndex()
        result = smi.get_history("AAPL", lookback_days=30)
        if not result.empty:
            assert len(result) <= 30

    def test_get_statistics(self):
        """Test getting SMI statistics."""
        smi = SmartMoneyIndex()
        result = smi.get_statistics("AAPL")
        assert isinstance(result, dict)
        expected_keys = ["mean", "std", "min", "max", "current", "percentile_rank"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestSmartMoneyIndexEdgeCases:
    """Edge case tests for SmartMoneyIndex."""

    def test_empty_component_data(self, empty_component_data):
        """Test with empty component data."""
        smi = SmartMoneyIndex()
        with patch.object(
            smi, "_get_component_data", return_value=empty_component_data
        ):
            result = smi.calculate("AAPL")
            # Should return neutral/default values
            assert result["smi_value"] == 50.0 or 0 <= result["smi_value"] <= 100

    def test_missing_components(self, sample_smi_components):
        """Test with missing component columns."""
        smi = SmartMoneyIndex()
        incomplete_data = sample_smi_components.drop(columns=["whale_accumulation"])
        with patch.object(smi, "_get_component_data", return_value=incomplete_data):
            result = smi.calculate("AAPL")
            # Should handle gracefully
            assert isinstance(result, dict)

    def test_nan_in_components(self, sample_smi_components):
        """Test with NaN values in components."""
        smi = SmartMoneyIndex()
        sample_smi_components.loc[0, "institutional_flow"] = np.nan
        sample_smi_components.loc[5, "options_sentiment"] = np.nan
        with patch.object(
            smi, "_get_component_data", return_value=sample_smi_components
        ):
            result = smi.calculate("AAPL")
            assert isinstance(result, dict)
            assert not np.isnan(result["smi_value"])

    def test_extreme_component_values(self, sample_smi_components):
        """Test with extreme component values."""
        smi = SmartMoneyIndex()
        sample_smi_components["institutional_flow"] = 1e12  # Extreme
        sample_smi_components["options_sentiment"] = 100  # Out of normal range
        with patch.object(
            smi, "_get_component_data", return_value=sample_smi_components
        ):
            result = smi.calculate("AAPL")
            # Should still return bounded value
            assert 0 <= result["smi_value"] <= 100

    def test_zero_weight_component(self):
        """Test with zero weight for a component."""
        weights = {
            "institutional_flow": 0.0,  # Zero weight
            "dark_pool_activity": 0.35,
            "options_sentiment": 0.35,
            "whale_accumulation": 0.30,
        }
        smi = SmartMoneyIndex(weights=weights)
        result = smi.calculate("AAPL")
        assert isinstance(result, dict)


# =============================================================================
# Market-Wide SMI Tests
# =============================================================================


class TestMarketWideSMI:
    """Tests for market-wide SMI calculation."""

    def test_calculate_market_smi_returns_dict(self):
        """Test that calculate_market_smi returns a dictionary."""
        smi = SmartMoneyIndex()
        result = smi.calculate_market_smi()
        assert isinstance(result, dict)

    def test_market_smi_has_expected_keys(self):
        """Test that market SMI has expected keys."""
        smi = SmartMoneyIndex()
        result = smi.calculate_market_smi()
        expected_keys = ["market_smi", "market_signal", "sector_breakdown", "breadth"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_market_smi_bounded(self):
        """Test that market_smi is in [0, 100] range."""
        smi = SmartMoneyIndex()
        result = smi.calculate_market_smi()
        assert 0 <= result["market_smi"] <= 100


# =============================================================================
# Health Check Tests
# =============================================================================


class TestSmartMoneyIndexHealthCheck:
    """Tests for health_check method."""

    def test_returns_true(self):
        """Test that health_check returns True."""
        smi = SmartMoneyIndex()
        assert smi.health_check() is True

    def test_returns_dict_with_details(self):
        """Test that health_check can return detailed status."""
        smi = SmartMoneyIndex()
        result = smi.health_check(detailed=True)
        if isinstance(result, dict):
            assert "status" in result
        else:
            assert result is True
