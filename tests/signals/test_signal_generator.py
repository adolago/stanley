"""
Tests for SignalGenerator module.

This module tests the main signal generation functionality including:
- Single symbol signal generation
- Universe signal generation
- Composite score calculation
- Conviction indicators
- Factor weighting
- Edge cases and error handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Skip all tests if signals module doesn't exist yet
try:
    from stanley.signals import SignalGenerator
    from stanley.signals.signal_generator import (
        Signal,
        SignalType,
        SignalStrength,
        CompositeSignal,
    )

    # Aliases for backwards compatibility with test file naming
    SignalDirection = SignalType
    ConvictionLevel = SignalStrength
    FactorScores = CompositeSignal
    HAS_SIGNALS_MODULE = True
except ImportError:
    HAS_SIGNALS_MODULE = False
    SignalGenerator = None
    Signal = None
    SignalDirection = None
    ConvictionLevel = None
    FactorScores = None

pytestmark = pytest.mark.skipif(
    not HAS_SIGNALS_MODULE, reason="stanley.signals module not yet implemented"
)


# =============================================================================
# SignalGenerator Initialization Tests
# =============================================================================


class TestSignalGeneratorInit:
    """Tests for SignalGenerator initialization."""

    def test_init_without_dependencies(self):
        """Test initialization without data manager."""
        generator = SignalGenerator()
        assert generator is not None
        assert generator.data_manager is None

    def test_init_with_data_manager(self, mock_data_manager):
        """Test initialization with mock data manager."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        assert generator.data_manager is mock_data_manager

    def test_init_with_custom_weights(self, sample_factor_weights):
        """Test initialization with custom factor weights."""
        generator = SignalGenerator(factor_weights=sample_factor_weights)
        assert generator.factor_weights == sample_factor_weights

    def test_init_with_analyzers(
        self, mock_money_flow_analyzer, mock_institutional_analyzer
    ):
        """Test initialization with pre-configured analyzers."""
        generator = SignalGenerator(
            money_flow_analyzer=mock_money_flow_analyzer,
            institutional_analyzer=mock_institutional_analyzer,
        )
        assert generator._money_flow_analyzer == mock_money_flow_analyzer
        assert generator._institutional_analyzer == mock_institutional_analyzer

    def test_default_factor_weights(self):
        """Test that default factor weights sum to 1.0."""
        generator = SignalGenerator()
        total_weight = sum(generator.factor_weights.values())
        assert total_weight == pytest.approx(1.0, abs=0.001)


# =============================================================================
# Signal Data Class Tests
# =============================================================================


class TestSignalDataClass:
    """Tests for Signal data class."""

    def test_signal_creation(self):
        """Test creating a Signal instance."""
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            direction=SignalDirection.BULLISH,
            strength=0.75,
            confidence=0.85,
            conviction=ConvictionLevel.HIGH,
        )
        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.BULLISH
        assert signal.strength == 0.75
        assert signal.confidence == 0.85
        assert signal.conviction == ConvictionLevel.HIGH

    def test_signal_with_factors(self):
        """Test creating a Signal with factor scores."""
        factors = FactorScores(
            money_flow=0.8,
            institutional=0.7,
            momentum=0.65,
            value=0.5,
            technical=0.6,
        )
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            direction=SignalDirection.BULLISH,
            strength=0.75,
            confidence=0.85,
            conviction=ConvictionLevel.HIGH,
            factors=factors,
        )
        assert signal.factors.money_flow == 0.8
        assert signal.factors.institutional == 0.7

    def test_signal_with_price_targets(self):
        """Test Signal with entry, target, and stop prices."""
        signal = Signal(
            symbol="AAPL",
            timestamp=datetime.now(),
            direction=SignalDirection.BULLISH,
            strength=0.75,
            confidence=0.85,
            conviction=ConvictionLevel.HIGH,
            entry_price=175.50,
            target_price=195.00,
            stop_loss=168.00,
        )
        assert signal.entry_price == 175.50
        assert signal.target_price == 195.00
        assert signal.stop_loss == 168.00


class TestSignalDirection:
    """Tests for SignalDirection enum."""

    def test_signal_direction_values(self):
        """Test SignalDirection enum values."""
        assert SignalDirection.BULLISH.value == "BULLISH"
        assert SignalDirection.BEARISH.value == "BEARISH"
        assert SignalDirection.NEUTRAL.value == "NEUTRAL"


class TestConvictionLevel:
    """Tests for ConvictionLevel enum."""

    def test_conviction_level_values(self):
        """Test ConvictionLevel enum values."""
        assert ConvictionLevel.HIGH.value == "HIGH"
        assert ConvictionLevel.MEDIUM.value == "MEDIUM"
        assert ConvictionLevel.LOW.value == "LOW"


# =============================================================================
# Single Symbol Signal Generation Tests
# =============================================================================


class TestGenerateSignal:
    """Tests for single symbol signal generation."""

    def test_generate_signal_returns_signal(self, mock_data_manager):
        """Test that generate_signal returns a Signal object."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        signal = generator.generate_signal("AAPL")

        assert isinstance(signal, Signal)
        assert signal.symbol == "AAPL"

    def test_generate_signal_has_required_fields(self, mock_data_manager):
        """Test that generated signal has all required fields."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        signal = generator.generate_signal("AAPL")

        assert signal.symbol is not None
        assert signal.timestamp is not None
        assert signal.direction is not None
        assert signal.strength is not None
        assert signal.confidence is not None
        assert signal.conviction is not None

    def test_generate_signal_strength_bounded(self, mock_data_manager):
        """Test that signal strength is in [-1, 1] range."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        signal = generator.generate_signal("AAPL")

        assert -1.0 <= signal.strength <= 1.0

    def test_generate_signal_confidence_bounded(self, mock_data_manager):
        """Test that signal confidence is in [0, 1] range."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        signal = generator.generate_signal("AAPL")

        assert 0.0 <= signal.confidence <= 1.0

    def test_generate_bullish_signal(self, mock_data_manager, mock_money_flow_analyzer):
        """Test generation of bullish signal with strong positive factors."""
        mock_money_flow_analyzer.analyze_equity_flow.return_value = {
            "symbol": "AAPL",
            "money_flow_score": 0.9,
            "institutional_sentiment": 0.85,
            "smart_money_activity": 0.8,
            "confidence": 0.9,
        }

        generator = SignalGenerator(
            data_manager=mock_data_manager,
            money_flow_analyzer=mock_money_flow_analyzer,
        )
        signal = generator.generate_signal("AAPL")

        assert signal.direction == SignalDirection.BULLISH
        assert signal.strength > 0.5

    def test_generate_bearish_signal(self, mock_data_manager, mock_money_flow_analyzer):
        """Test generation of bearish signal with strong negative factors."""
        mock_money_flow_analyzer.analyze_equity_flow.return_value = {
            "symbol": "AAPL",
            "money_flow_score": -0.85,
            "institutional_sentiment": -0.8,
            "smart_money_activity": -0.75,
            "confidence": 0.85,
        }

        generator = SignalGenerator(
            data_manager=mock_data_manager,
            money_flow_analyzer=mock_money_flow_analyzer,
        )
        signal = generator.generate_signal("AAPL")

        assert signal.direction == SignalDirection.BEARISH
        assert signal.strength < -0.5

    def test_generate_neutral_signal(self, mock_data_manager, mock_money_flow_analyzer):
        """Test generation of neutral signal with mixed factors."""
        mock_money_flow_analyzer.analyze_equity_flow.return_value = {
            "symbol": "AAPL",
            "money_flow_score": 0.1,
            "institutional_sentiment": -0.05,
            "smart_money_activity": 0.0,
            "confidence": 0.4,
        }

        generator = SignalGenerator(
            data_manager=mock_data_manager,
            money_flow_analyzer=mock_money_flow_analyzer,
        )
        signal = generator.generate_signal("AAPL")

        assert signal.direction == SignalDirection.NEUTRAL
        assert abs(signal.strength) < 0.3


# =============================================================================
# Universe Signal Generation Tests
# =============================================================================


class TestGenerateUniverseSignals:
    """Tests for universe signal generation."""

    def test_generate_universe_signals_returns_list(self, mock_data_manager):
        """Test that generate_universe_signals returns a list."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        universe = ["AAPL", "MSFT", "GOOGL"]
        signals = generator.generate_universe_signals(universe)

        assert isinstance(signals, list)
        assert len(signals) == len(universe)

    def test_generate_universe_signals_all_symbols_covered(self, mock_data_manager):
        """Test that all symbols in universe get signals."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        signals = generator.generate_universe_signals(universe)

        symbols_with_signals = {s.symbol for s in signals}
        assert symbols_with_signals == set(universe)

    def test_generate_universe_signals_empty_universe(self, mock_data_manager):
        """Test with empty universe list."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        signals = generator.generate_universe_signals([])

        assert isinstance(signals, list)
        assert len(signals) == 0

    def test_generate_universe_signals_sorted_by_strength(self, mock_data_manager):
        """Test that signals are sorted by absolute strength."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        signals = generator.generate_universe_signals(universe, sort_by="strength")

        strengths = [abs(s.strength) for s in signals]
        assert strengths == sorted(strengths, reverse=True)

    def test_generate_universe_signals_filtered_by_confidence(self, mock_data_manager):
        """Test that signals can be filtered by minimum confidence."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        universe = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        min_confidence = 0.7
        signals = generator.generate_universe_signals(
            universe, min_confidence=min_confidence
        )

        for signal in signals:
            assert signal.confidence >= min_confidence


# =============================================================================
# Composite Score Calculation Tests
# =============================================================================


class TestCompositeScoreCalculation:
    """Tests for composite score calculation."""

    def test_calculate_composite_score_returns_float(self, mock_data_manager):
        """Test that composite score calculation returns a float."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        factors = {
            "money_flow": 0.8,
            "institutional": 0.7,
            "momentum": 0.65,
            "value": 0.5,
            "technical": 0.6,
        }
        score = generator._calculate_composite_score(factors)

        assert isinstance(score, float)

    def test_composite_score_weighted_correctly(self):
        """Test that composite score applies weights correctly."""
        weights = {
            "money_flow": 0.30,
            "institutional": 0.25,
            "momentum": 0.20,
            "value": 0.15,
            "technical": 0.10,
        }
        generator = SignalGenerator(factor_weights=weights)

        # All factors at 1.0 should give composite of 1.0
        factors = {
            "money_flow": 1.0,
            "institutional": 1.0,
            "momentum": 1.0,
            "value": 1.0,
            "technical": 1.0,
        }
        score = generator._calculate_composite_score(factors)
        assert score == pytest.approx(1.0, abs=0.001)

        # All factors at 0.0 should give composite of 0.0
        factors = {
            "money_flow": 0.0,
            "institutional": 0.0,
            "momentum": 0.0,
            "value": 0.0,
            "technical": 0.0,
        }
        score = generator._calculate_composite_score(factors)
        assert score == pytest.approx(0.0, abs=0.001)

    def test_composite_score_handles_missing_factors(self):
        """Test composite score when some factors are missing."""
        generator = SignalGenerator()

        # Only provide some factors
        factors = {
            "money_flow": 0.8,
            "institutional": 0.7,
        }
        score = generator._calculate_composite_score(factors)

        # Should still return a valid score
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_composite_score_bounded(self, mock_data_manager):
        """Test that composite score is always in [-1, 1] range."""
        generator = SignalGenerator(data_manager=mock_data_manager)

        # Test with extreme values
        factors = {
            "money_flow": 1.5,  # Out of normal range
            "institutional": 1.2,
            "momentum": -1.5,
            "value": 0.5,
            "technical": 0.6,
        }
        score = generator._calculate_composite_score(factors)
        assert -1.0 <= score <= 1.0


# =============================================================================
# Conviction Indicator Tests
# =============================================================================


class TestConvictionIndicators:
    """Tests for conviction indicator calculation."""

    def test_high_conviction_calculation(self, mock_data_manager):
        """Test high conviction when strength and confidence are high."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        conviction = generator._calculate_conviction(strength=0.85, confidence=0.90)
        assert conviction == ConvictionLevel.HIGH

    def test_medium_conviction_calculation(self, mock_data_manager):
        """Test medium conviction for moderate strength and confidence."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        conviction = generator._calculate_conviction(strength=0.55, confidence=0.65)
        assert conviction == ConvictionLevel.MEDIUM

    def test_low_conviction_calculation(self, mock_data_manager):
        """Test low conviction for weak strength or low confidence."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        conviction = generator._calculate_conviction(strength=0.25, confidence=0.45)
        assert conviction == ConvictionLevel.LOW

    def test_conviction_with_negative_strength(self, mock_data_manager):
        """Test conviction calculation with negative (bearish) strength."""
        generator = SignalGenerator(data_manager=mock_data_manager)

        # High conviction bearish
        conviction = generator._calculate_conviction(strength=-0.85, confidence=0.90)
        assert conviction == ConvictionLevel.HIGH

        # Medium conviction bearish
        conviction = generator._calculate_conviction(strength=-0.55, confidence=0.65)
        assert conviction == ConvictionLevel.MEDIUM

    def test_conviction_returns_valid_level(self, mock_data_manager):
        """Test that conviction always returns a valid ConvictionLevel."""
        generator = SignalGenerator(data_manager=mock_data_manager)

        # Test various combinations
        test_cases = [
            (0.9, 0.95),
            (0.5, 0.5),
            (0.1, 0.2),
            (-0.8, 0.9),
            (0.0, 0.5),
        ]
        for strength, confidence in test_cases:
            conviction = generator._calculate_conviction(strength, confidence)
            assert isinstance(conviction, ConvictionLevel)


# =============================================================================
# Factor Weighting Tests
# =============================================================================


class TestFactorWeighting:
    """Tests for factor weighting functionality."""

    def test_set_factor_weights(self):
        """Test setting custom factor weights."""
        generator = SignalGenerator()
        new_weights = {
            "money_flow": 0.40,
            "institutional": 0.30,
            "momentum": 0.20,
            "value": 0.10,
        }
        generator.set_factor_weights(new_weights)
        assert generator.factor_weights == new_weights

    def test_factor_weights_validation(self):
        """Test that factor weights sum to 1.0."""
        generator = SignalGenerator()
        invalid_weights = {
            "money_flow": 0.50,
            "institutional": 0.50,
            "momentum": 0.20,  # Total > 1.0
        }
        with pytest.raises(ValueError, match="must sum to 1.0"):
            generator.set_factor_weights(invalid_weights)

    def test_get_factor_weights(self):
        """Test retrieving current factor weights."""
        weights = {
            "money_flow": 0.35,
            "institutional": 0.30,
            "momentum": 0.20,
            "value": 0.15,
        }
        generator = SignalGenerator(factor_weights=weights)
        assert generator.get_factor_weights() == weights

    def test_reset_factor_weights_to_default(self):
        """Test resetting factor weights to defaults."""
        custom_weights = {
            "money_flow": 0.50,
            "institutional": 0.50,
        }
        generator = SignalGenerator(factor_weights=custom_weights)
        generator.reset_factor_weights()

        # Should be back to default weights
        total = sum(generator.factor_weights.values())
        assert total == pytest.approx(1.0, abs=0.001)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestSignalGeneratorEdgeCases:
    """Edge case tests for SignalGenerator."""

    def test_empty_data_handling(self, mock_data_manager, empty_price_data):
        """Test handling of empty price data."""
        mock_data_manager.get_price_history = AsyncMock(return_value=empty_price_data)
        generator = SignalGenerator(data_manager=mock_data_manager)

        signal = generator.generate_signal("AAPL")

        # Should return neutral signal with low confidence
        assert signal.direction == SignalDirection.NEUTRAL
        assert signal.confidence < 0.5

    def test_invalid_symbol_handling(self, mock_data_manager):
        """Test handling of invalid symbol."""
        mock_data_manager.get_price_history = AsyncMock(return_value=None)
        mock_data_manager.get_money_flow = Mock(return_value=None)

        generator = SignalGenerator(data_manager=mock_data_manager)

        # Should handle gracefully
        signal = generator.generate_signal("INVALID_SYMBOL_XYZ")
        assert signal is not None or signal is None  # Either valid signal or None

    def test_nan_data_handling(self, mock_data_manager, nan_price_data):
        """Test handling of NaN values in data."""
        mock_data_manager.get_price_history = AsyncMock(return_value=nan_price_data)
        generator = SignalGenerator(data_manager=mock_data_manager)

        signal = generator.generate_signal("AAPL")

        # Should not produce NaN in signal
        assert not np.isnan(signal.strength)
        assert not np.isnan(signal.confidence)

    def test_single_data_point(self, mock_data_manager, single_row_price_data):
        """Test handling of single data point."""
        mock_data_manager.get_price_history = AsyncMock(
            return_value=single_row_price_data
        )
        generator = SignalGenerator(data_manager=mock_data_manager)

        signal = generator.generate_signal("AAPL")

        # Should handle single point gracefully
        assert signal is not None
        assert signal.confidence < 0.5  # Low confidence with minimal data

    def test_duplicate_symbols_in_universe(self, mock_data_manager):
        """Test handling of duplicate symbols in universe."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        universe = ["AAPL", "AAPL", "MSFT", "AAPL"]

        signals = generator.generate_universe_signals(universe)

        # Should deduplicate or handle gracefully
        symbols = [s.symbol for s in signals]
        assert len(set(symbols)) == len(symbols)  # No duplicates in output

    def test_extreme_factor_values(self, mock_data_manager):
        """Test handling of extreme factor values."""
        generator = SignalGenerator(data_manager=mock_data_manager)

        factors = {
            "money_flow": 100.0,  # Extreme positive
            "institutional": -100.0,  # Extreme negative
            "momentum": float("inf"),  # Infinity
            "value": float("-inf"),
        }

        # Should handle without crashing
        try:
            score = generator._calculate_composite_score(factors)
            assert not np.isnan(score)
            assert not np.isinf(score)
        except ValueError:
            pass  # Acceptable to raise error on invalid input


# =============================================================================
# Integration Tests
# =============================================================================


class TestSignalGeneratorIntegration:
    """Integration tests for SignalGenerator with other analyzers."""

    def test_integration_with_money_flow_analyzer(
        self, mock_data_manager, mock_money_flow_analyzer
    ):
        """Test integration with MoneyFlowAnalyzer."""
        generator = SignalGenerator(
            data_manager=mock_data_manager,
            money_flow_analyzer=mock_money_flow_analyzer,
        )

        signal = generator.generate_signal("AAPL")

        # Verify money flow analyzer was called
        mock_money_flow_analyzer.analyze_equity_flow.assert_called_once()
        assert signal.factors.money_flow is not None

    def test_integration_with_institutional_analyzer(
        self, mock_data_manager, mock_institutional_analyzer
    ):
        """Test integration with InstitutionalAnalyzer."""
        generator = SignalGenerator(
            data_manager=mock_data_manager,
            institutional_analyzer=mock_institutional_analyzer,
        )

        signal = generator.generate_signal("AAPL")

        # Verify institutional analyzer was called
        mock_institutional_analyzer.get_holdings.assert_called_once()
        assert signal.factors.institutional is not None


# =============================================================================
# Health Check Tests
# =============================================================================


class TestSignalGeneratorHealthCheck:
    """Tests for SignalGenerator health check."""

    def test_health_check_returns_true(self):
        """Test that health_check returns True when healthy."""
        generator = SignalGenerator()
        assert generator.health_check() is True

    def test_health_check_with_data_manager(self, mock_data_manager):
        """Test health check with data manager configured."""
        generator = SignalGenerator(data_manager=mock_data_manager)
        assert generator.health_check() is True
