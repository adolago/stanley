"""
Tests for SectorRotationAnalyzer module.

Tests sector rotation pattern detection, risk-on/off regime identification,
sector momentum ranking, and leadership change detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch


# Import will be created when module exists
try:
    from stanley.analytics.sector_rotation import SectorRotationAnalyzer
except ImportError:
    SectorRotationAnalyzer = None


# Skip all tests if module not yet implemented
pytestmark = pytest.mark.skipif(
    SectorRotationAnalyzer is None,
    reason="SectorRotationAnalyzer module not yet implemented"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_sector_data():
    """Sample sector ETF price/flow data."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq="D")
    np.random.seed(42)

    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC"]

    data = []
    for sector in sectors:
        base_price = np.random.uniform(30, 150)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)

        for i, date in enumerate(dates):
            data.append({
                "date": date,
                "sector": sector,
                "close": prices[i],
                "volume": np.random.randint(10_000_000, 100_000_000),
                "net_flow": np.random.normal(0, 50_000_000),
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_rotation_matrix():
    """Sample sector rotation correlation matrix."""
    sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB"]
    n = len(sectors)

    # Create a semi-realistic correlation matrix
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = np.random.uniform(-0.3, 0.9)

    return pd.DataFrame(corr, index=sectors, columns=sectors)


@pytest.fixture
def sample_momentum_data():
    """Sample sector momentum data."""
    return pd.DataFrame({
        "sector": ["XLK", "XLE", "XLV", "XLF", "XLY", "XLP", "XLI", "XLB"],
        "momentum_1m": [0.08, -0.05, 0.03, 0.02, 0.06, 0.01, 0.04, -0.02],
        "momentum_3m": [0.15, -0.10, 0.08, 0.05, 0.12, 0.04, 0.09, -0.03],
        "momentum_6m": [0.25, -0.15, 0.12, 0.10, 0.20, 0.08, 0.15, -0.05],
        "relative_strength": [1.2, 0.7, 1.0, 0.9, 1.1, 0.85, 1.05, 0.75],
        "rank": [1, 8, 4, 5, 2, 6, 3, 7],
    })


@pytest.fixture
def empty_sector_data():
    """Empty sector data DataFrame."""
    return pd.DataFrame(columns=["date", "sector", "close", "volume", "net_flow"])


@pytest.fixture
def mock_data_manager_for_sectors():
    """Mock DataManager for sector rotation."""
    mock = Mock()
    mock.get_sector_etf_data = AsyncMock(return_value=pd.DataFrame({
        "date": pd.date_range(end=datetime.now(), periods=5, freq="D"),
        "close": [100, 101, 102, 101, 103],
        "volume": [10_000_000, 12_000_000, 11_000_000, 9_000_000, 15_000_000],
    }))
    return mock


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSectorRotationAnalyzerInit:
    """Tests for SectorRotationAnalyzer initialization."""

    def test_init_without_data_manager(self):
        """Test initialization without data_manager."""
        analyzer = SectorRotationAnalyzer()
        assert analyzer is not None
        assert analyzer.data_manager is None

    def test_init_with_data_manager(self, mock_data_manager_for_sectors):
        """Test initialization with mock data_manager."""
        analyzer = SectorRotationAnalyzer(data_manager=mock_data_manager_for_sectors)
        assert analyzer.data_manager is mock_data_manager_for_sectors

    def test_default_sectors(self):
        """Test default sector ETF list."""
        analyzer = SectorRotationAnalyzer()
        expected_sectors = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLU", "XLC"]
        assert set(analyzer.default_sectors) == set(expected_sectors)

    def test_custom_sectors(self):
        """Test custom sector list."""
        custom = ["XLK", "XLF", "XLE"]
        analyzer = SectorRotationAnalyzer(sectors=custom)
        assert analyzer.sectors == custom


# =============================================================================
# Rotation Pattern Detection Tests
# =============================================================================


class TestRotationPatternDetection:
    """Tests for rotation pattern detection."""

    def test_returns_dict(self):
        """Test that method returns a dictionary."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected keys."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        expected_keys = [
            "current_phase",
            "rotation_direction",
            "rotation_speed",
            "sector_leaders",
            "sector_laggards",
            "phase_duration",
            "confidence",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_current_phase_valid_values(self):
        """Test that current_phase is valid."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        valid_phases = ["early_cycle", "mid_cycle", "late_cycle", "recession", "recovery"]
        assert result["current_phase"] in valid_phases

    def test_rotation_direction_valid_values(self):
        """Test that rotation_direction is valid."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        valid_directions = ["into_cyclicals", "into_defensives", "mixed", "neutral"]
        assert result["rotation_direction"] in valid_directions

    def test_rotation_speed_bounded(self):
        """Test that rotation_speed is in [0, 1] range."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        assert 0 <= result["rotation_speed"] <= 1

    def test_sector_leaders_is_list(self):
        """Test that sector_leaders is a list."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        assert isinstance(result["sector_leaders"], list)

    def test_sector_laggards_is_list(self):
        """Test that sector_laggards is a list."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        assert isinstance(result["sector_laggards"], list)

    def test_confidence_bounded(self):
        """Test that confidence is in [0, 1] range."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern()
        assert 0 <= result["confidence"] <= 1


# =============================================================================
# Risk-On/Off Regime Tests
# =============================================================================


class TestRiskOnOffRegime:
    """Tests for risk_on_off regime identification."""

    def test_returns_dict(self):
        """Test that method returns a dictionary."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.identify_risk_regime()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected keys."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.identify_risk_regime()
        expected_keys = [
            "regime",
            "regime_strength",
            "cyclical_vs_defensive_ratio",
            "risk_appetite_score",
            "regime_duration_days",
            "transition_probability",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_regime_valid_values(self):
        """Test that regime is valid."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.identify_risk_regime()
        valid_regimes = ["risk_on", "risk_off", "neutral", "transitioning"]
        assert result["regime"] in valid_regimes

    def test_regime_strength_bounded(self):
        """Test that regime_strength is in [0, 1] range."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.identify_risk_regime()
        assert 0 <= result["regime_strength"] <= 1

    def test_risk_appetite_score_bounded(self):
        """Test that risk_appetite_score is in [-1, 1] range."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.identify_risk_regime()
        assert -1 <= result["risk_appetite_score"] <= 1

    def test_transition_probability_bounded(self):
        """Test that transition_probability is in [0, 1] range."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.identify_risk_regime()
        assert 0 <= result["transition_probability"] <= 1


# =============================================================================
# Sector Momentum Ranking Tests
# =============================================================================


class TestSectorMomentumRanking:
    """Tests for sector momentum ranking."""

    def test_returns_dataframe(self):
        """Test that method returns a DataFrame."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.rank_sector_momentum()
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        """Test that DataFrame has expected columns."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.rank_sector_momentum()
        expected_cols = [
            "sector", "momentum_1m", "momentum_3m", "momentum_6m",
            "relative_strength", "rank"
        ]
        for col in expected_cols:
            if not result.empty:
                assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_rank(self):
        """Test that result is sorted by rank ascending."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.rank_sector_momentum()
        if len(result) > 1 and "rank" in result.columns:
            for i in range(len(result) - 1):
                assert result["rank"].iloc[i] <= result["rank"].iloc[i + 1]

    def test_momentum_period_parameter(self):
        """Test momentum_period parameter."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.rank_sector_momentum(primary_period="3m")
        assert isinstance(result, pd.DataFrame)

    def test_relative_strength_non_negative(self):
        """Test that relative_strength is non-negative."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.rank_sector_momentum()
        if not result.empty and "relative_strength" in result.columns:
            assert all(result["relative_strength"] >= 0)

    def test_custom_sectors(self):
        """Test with custom sector list."""
        analyzer = SectorRotationAnalyzer()
        custom_sectors = ["XLK", "XLF", "XLE"]
        result = analyzer.rank_sector_momentum(sectors=custom_sectors)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert len(result) <= len(custom_sectors)


# =============================================================================
# Leadership Change Detection Tests
# =============================================================================


class TestLeadershipChangeDetection:
    """Tests for leadership change detection."""

    def test_returns_dict(self):
        """Test that method returns a dictionary."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change()
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        """Test that result has expected keys."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change()
        expected_keys = [
            "leadership_changed",
            "new_leaders",
            "former_leaders",
            "change_magnitude",
            "days_since_change",
            "leadership_stability",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_leadership_changed_is_boolean(self):
        """Test that leadership_changed is a boolean."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change()
        assert isinstance(result["leadership_changed"], bool)

    def test_new_leaders_is_list(self):
        """Test that new_leaders is a list."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change()
        assert isinstance(result["new_leaders"], list)

    def test_former_leaders_is_list(self):
        """Test that former_leaders is a list."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change()
        assert isinstance(result["former_leaders"], list)

    def test_change_magnitude_bounded(self):
        """Test that change_magnitude is in [0, 1] range."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change()
        assert 0 <= result["change_magnitude"] <= 1

    def test_leadership_stability_bounded(self):
        """Test that leadership_stability is in [0, 1] range."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change()
        assert 0 <= result["leadership_stability"] <= 1

    def test_lookback_period_parameter(self):
        """Test lookback_period parameter."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_leadership_change(lookback_days=30)
        assert isinstance(result, dict)


# =============================================================================
# Sector Flow Analysis Tests
# =============================================================================


class TestSectorFlowAnalysis:
    """Tests for sector flow analysis integration."""

    def test_get_sector_flows_returns_dataframe(self):
        """Test that get_sector_flows returns a DataFrame."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.get_sector_flows()
        assert isinstance(result, pd.DataFrame)

    def test_sector_flows_has_expected_columns(self):
        """Test that sector flows DataFrame has expected columns."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.get_sector_flows()
        expected_cols = ["sector", "net_flow_1d", "net_flow_5d", "net_flow_20d", "flow_trend"]
        if not result.empty:
            for col in expected_cols:
                assert col in result.columns, f"Missing column: {col}"

    def test_flow_trend_valid_values(self):
        """Test that flow_trend values are valid."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.get_sector_flows()
        valid_trends = ["accelerating_inflow", "decelerating_inflow", "accelerating_outflow", "decelerating_outflow", "neutral"]
        if not result.empty and "flow_trend" in result.columns:
            assert all(result["flow_trend"].isin(valid_trends))


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestSectorRotationEdgeCases:
    """Edge case tests for SectorRotationAnalyzer."""

    def test_empty_sector_data(self, empty_sector_data):
        """Test with empty sector data."""
        analyzer = SectorRotationAnalyzer()
        with patch.object(analyzer, '_get_sector_data', return_value=empty_sector_data):
            result = analyzer.detect_rotation_pattern()
            assert result["confidence"] == 0.0

    def test_single_sector(self):
        """Test with single sector."""
        analyzer = SectorRotationAnalyzer(sectors=["XLK"])
        result = analyzer.detect_rotation_pattern()
        assert isinstance(result, dict)

    def test_very_short_lookback(self):
        """Test with very short lookback period."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern(lookback_days=1)
        assert isinstance(result, dict)

    def test_very_long_lookback(self):
        """Test with very long lookback period."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.detect_rotation_pattern(lookback_days=365)
        assert isinstance(result, dict)

    def test_all_sectors_same_performance(self, sample_sector_data):
        """Test when all sectors have same performance."""
        analyzer = SectorRotationAnalyzer()
        # Make all returns the same
        sample_sector_data["close"] = 100.0
        with patch.object(analyzer, '_get_sector_data', return_value=sample_sector_data):
            result = analyzer.rank_sector_momentum()
            # Should handle gracefully
            assert isinstance(result, pd.DataFrame)

    def test_nan_handling(self, sample_sector_data):
        """Test handling of NaN values."""
        analyzer = SectorRotationAnalyzer()
        sample_sector_data.loc[0, "close"] = np.nan
        sample_sector_data.loc[5, "net_flow"] = np.nan
        with patch.object(analyzer, '_get_sector_data', return_value=sample_sector_data):
            result = analyzer.detect_rotation_pattern()
            assert isinstance(result, dict)

    def test_extreme_momentum_values(self, sample_momentum_data):
        """Test with extreme momentum values."""
        analyzer = SectorRotationAnalyzer()
        sample_momentum_data["momentum_1m"] = [10.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Extreme
        with patch.object(analyzer, '_calculate_momentum', return_value=sample_momentum_data):
            result = analyzer.rank_sector_momentum()
            assert isinstance(result, pd.DataFrame)


# =============================================================================
# Comprehensive Analysis Tests
# =============================================================================


class TestComprehensiveAnalysis:
    """Tests for comprehensive sector analysis."""

    def test_get_sector_analysis_returns_dict(self):
        """Test that get_sector_analysis returns a dictionary."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.get_sector_analysis()
        assert isinstance(result, dict)

    def test_sector_analysis_has_all_components(self):
        """Test that sector analysis includes all components."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.get_sector_analysis()
        expected_keys = [
            "rotation_pattern",
            "risk_regime",
            "momentum_rankings",
            "leadership_change",
            "sector_flows",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_sector_correlation_matrix(self):
        """Test sector correlation matrix calculation."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.get_sector_correlation_matrix()
        assert isinstance(result, pd.DataFrame)
        # Should be square
        if not result.empty:
            assert result.shape[0] == result.shape[1]


# =============================================================================
# Health Check Tests
# =============================================================================


class TestSectorRotationHealthCheck:
    """Tests for health_check method."""

    def test_returns_true(self):
        """Test that health_check returns True."""
        analyzer = SectorRotationAnalyzer()
        assert analyzer.health_check() is True

    def test_returns_dict_with_details(self):
        """Test that health_check can return detailed status."""
        analyzer = SectorRotationAnalyzer()
        result = analyzer.health_check(detailed=True)
        if isinstance(result, dict):
            assert "status" in result
        else:
            assert result is True
