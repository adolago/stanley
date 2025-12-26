"""Tests for earnings quality metrics."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from stanley.accounting.earnings_quality import (
    BeneishMScore,
    PiotroskiFScore,
    AltmanZScore,
    AccrualAnalyzer,
    EarningsQualityAnalyzer,
    QualityRating,
    MScoreResult,
    FScoreResult,
    ZScoreResult,
)


class TestBeneishMScore:
    """Test Beneish M-Score calculation."""

    def test_mscore_healthy_company(self, healthy_company_data):
        """Test M-Score for a healthy company (should be < -2.22)."""
        calculator = BeneishMScore()
        result = calculator.calculate(healthy_company_data)

        assert isinstance(result, MScoreResult)
        assert result.m_score < -2.22, "Healthy company should have low M-Score"
        assert result.manipulation_risk == "Low"
        assert len(result.flags) == 0
        assert all(key in result.components for key in [
            'dsri', 'gmi', 'aqi', 'sgi', 'depi', 'sgai', 'tata', 'lvgi'
        ])

    def test_mscore_manipulator(self, manipulator_company_data):
        """Test M-Score for company with manipulation patterns (should be > -2.22)."""
        calculator = BeneishMScore()
        result = calculator.calculate(manipulator_company_data)

        assert result.m_score > -2.22, "Manipulator should have high M-Score"
        assert result.manipulation_risk in ["Moderate", "High"]
        assert len(result.flags) > 0
        assert any("Days Sales in Receivables" in flag for flag in result.flags)

    def test_mscore_components_calculation(self):
        """Test individual M-Score component calculations."""
        # Create minimal data for component testing
        data = {
            'balance_sheet': pd.DataFrame({
                'receivables': [100, 120],  # 20% increase
                'total_assets': [1000, 1100],  # 10% increase
                'current_assets': [500, 550],
                'ppe_net': [300, 320],
                'total_liabilities': [400, 450],
            }, index=['2022-12-31', '2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [1000, 1100],  # 10% increase
                'cogs': [600, 660],
                'sga': [200, 220],
            }, index=['2022-12-31', '2023-12-31']),
            'cash_flow': pd.DataFrame({
                'net_income': [100, 110],
                'operating_cash_flow': [120, 130],
            }, index=['2022-12-31', '2023-12-31']),
        }

        calculator = BeneishMScore()
        result = calculator.calculate(data)

        # DSRI should be (120/1100) / (100/1000) = 1.09
        assert abs(result.components['dsri'] - 1.09) < 0.01

        # GMI should be (1-660/1100) / (1-600/1000) = 0.40/0.40 = 1.0
        assert abs(result.components['gmi'] - 1.0) < 0.01

        # SGI should be 1100/1000 = 1.1
        assert abs(result.components['sgi'] - 1.1) < 0.01

    def test_mscore_missing_data(self):
        """Test M-Score handling of missing data."""
        data = {
            'balance_sheet': pd.DataFrame({
                'receivables': [100],
                'total_assets': [1000],
            }, index=['2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [1000],
            }, index=['2023-12-31']),
            'cash_flow': pd.DataFrame({
                'net_income': [100],
            }, index=['2023-12-31']),
        }

        calculator = BeneishMScore()
        with pytest.raises(ValueError, match="Insufficient historical data"):
            calculator.calculate(data)

    def test_mscore_zero_denominator(self):
        """Test M-Score handling of zero denominators."""
        data = {
            'balance_sheet': pd.DataFrame({
                'receivables': [0, 100],
                'total_assets': [1000, 1100],
                'current_assets': [500, 550],
                'ppe_net': [300, 320],
                'total_liabilities': [400, 450],
            }, index=['2022-12-31', '2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [0, 1100],  # Zero revenue in first period
                'cogs': [0, 660],
                'sga': [200, 220],
            }, index=['2022-12-31', '2023-12-31']),
            'cash_flow': pd.DataFrame({
                'net_income': [100, 110],
                'operating_cash_flow': [120, 130],
            }, index=['2022-12-31', '2023-12-31']),
        }

        calculator = BeneishMScore()
        result = calculator.calculate(data)

        # Should handle gracefully, possibly with NaN or default values
        assert result.m_score is not None
        assert not np.isnan(result.m_score) or "Zero denominator" in result.flags


class TestPiotroskiFScore:
    """Test Piotroski F-Score calculation."""

    def test_fscore_perfect_score(self, excellent_fundamentals_data):
        """Test F-Score for company with excellent fundamentals (9/9)."""
        calculator = PiotroskiFScore()
        result = calculator.calculate(excellent_fundamentals_data)

        assert isinstance(result, FScoreResult)
        assert result.f_score == 9
        assert result.quality_rating == "Strong"
        assert all(result.signals.values())
        assert len(result.positive_signals) == 9
        assert len(result.negative_signals) == 0

    def test_fscore_poor_fundamentals(self, poor_fundamentals_data):
        """Test F-Score for company with poor fundamentals (0-3/9)."""
        calculator = PiotroskiFScore()
        result = calculator.calculate(poor_fundamentals_data)

        assert result.f_score <= 3
        assert result.quality_rating == "Weak"
        assert len(result.negative_signals) >= 6

    def test_fscore_individual_signals(self):
        """Test each of the 9 F-Score signals individually."""
        # Profitability signals (4)
        data = {
            'cash_flow': pd.DataFrame({
                'net_income': [100],  # Positive
                'operating_cash_flow': [150],  # Positive and > NI
            }, index=['2023-12-31']),
            'balance_sheet': pd.DataFrame({
                'total_assets': [1000, 1100],
                'total_liabilities': [400, 380],  # Decreasing leverage
                'current_assets': [500, 600],
                'current_liabilities': [200, 200],  # Improving current ratio
                'shares_outstanding': [100, 100],  # No dilution
            }, index=['2022-12-31', '2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [1000, 1200],
                'cogs': [600, 720],  # Improving margin
            }, index=['2022-12-31', '2023-12-31']),
        }

        calculator = PiotroskiFScore()
        result = calculator.calculate(data)

        # Check profitability signals
        assert result.signals['roa_positive'] == True
        assert result.signals['ocf_positive'] == True
        assert result.signals['roa_improving'] == True  # Can't verify without prior year
        assert result.signals['accrual_quality'] == True  # OCF > NI

        # Check leverage signals
        assert result.signals['leverage_decreasing'] == True
        assert result.signals['current_ratio_improving'] == True
        assert result.signals['no_dilution'] == True

        # Check operating efficiency
        assert result.signals['margin_improving'] == True
        assert result.signals['turnover_improving'] == True

    def test_fscore_moderate_company(self, healthy_company_data):
        """Test F-Score for moderate company (4-6/9)."""
        calculator = PiotroskiFScore()
        result = calculator.calculate(healthy_company_data)

        assert 4 <= result.f_score <= 6
        assert result.quality_rating == "Moderate"


class TestAltmanZScore:
    """Test Altman Z-Score calculation."""

    def test_zscore_manufacturing_safe(self):
        """Test Z-Score for safe manufacturing company (>2.99)."""
        data = {
            'balance_sheet': pd.DataFrame({
                'working_capital': [500],
                'retained_earnings': [800],
                'ebit': [200],
                'total_assets': [2000],
                'total_liabilities': [800],
                'market_cap': [3000],
            }, index=['2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [2500],
            }, index=['2023-12-31']),
        }

        calculator = AltmanZScore()
        result = calculator.calculate(data, company_type='manufacturing')

        assert isinstance(result, ZScoreResult)
        assert result.z_score > 2.99
        assert result.distress_zone == "Safe"
        assert result.bankruptcy_probability < 0.05

    def test_zscore_service_safe(self):
        """Test Z-Score for safe service company (>2.6)."""
        data = {
            'balance_sheet': pd.DataFrame({
                'working_capital': [400],
                'retained_earnings': [700],
                'ebit': [180],
                'total_assets': [1500],
                'total_liabilities': [600],
                'book_equity': [900],
            }, index=['2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [2000],
            }, index=['2023-12-31']),
        }

        calculator = AltmanZScore()
        result = calculator.calculate(data, company_type='service')

        assert result.z_score > 2.6
        assert result.distress_zone == "Safe"

    def test_zscore_distress_zone(self):
        """Test Z-Score for company in distress zone."""
        data = {
            'balance_sheet': pd.DataFrame({
                'working_capital': [-100],  # Negative working capital
                'retained_earnings': [50],  # Low retained earnings
                'ebit': [20],  # Low profitability
                'total_assets': [1000],
                'total_liabilities': [950],  # High leverage
                'market_cap': [200],  # Low market value
            }, index=['2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [800],
            }, index=['2023-12-31']),
        }

        calculator = AltmanZScore()
        result = calculator.calculate(data, company_type='manufacturing')

        assert result.z_score < 1.81
        assert result.distress_zone == "Distress"
        assert result.bankruptcy_probability > 0.50

    def test_zscore_gray_zone(self):
        """Test Z-Score for company in gray zone."""
        data = {
            'balance_sheet': pd.DataFrame({
                'working_capital': [200],
                'retained_earnings': [300],
                'ebit': [100],
                'total_assets': [1200],
                'total_liabilities': [700],
                'market_cap': [800],
            }, index=['2023-12-31']),
            'income_statement': pd.DataFrame({
                'revenue': [1500],
            }, index=['2023-12-31']),
        }

        calculator = AltmanZScore()
        result = calculator.calculate(data, company_type='manufacturing')

        assert 1.81 <= result.z_score <= 2.99
        assert result.distress_zone == "Gray Zone"


class TestAccrualAnalyzer:
    """Test accrual analysis."""

    def test_accrual_ratio_normal(self):
        """Test accrual ratio for normal company."""
        data = {
            'cash_flow': pd.DataFrame({
                'net_income': [100],
                'operating_cash_flow': [110],
            }, index=['2023-12-31']),
            'balance_sheet': pd.DataFrame({
                'total_assets': [1000, 1100],
            }, index=['2022-12-31', '2023-12-31']),
        }

        analyzer = AccrualAnalyzer()
        result = analyzer.calculate_accruals(data)

        # Accruals = NI - OCF = 100 - 110 = -10
        # Accrual ratio = -10 / avg(1000, 1100) = -10/1050 = -0.0095
        assert abs(result['accrual_ratio'] - (-0.0095)) < 0.001
        assert result['quality_flag'] == False  # Low accruals

    def test_accrual_ratio_high(self):
        """Test accrual ratio for company with high accruals."""
        data = {
            'cash_flow': pd.DataFrame({
                'net_income': [200],
                'operating_cash_flow': [100],  # Large gap
            }, index=['2023-12-31']),
            'balance_sheet': pd.DataFrame({
                'total_assets': [1000, 1100],
            }, index=['2022-12-31', '2023-12-31']),
        }

        analyzer = AccrualAnalyzer()
        result = analyzer.calculate_accruals(data)

        # Accruals = 200 - 100 = 100
        # Accrual ratio = 100/1050 = 0.095
        assert result['accrual_ratio'] > 0.05
        assert result['quality_flag'] == True  # High accruals warning

    def test_working_capital_accruals(self):
        """Test working capital accrual decomposition."""
        data = {
            'balance_sheet': pd.DataFrame({
                'current_assets': [500, 600],
                'cash': [100, 120],
                'current_liabilities': [200, 250],
                'short_term_debt': [50, 60],
                'total_assets': [1000, 1100],
            }, index=['2022-12-31', '2023-12-31']),
        }

        analyzer = AccrualAnalyzer()
        result = analyzer.calculate_working_capital_accruals(data)

        # WC change = (600-120) - (500-100) - (250-60) + (200-50)
        # = 480 - 400 - 190 + 150 = 40
        assert result['wc_accruals'] == 40
        assert 'wc_accrual_ratio' in result


class TestEarningsQualityAnalyzer:
    """Test integrated earnings quality analysis."""

    @patch('stanley.accounting.earnings_quality.EdgarAdapter')
    def test_analyze_symbol_integration(self, mock_edgar, healthy_company_data):
        """Test full analysis integration for a symbol."""
        mock_adapter = Mock()
        mock_adapter.get_financial_statements.return_value = healthy_company_data
        mock_edgar.return_value = mock_adapter

        analyzer = EarningsQualityAnalyzer()
        result = analyzer.analyze('AAPL')

        assert 'overall_quality' in result
        assert 'm_score' in result
        assert 'f_score' in result
        assert 'z_score' in result
        assert 'accrual_quality' in result
        assert isinstance(result['overall_quality'], QualityRating)

    def test_aggregate_quality_rating(self):
        """Test quality rating aggregation logic."""
        analyzer = EarningsQualityAnalyzer()

        # Test high quality scenario
        scores = {
            'm_score': MScoreResult(
                m_score=-2.5, manipulation_risk="Low", components={}, flags=[]
            ),
            'f_score': FScoreResult(
                f_score=8, quality_rating="Strong", signals={},
                positive_signals=[], negative_signals=[]
            ),
            'z_score': ZScoreResult(
                z_score=3.5, distress_zone="Safe",
                bankruptcy_probability=0.01, components={}
            ),
        }

        rating = analyzer._aggregate_rating(scores)
        assert rating in [QualityRating.HIGH, QualityRating.MODERATE]

        # Test low quality scenario
        scores['m_score'] = MScoreResult(
            m_score=-1.5, manipulation_risk="High", components={},
            flags=["High DSRI"]
        )
        scores['f_score'] = FScoreResult(
            f_score=2, quality_rating="Weak", signals={},
            positive_signals=[], negative_signals=["negative_roa"]
        )
        scores['z_score'] = ZScoreResult(
            z_score=1.5, distress_zone="Distress",
            bankruptcy_probability=0.70, components={}
        )

        rating = analyzer._aggregate_rating(scores)
        assert rating in [QualityRating.LOW, QualityRating.VERY_LOW]

    def test_time_series_analysis(self, multi_year_financial_data):
        """Test earnings quality trends over multiple years."""
        analyzer = EarningsQualityAnalyzer()
        trend = analyzer.analyze_trend(multi_year_financial_data, periods=3)

        assert 'trend_direction' in trend
        assert 'quality_stability' in trend
        assert len(trend['yearly_scores']) == 3
