"""Tests for the Research module."""

import pytest
from datetime import datetime

from stanley.research import ResearchAnalyzer
from stanley.research.valuation import DCFResult, ValuationMetrics
from stanley.research.earnings import EarningsQuarter, EarningsAnalysis


class TestDCFResult:
    """Tests for DCFResult dataclass."""

    def test_dcf_result_creation(self):
        result = DCFResult(
            symbol="TEST",
            intrinsic_value=15000.0,
            current_price=100.0,
            upside_percentage=50.0,
            margin_of_safety=30.0,
            discount_rate=0.10,
            terminal_growth_rate=0.025,
            projection_years=5,
            pv_cash_flows=5000.0,
            pv_terminal_value=10000.0,
            net_debt=500.0,
            shares_outstanding=100.0,
        )
        assert result.intrinsic_value == 15000.0
        assert result.symbol == "TEST"


class TestValuationMetrics:
    """Tests for ValuationMetrics dataclass."""

    def test_metrics_creation(self):
        metrics = ValuationMetrics(
            symbol="AAPL",
            price=175.0,
            market_cap=2.7e12,
            enterprise_value=2.8e12,
            pe_ratio=30.0,
            forward_pe=25.0,
            peg_ratio=1.5,
            price_to_sales=7.5,
            ev_to_sales=7.8,
            price_to_book=45.0,
            price_to_tangible_book=50.0,
            ev_to_ebitda=22.0,
            price_to_fcf=28.0,
            ev_to_fcf=29.0,
            earnings_yield=3.3,
            fcf_yield=3.5,
            dividend_yield=0.5,
        )
        assert metrics.symbol == "AAPL"
        assert metrics.pe_ratio == 30.0


class TestEarningsQuarter:
    """Tests for EarningsQuarter dataclass."""

    def test_quarter_creation(self):
        quarter = EarningsQuarter(
            fiscal_quarter="Q3 2024",
            fiscal_year=2024,
            fiscal_period=3,
            eps_actual=1.50,
            revenue_actual=90e9,
            report_date=datetime(2024, 10, 31),
            eps_estimate=1.45,
            revenue_estimate=88e9,
        )
        assert quarter.fiscal_quarter == "Q3 2024"
        assert quarter.eps_actual == 1.50


class TestEarningsAnalysis:
    """Tests for EarningsAnalysis dataclass."""

    def test_analysis_creation(self):
        analysis = EarningsAnalysis(
            symbol="AAPL",
            quarters=[],
            eps_growth_yoy=5.5,
            beat_rate=75.0,
            consecutive_beats=3,
            earnings_volatility=0.05,
        )
        assert analysis.symbol == "AAPL"
        assert analysis.beat_rate == 75.0


class TestResearchAnalyzer:
    """Tests for ResearchAnalyzer."""

    def test_analyzer_creation(self):
        analyzer = ResearchAnalyzer()
        assert analyzer is not None
        assert analyzer.health_check() is True
