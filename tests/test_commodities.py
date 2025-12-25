"""Tests for the Commodities module."""

import pytest
from datetime import datetime

from stanley.commodities import (
    CommoditiesAnalyzer,
    CommoditySummary,
    MacroLinkage,
    CommodityPriceProvider,
    CommodityPrice,
    Commodity,
    CommodityCategory,
    COMMODITY_REGISTRY,
    get_commodity,
    get_commodities_by_category,
    list_all_commodities,
)


class TestCommodityRegistry:
    """Tests for commodity registry."""

    def test_registry_not_empty(self):
        assert len(COMMODITY_REGISTRY) > 0

    def test_registry_has_major_commodities(self):
        major = ["CL", "GC", "SI", "HG", "ZC", "NG"]
        for symbol in major:
            assert symbol in COMMODITY_REGISTRY

    def test_get_commodity_exists(self):
        comm = get_commodity("CL")
        assert comm is not None
        assert comm.symbol == "CL"
        assert comm.name == "Crude Oil (WTI)"
        assert comm.category == CommodityCategory.ENERGY

    def test_get_commodity_not_exists(self):
        comm = get_commodity("INVALID")
        assert comm is None

    def test_get_commodity_case_insensitive(self):
        comm1 = get_commodity("cl")
        comm2 = get_commodity("CL")
        assert comm1 is not None
        assert comm1.symbol == comm2.symbol


class TestCommodityCategory:
    """Tests for commodity categories."""

    def test_all_categories_exist(self):
        categories = [
            CommodityCategory.ENERGY,
            CommodityCategory.PRECIOUS_METALS,
            CommodityCategory.BASE_METALS,
            CommodityCategory.AGRICULTURE,
            CommodityCategory.SOFTS,
            CommodityCategory.LIVESTOCK,
        ]
        for cat in categories:
            assert cat.value is not None

    def test_get_commodities_by_category(self):
        energy = get_commodities_by_category(CommodityCategory.ENERGY)
        assert len(energy) > 0
        assert all(c.category == CommodityCategory.ENERGY for c in energy)

    def test_list_all_commodities(self):
        all_symbols = list_all_commodities()
        assert len(all_symbols) == len(COMMODITY_REGISTRY)
        assert "CL" in all_symbols
        assert "GC" in all_symbols


class TestCommodity:
    """Tests for Commodity dataclass."""

    def test_commodity_creation(self):
        comm = Commodity(
            symbol="TEST",
            name="Test Commodity",
            category=CommodityCategory.ENERGY,
            unit="USD/unit",
            exchange="TEST",
            description="A test commodity",
        )
        assert comm.symbol == "TEST"
        assert comm.category == CommodityCategory.ENERGY


class TestCommodityPrice:
    """Tests for CommodityPrice dataclass."""

    def test_price_creation(self):
        price = CommodityPrice(
            symbol="CL",
            name="Crude Oil",
            price=75.50,
            change=1.25,
            change_percent=1.68,
            high=76.00,
            low=74.00,
            volume=500000,
            open_interest=1000000,
            timestamp=datetime.now(),
        )
        assert price.symbol == "CL"
        assert price.price == 75.50


class TestCommodityPriceProvider:
    """Tests for CommodityPriceProvider."""

    def test_provider_creation(self):
        provider = CommodityPriceProvider()
        assert provider is not None


class TestCommoditySummary:
    """Tests for CommoditySummary dataclass."""

    def test_summary_creation(self):
        summary = CommoditySummary(
            symbol="CL",
            name="Crude Oil (WTI)",
            category="energy",
            price=75.50,
            change_1d=1.5,
            change_1w=3.2,
            change_1m=-2.1,
            change_ytd=15.5,
            volatility_30d=25.0,
            trend="bullish",
            relative_strength=5.5,
        )
        assert summary.symbol == "CL"
        assert summary.trend == "bullish"


class TestMacroLinkage:
    """Tests for MacroLinkage dataclass."""

    def test_linkage_creation(self):
        linkage = MacroLinkage(
            commodity="CL",
            macro_indicator="USD Index",
            correlation=-0.65,
            lead_lag_days=0,
            relationship="Inverse - weak USD supports oil",
            strength="strong",
        )
        assert linkage.commodity == "CL"
        assert linkage.correlation == -0.65


class TestCommoditiesAnalyzer:
    """Tests for CommoditiesAnalyzer."""

    def test_analyzer_creation(self):
        analyzer = CommoditiesAnalyzer()
        assert analyzer is not None
        assert analyzer.health_check() is True
