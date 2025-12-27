"""
Commodities Router

API endpoints for commodity market analysis including:
- Market overview by category
- Individual commodity details
- Macro-commodity linkages
- Correlation matrices
- Futures curve analysis
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/commodities", tags=["Commodities"])


# =============================================================================
# Response Models
# =============================================================================


class CommodityPrice(BaseModel):
    """Current commodity price data."""

    symbol: str
    name: str
    price: float
    change: float
    changePercent: float
    volume: Optional[int] = None
    timestamp: str


class CommoditySummaryResponse(BaseModel):
    """Detailed commodity summary."""

    symbol: str
    name: str
    category: str
    price: float
    change1d: float
    change1w: float
    change1m: float
    changeYtd: float
    volatility30d: float
    trend: str
    relativeStrength: float


class MacroLinkageItem(BaseModel):
    """Single macro-commodity relationship."""

    commodity: str
    macroIndicator: str
    correlation: float
    leadLagDays: int
    relationship: str
    strength: str


class MacroLinkageResponse(BaseModel):
    """Macro linkage analysis response."""

    commodity: str
    name: str
    category: str
    linkages: List[MacroLinkageItem]
    primaryDriver: Optional[str] = None


class CategoryOverview(BaseModel):
    """Commodity category overview."""

    category: str
    count: int
    avgChange: float
    leader: Optional[CommodityPrice] = None
    laggard: Optional[CommodityPrice] = None
    commodities: List[CommodityPrice]


class MarketOverviewResponse(BaseModel):
    """Complete commodity market overview."""

    timestamp: str
    sentiment: str
    avgChange: float
    categories: Dict[str, CategoryOverview]


class CorrelationMatrixResponse(BaseModel):
    """Commodity correlation matrix."""

    commodities: List[str]
    matrix: Dict[str, Dict[str, float]]


class FuturesCurvePoint(BaseModel):
    """Single point on futures curve."""

    contract: str
    expiry: str
    price: float
    openInterest: int
    volume: int


class FuturesCurveResponse(BaseModel):
    """Futures curve analysis."""

    symbol: str
    name: str
    curveShape: str  # contango, backwardation, flat
    frontMonth: FuturesCurvePoint
    curve: List[FuturesCurvePoint]
    rollYield: float
    timestamp: str


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None


def create_response(
    data: Any = None, error: str = None, success: bool = True
) -> ApiResponse:
    """Create standardized API response."""
    from datetime import datetime

    return ApiResponse(
        success=success if error is None else False,
        data=data,
        error=error,
        timestamp=datetime.now().isoformat(),
    )


def get_app_state():
    """Get application state from main module."""
    from ..main import app_state

    return app_state


# =============================================================================
# Endpoints
# =============================================================================


@router.get("", response_model=ApiResponse)
async def get_commodities_overview():
    """
    Get commodity market overview.

    Returns prices and trends across all commodity categories:
    - Energy (crude oil, natural gas, etc.)
    - Precious Metals (gold, silver, platinum)
    - Base Metals (copper, aluminum, etc.)
    - Agriculture (corn, wheat, soybeans)
    - Softs (coffee, sugar, cocoa)
    """
    try:
        app_state = get_app_state()

        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        overview = await app_state.commodities_analyzer.get_market_overview()

        return create_response(data=overview)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching commodities overview: {e}")
        return create_response(error=str(e), success=False)


@router.get("/correlations", response_model=ApiResponse)
async def get_commodity_correlations(
    commodities: Optional[str] = Query(
        None, description="Comma-separated list of commodity symbols (e.g., CL,GC,NG)"
    ),
    lookback_days: int = Query(
        252, ge=30, le=756, description="Days of historical data for calculation"
    ),
):
    """
    Get correlation matrix for commodities.

    Calculates rolling correlations between commodity returns over the
    specified lookback period. Useful for:
    - Portfolio diversification analysis
    - Identifying regime changes
    - Pairs trading opportunities

    Args:
        commodities: Comma-separated list of symbols (default: major commodities)
        lookback_days: Historical period for correlation calculation
    """
    try:
        app_state = get_app_state()

        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        symbols = commodities.split(",") if commodities else None
        corr_matrix = await app_state.commodities_analyzer.get_correlations(
            symbols, lookback_days
        )

        if corr_matrix.empty:
            return create_response(data={"commodities": [], "matrix": {}})

        # Convert to JSON-serializable format
        return create_response(
            data={
                "commodities": list(corr_matrix.columns),
                "matrix": corr_matrix.round(3).to_dict(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching commodity correlations: {e}")
        return create_response(error=str(e), success=False)


@router.get("/futures-curve/{symbol}", response_model=ApiResponse)
async def get_futures_curve(
    symbol: str,
    num_contracts: int = Query(
        6, ge=2, le=12, description="Number of future contracts to include"
    ),
):
    """
    Get futures curve for a commodity.

    Analyzes the term structure of futures prices:
    - Contango: Future prices higher than spot (storage costs)
    - Backwardation: Future prices lower than spot (supply tightness)
    - Roll yield implications

    Args:
        symbol: Commodity symbol (e.g., CL for crude oil)
        num_contracts: Number of contracts to show on curve

    Returns:
        Futures curve data with roll yield calculation
    """
    try:
        symbol = symbol.upper()
        app_state = get_app_state()

        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        # Get commodity details
        from ...commodities import get_commodity

        commodity = get_commodity(symbol)
        if not commodity:
            raise HTTPException(status_code=404, detail=f"Unknown commodity: {symbol}")

        # Get current price for mock curve generation
        price = await app_state.commodities_analyzer.price_provider.get_price(symbol)

        # Generate mock futures curve (in production, fetch from exchange)
        import numpy as np
        from datetime import datetime, timedelta

        contracts = []
        base_price = price.price
        months = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
        current_month = datetime.now().month

        # Simulate curve shape based on commodity type
        if commodity.category.value in ["energy", "agriculture"]:
            # Often in backwardation during supply tightness
            curve_slope = np.random.uniform(-0.02, 0.01)
        else:
            # Metals typically in contango
            curve_slope = np.random.uniform(0.005, 0.03)

        for i in range(num_contracts):
            month_idx = (current_month + i) % 12
            year = datetime.now().year + ((current_month + i - 1) // 12)
            expiry = datetime(year, month_idx + 1, 15)

            contract_price = base_price * (1 + curve_slope * (i + 1))
            contracts.append(
                FuturesCurvePoint(
                    contract=f"{symbol}{months[month_idx]}{str(year)[-2:]}",
                    expiry=expiry.strftime("%Y-%m-%d"),
                    price=round(contract_price, 2),
                    openInterest=int(np.random.uniform(10000, 100000)),
                    volume=int(np.random.uniform(5000, 50000)),
                )
            )

        # Determine curve shape
        if len(contracts) >= 2:
            price_diff = contracts[-1].price - contracts[0].price
            if price_diff > base_price * 0.01:
                curve_shape = "contango"
            elif price_diff < -base_price * 0.01:
                curve_shape = "backwardation"
            else:
                curve_shape = "flat"
        else:
            curve_shape = "flat"

        # Calculate annualized roll yield
        if len(contracts) >= 2:
            front_price = contracts[0].price
            next_price = contracts[1].price
            # Approximate 1-month roll yield, annualized
            roll_yield = ((front_price - next_price) / front_price) * 12 * 100
        else:
            roll_yield = 0

        return create_response(
            data={
                "symbol": symbol,
                "name": commodity.name,
                "curveShape": curve_shape,
                "frontMonth": contracts[0].model_dump() if contracts else None,
                "curve": [c.model_dump() for c in contracts],
                "rollYield": round(roll_yield, 2),
                "timestamp": datetime.now().isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching futures curve for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@router.get("/{symbol}", response_model=ApiResponse)
async def get_commodity_detail(
    symbol: str,
    lookback_days: int = Query(
        252, ge=30, le=756, description="Days of historical data for analysis"
    ),
):
    """
    Get detailed analysis for a specific commodity.

    Returns comprehensive analysis including:
    - Current price and recent changes
    - Trend analysis (bullish/bearish/neutral)
    - Volatility metrics
    - Relative strength vs category peers

    Args:
        symbol: Commodity symbol (e.g., CL, GC, NG, ZC, ZW)
        lookback_days: Days of history for calculations
    """
    try:
        symbol = symbol.upper()
        app_state = get_app_state()

        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        summary = await app_state.commodities_analyzer.get_summary(
            symbol, lookback_days
        )

        return create_response(data=summary.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching commodity data for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@router.get("/{symbol}/macro", response_model=ApiResponse)
async def get_commodity_macro_linkage(
    symbol: str,
    lookback_days: int = Query(
        252, ge=60, le=756, description="Days of historical data for analysis"
    ),
):
    """
    Get macro-commodity linkage analysis.

    Analyzes relationships between commodity and macroeconomic factors:
    - USD correlation (commodities are USD-denominated)
    - Inflation linkages
    - Economic growth indicators (PMI, industrial production)
    - Safe-haven dynamics (for precious metals)

    Args:
        symbol: Commodity symbol
        lookback_days: Days of history for correlation analysis
    """
    try:
        symbol = symbol.upper()
        app_state = get_app_state()

        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        linkage = await app_state.commodities_analyzer.analyze_macro_linkage(
            symbol, lookback_days
        )

        return create_response(data=linkage)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing macro linkage for {symbol}: {e}")
        return create_response(error=str(e), success=False)
