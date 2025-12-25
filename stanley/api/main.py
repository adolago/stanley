"""
Stanley REST API

FastAPI-based REST API for the Stanley institutional investment analysis platform.
Provides endpoints for the stanley-ui (Tauri/React) application.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..analytics.institutional import InstitutionalAnalyzer
from ..analytics.money_flow import MoneyFlowAnalyzer
from ..commodities import CommoditiesAnalyzer
from ..data.data_manager import DataManager
from ..notes import NoteManager
from ..portfolio import PortfolioAnalyzer
from ..research import ResearchAnalyzer

logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models - Request Types
# =============================================================================


class MoneyFlowRequest(BaseModel):
    """Request body for money flow analysis."""

    sectors: List[str] = Field(
        ..., description="List of sector ETF symbols (e.g., ['XLK', 'XLF', 'XLE'])"
    )
    lookback_days: int = Field(
        default=63, ge=1, le=365, description="Number of days to analyze"
    )


class PortfolioHoldingInput(BaseModel):
    """Individual holding in a portfolio."""

    symbol: str = Field(..., description="Stock symbol")
    shares: float = Field(..., ge=0, description="Number of shares held")
    average_cost: Optional[float] = Field(
        default=None, ge=0, description="Average cost per share"
    )


class PortfolioAnalyticsRequest(BaseModel):
    """Request body for portfolio analytics."""

    holdings: List[PortfolioHoldingInput] = Field(
        ..., description="List of portfolio holdings"
    )


# =============================================================================
# Pydantic Models - Response Types (matching stanley-ui types)
# =============================================================================


class MarketData(BaseModel):
    """Stock market data response."""

    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int
    marketCap: Optional[float] = None
    timestamp: str


class InstitutionalHolding(BaseModel):
    """Institutional holding data."""

    managerName: str
    managerCik: str
    sharesHeld: int
    valueHeld: float
    ownershipPercentage: float
    changeFromLastQuarter: Optional[float] = None


class MoneyFlowData(BaseModel):
    """Money flow analysis data."""

    symbol: str
    netFlow1m: float
    netFlow3m: float
    institutionalChange: float
    smartMoneySentiment: float
    flowAcceleration: float
    confidenceScore: float


class PortfolioHolding(BaseModel):
    """Portfolio holding with current values."""

    symbol: str
    shares: float
    averageCost: float
    currentPrice: float
    marketValue: float
    weight: float


class PortfolioAnalytics(BaseModel):
    """Portfolio analytics response."""

    totalValue: float
    totalReturn: float
    totalReturnPercent: float
    beta: float
    var95: float
    var99: float
    sectorExposure: Dict[str, float]
    topHoldings: List[PortfolioHolding]


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    components: Dict[str, bool]
    timestamp: str


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state container."""

    def __init__(self):
        self.data_manager: Optional[DataManager] = None
        self.money_flow_analyzer: Optional[MoneyFlowAnalyzer] = None
        self.institutional_analyzer: Optional[InstitutionalAnalyzer] = None
        self.portfolio_analyzer: Optional[PortfolioAnalyzer] = None
        self.research_analyzer: Optional[ResearchAnalyzer] = None
        self.commodities_analyzer: Optional[CommoditiesAnalyzer] = None
        self.note_manager: Optional[NoteManager] = None


app_state = AppState()


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Stanley API starting up...")

    # Initialize data manager with mock fallback
    app_state.data_manager = DataManager(use_mock=False)
    try:
        await app_state.data_manager.initialize()
    except Exception as e:
        logger.warning(f"Data manager initialization failed, using mock: {e}")
        app_state.data_manager = DataManager(use_mock=True)
        await app_state.data_manager.initialize()

    # Initialize analyzers
    app_state.money_flow_analyzer = MoneyFlowAnalyzer(app_state.data_manager)
    app_state.institutional_analyzer = InstitutionalAnalyzer(app_state.data_manager)
    app_state.portfolio_analyzer = PortfolioAnalyzer(app_state.data_manager)
    app_state.research_analyzer = ResearchAnalyzer(app_state.data_manager)
    app_state.commodities_analyzer = CommoditiesAnalyzer(app_state.data_manager)

    # Initialize note manager
    try:
        app_state.note_manager = NoteManager()
        logger.info("Note manager initialized")
    except Exception as e:
        logger.warning(f"Note manager initialization failed: {e}")

    logger.info("Stanley API ready")

    yield

    # Shutdown
    logger.info("Stanley API shutting down...")
    if app_state.data_manager:
        await app_state.data_manager.close()
    logger.info("Stanley API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="Stanley API",
    description="Institutional investment analysis API for Stanley",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for Tauri dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",  # Tauri dev server
        "http://127.0.0.1:1420",
        "tauri://localhost",  # Tauri production
        "https://tauri.localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.utcnow().isoformat() + "Z"


def create_response(
    data: Any = None, error: Optional[str] = None, success: bool = True
) -> ApiResponse:
    """Create a standardized API response."""
    return ApiResponse(
        success=success and error is None,
        data=data,
        error=error,
        timestamp=get_timestamp(),
    )


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns the status of the API and its components.
    """
    components = {
        "api": True,
        "data_manager": False,
        "money_flow_analyzer": False,
        "institutional_analyzer": False,
        "portfolio_analyzer": False,
        "research_analyzer": False,
        "commodities_analyzer": False,
    }

    try:
        if app_state.data_manager:
            components["data_manager"] = await app_state.data_manager.health_check()
        if app_state.money_flow_analyzer:
            components["money_flow_analyzer"] = (
                app_state.money_flow_analyzer.health_check()
            )
        if app_state.institutional_analyzer:
            components["institutional_analyzer"] = (
                app_state.institutional_analyzer.health_check()
            )
        if app_state.portfolio_analyzer:
            components["portfolio_analyzer"] = (
                app_state.portfolio_analyzer.health_check()
            )
        if app_state.research_analyzer:
            components["research_analyzer"] = (
                app_state.research_analyzer.health_check()
            )
        if app_state.commodities_analyzer:
            components["commodities_analyzer"] = (
                app_state.commodities_analyzer.health_check()
            )
    except Exception as e:
        logger.error(f"Health check error: {e}")

    all_healthy = all(components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        components=components,
        timestamp=get_timestamp(),
    )


@app.get("/api/market/{symbol}", response_model=ApiResponse, tags=["Market Data"])
async def get_market_data(symbol: str):
    """
    Get market data for a symbol.

    Returns current price, change, volume, and other market data.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    try:
        symbol = symbol.upper()

        if not app_state.data_manager:
            raise HTTPException(status_code=503, detail="Data manager not initialized")

        # Get stock data for the last few days to calculate change
        end_date = datetime.now()
        start_date = datetime(end_date.year, end_date.month, end_date.day - 5)

        try:
            stock_data = await app_state.data_manager.get_stock_data(
                symbol, start_date, end_date
            )
        except Exception as e:
            logger.warning(f"Failed to fetch stock data for {symbol}: {e}")
            # Return mock data as fallback
            stock_data = pd.DataFrame({
                'date': pd.date_range(start=start_date, end=end_date, freq='D'),
                'close': [150.0 + np.random.uniform(-5, 5) for _ in range(6)],
                'volume': [np.random.randint(10000000, 100000000) for _ in range(6)],
            })

        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        # Get the latest values
        latest = stock_data.iloc[-1]
        previous = stock_data.iloc[-2] if len(stock_data) > 1 else stock_data.iloc[-1]

        current_price = float(latest["close"])
        previous_close = float(previous["close"])
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close != 0 else 0

        market_data = MarketData(
            symbol=symbol,
            price=round(current_price, 2),
            change=round(change, 2),
            changePercent=round(change_percent, 2),
            volume=int(latest["volume"]),
            marketCap=None,  # Would require additional data source
            timestamp=get_timestamp(),
        )

        return create_response(data=market_data.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/{symbol}", response_model=ApiResponse, tags=["Institutional"]
)
async def get_institutional_holdings(symbol: str):
    """
    Get institutional holdings data for a symbol.

    Returns 13F institutional holdings including top holders and ownership percentages.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        # Get holdings data from the analyzer
        holdings_data = app_state.institutional_analyzer.get_holdings(symbol)

        # Convert top holders DataFrame to list of InstitutionalHolding
        holdings_list = []
        top_holders = holdings_data.get("top_holders", pd.DataFrame())

        if isinstance(top_holders, pd.DataFrame) and not top_holders.empty:
            for _, row in top_holders.iterrows():
                holdings_list.append(
                    InstitutionalHolding(
                        managerName=str(row.get("manager_name", "Unknown")),
                        managerCik=str(row.get("manager_cik", "0000000000")),
                        sharesHeld=int(row.get("shares_held", 0)),
                        valueHeld=float(row.get("value_held", 0)),
                        ownershipPercentage=float(
                            row.get("ownership_percentage", 0) * 100
                        ),  # Convert to percentage
                        changeFromLastQuarter=None,
                    ).model_dump()
                )
        else:
            # Use the 13F holdings data directly
            holdings_df = app_state.institutional_analyzer._get_13f_holdings(symbol)
            for _, row in holdings_df.iterrows():
                holdings_list.append(
                    InstitutionalHolding(
                        managerName=str(row.get("manager_name", "Unknown")),
                        managerCik=str(row.get("manager_cik", "0000000000")),
                        sharesHeld=int(row.get("shares_held", 0)),
                        valueHeld=float(row.get("value_held", 0)),
                        ownershipPercentage=float(
                            row.get("ownership_percentage", 0) * 100
                        ),
                        changeFromLastQuarter=None,
                    ).model_dump()
                )

        return create_response(data=holdings_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching institutional holdings for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/money-flow", response_model=ApiResponse, tags=["Analytics"])
async def analyze_money_flow(request: MoneyFlowRequest):
    """
    Analyze money flow across sectors.

    Returns money flow metrics for each sector including net flows,
    institutional changes, and smart money sentiment.

    Args:
        request: MoneyFlowRequest with list of sector ETFs
    """
    try:
        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        if not request.sectors:
            return create_response(data=[])

        # Analyze sector flow
        flow_df = app_state.money_flow_analyzer.analyze_sector_flow(
            sectors=request.sectors, lookback_days=request.lookback_days
        )

        # Convert DataFrame to list of MoneyFlowData
        flow_list = []

        if flow_df.empty:
            # Return empty list if no data
            return create_response(data=[])

        for sector in flow_df.index:
            row = flow_df.loc[sector]
            flow_list.append(
                MoneyFlowData(
                    symbol=str(sector),
                    netFlow1m=float(row.get("net_flow_1m", 0)),
                    netFlow3m=float(row.get("net_flow_3m", 0)),
                    institutionalChange=float(row.get("institutional_change", 0)),
                    smartMoneySentiment=float(row.get("smart_money_sentiment", 0)),
                    flowAcceleration=float(row.get("flow_acceleration", 0)),
                    confidenceScore=float(row.get("confidence_score", 0)),
                ).model_dump()
            )

        return create_response(data=flow_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing money flow: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/portfolio-analytics", response_model=ApiResponse, tags=["Portfolio"])
async def analyze_portfolio(request: PortfolioAnalyticsRequest):
    """
    Analyze portfolio holdings.

    Returns portfolio analytics including total value, returns, beta,
    VaR metrics, and sector exposure.

    Args:
        request: PortfolioAnalyticsRequest with list of holdings
    """
    try:
        if not app_state.portfolio_analyzer:
            raise HTTPException(status_code=503, detail="Portfolio analyzer not initialized")

        if not request.holdings:
            return create_response(data=None, error="No holdings provided")

        # Convert request holdings to analyzer format
        holdings_input = [
            {
                "symbol": h.symbol.upper(),
                "shares": h.shares,
                "average_cost": h.average_cost or 0,
            }
            for h in request.holdings
        ]

        # Use real portfolio analyzer
        summary = await app_state.portfolio_analyzer.analyze(holdings_input)

        # Convert to API response format
        analytics = PortfolioAnalytics(
            totalValue=summary.total_value,
            totalReturn=summary.total_return,
            totalReturnPercent=summary.total_return_percent,
            beta=summary.beta,
            var95=summary.var_95,
            var99=summary.var_99,
            sectorExposure=summary.sector_exposure,
            topHoldings=[
                PortfolioHolding(
                    symbol=h["symbol"],
                    shares=h["shares"],
                    averageCost=h.get("average_cost", h.get("averageCost", 0)),
                    currentPrice=h.get("current_price", h.get("currentPrice", 0)),
                    marketValue=h.get("market_value", h.get("marketValue", 0)),
                    weight=h.get("weight", 0),
                ).model_dump()
                for h in summary.top_holdings
            ],
        )

        return create_response(data=analytics.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Additional Utility Endpoints
# =============================================================================


@app.get("/api/dark-pool/{symbol}", response_model=ApiResponse, tags=["Analytics"])
async def get_dark_pool_activity(symbol: str, lookback_days: int = 20):
    """
    Get dark pool activity for a symbol.

    Returns dark pool volume and large block activity data.

    Args:
        symbol: Stock ticker symbol
        lookback_days: Number of days to analyze (default: 20)
    """
    try:
        symbol = symbol.upper()

        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        dark_pool_df = app_state.money_flow_analyzer.get_dark_pool_activity(
            symbol, lookback_days
        )

        # Convert to response format
        dark_pool_data = []
        for _, row in dark_pool_df.iterrows():
            dark_pool_data.append({
                "symbol": symbol,
                "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
                "darkPoolVolume": int(row.get("dark_pool_volume", 0)),
                "totalVolume": int(row.get("total_volume", 0)),
                "darkPoolPercentage": round(float(row.get("dark_pool_percentage", 0)) * 100, 2),
                "largeBlockActivity": round(float(row.get("large_block_activity", 0)) * 100, 2),
                "signal": "bullish" if row.get("dark_pool_signal", 0) > 0 else (
                    "bearish" if row.get("dark_pool_signal", 0) < 0 else "neutral"
                ),
            })

        return create_response(data=dark_pool_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dark pool activity for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/equity-flow/{symbol}", response_model=ApiResponse, tags=["Analytics"])
async def get_equity_flow(symbol: str, lookback_days: int = 20):
    """
    Get money flow analysis for a specific equity.

    Returns money flow score, institutional sentiment, and smart money activity.

    Args:
        symbol: Stock ticker symbol
        lookback_days: Number of days to analyze (default: 20)
    """
    try:
        symbol = symbol.upper()

        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        flow_data = app_state.money_flow_analyzer.analyze_equity_flow(
            symbol, lookback_days
        )

        return create_response(data={
            "symbol": flow_data.get("symbol", symbol),
            "moneyFlowScore": round(float(flow_data.get("money_flow_score", 0)), 3),
            "institutionalSentiment": round(float(flow_data.get("institutional_sentiment", 0)), 3),
            "smartMoneyActivity": round(float(flow_data.get("smart_money_activity", 0)), 3),
            "shortPressure": round(float(flow_data.get("short_pressure", 0)), 3),
            "accumulationDistribution": round(float(flow_data.get("accumulation_distribution", 0)), 3),
            "confidence": round(float(flow_data.get("confidence", 0)), 3),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching equity flow for {symbol}: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Research Endpoints
# =============================================================================


@app.get("/api/research/{symbol}", response_model=ApiResponse, tags=["Research"])
async def get_research(symbol: str):
    """
    Get comprehensive research analysis for a symbol.

    Returns valuation, earnings analysis, and fundamental metrics.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.research_analyzer:
            raise HTTPException(
                status_code=503, detail="Research analyzer not initialized"
            )

        report = await app_state.research_analyzer.generate_report(symbol)

        return create_response(data=report.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating research for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/valuation/{symbol}", response_model=ApiResponse, tags=["Research"])
async def get_valuation(symbol: str, include_dcf: bool = True):
    """
    Get valuation analysis for a symbol.

    Returns valuation multiples and optional DCF analysis.

    Args:
        symbol: Stock ticker symbol
        include_dcf: Whether to include DCF analysis
    """
    try:
        symbol = symbol.upper()

        if not app_state.research_analyzer:
            raise HTTPException(
                status_code=503, detail="Research analyzer not initialized"
            )

        valuation = await app_state.research_analyzer.get_valuation(symbol, include_dcf)

        return create_response(data=valuation)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching valuation for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/earnings/{symbol}", response_model=ApiResponse, tags=["Research"])
async def get_earnings(symbol: str, quarters: int = 12):
    """
    Get earnings analysis for a symbol.

    Returns earnings history, trends, and surprise metrics.

    Args:
        symbol: Stock ticker symbol
        quarters: Number of quarters to analyze
    """
    try:
        symbol = symbol.upper()

        if not app_state.research_analyzer:
            raise HTTPException(
                status_code=503, detail="Research analyzer not initialized"
            )

        earnings = await app_state.research_analyzer.analyze_earnings(symbol, quarters)

        return create_response(data=earnings.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing earnings for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/peers/{symbol}", response_model=ApiResponse, tags=["Research"])
async def get_peer_comparison(symbol: str):
    """
    Get peer comparison analysis for a symbol.

    Returns relative valuation vs peer group.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.research_analyzer:
            raise HTTPException(
                status_code=503, detail="Research analyzer not initialized"
            )

        comparison = await app_state.research_analyzer.get_peer_comparison(symbol)

        return create_response(data=comparison)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching peer comparison for {symbol}: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Commodities Endpoints
# =============================================================================


@app.get("/api/commodities", response_model=ApiResponse, tags=["Commodities"])
async def get_commodities_overview():
    """
    Get commodity market overview.

    Returns prices and trends across all commodity categories.
    """
    try:
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


@app.get("/api/commodities/{symbol}", response_model=ApiResponse, tags=["Commodities"])
async def get_commodity_detail(symbol: str):
    """
    Get detailed analysis for a specific commodity.

    Args:
        symbol: Commodity symbol (e.g., CL, GC, NG)
    """
    try:
        symbol = symbol.upper()

        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        summary = await app_state.commodities_analyzer.get_summary(symbol)

        return create_response(data=summary.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching commodity data for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/commodities/{symbol}/macro", response_model=ApiResponse, tags=["Commodities"])
async def get_commodity_macro_linkage(symbol: str):
    """
    Get macro-commodity linkage analysis.

    Args:
        symbol: Commodity symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        linkage = await app_state.commodities_analyzer.analyze_macro_linkage(symbol)

        return create_response(data=linkage)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing macro linkage for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/commodities/correlations", response_model=ApiResponse, tags=["Commodities"])
async def get_commodity_correlations(commodities: Optional[str] = None):
    """
    Get correlation matrix for commodities.

    Args:
        commodities: Comma-separated list of symbols (optional)
    """
    try:
        if not app_state.commodities_analyzer:
            raise HTTPException(
                status_code=503, detail="Commodities analyzer not initialized"
            )

        symbols = commodities.split(",") if commodities else None
        corr_matrix = await app_state.commodities_analyzer.get_correlations(symbols)

        if corr_matrix.empty:
            return create_response(data={})

        # Convert to JSON-serializable format
        return create_response(data={
            "commodities": list(corr_matrix.columns),
            "matrix": corr_matrix.round(3).to_dict(),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching commodity correlations: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Notes Endpoints
# =============================================================================


class CreateThesisRequest(BaseModel):
    """Request to create an investment thesis."""

    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(default="", description="Company name")
    sector: str = Field(default="", description="Sector/industry")
    conviction: str = Field(default="medium", description="Conviction level")
    content: Optional[str] = Field(default=None, description="Custom content")


class CreateTradeRequest(BaseModel):
    """Request to create a trade journal entry."""

    symbol: str = Field(..., description="Stock symbol")
    direction: str = Field(default="long", description="Trade direction")
    entry_price: float = Field(default=0.0, ge=0, description="Entry price")
    shares: float = Field(default=0.0, ge=0, description="Number of shares")
    entry_date: Optional[str] = Field(default=None, description="Entry date (ISO format)")
    content: Optional[str] = Field(default=None, description="Custom content")


class CloseTradeRequest(BaseModel):
    """Request to close a trade."""

    exit_price: float = Field(..., ge=0, description="Exit price")
    exit_date: Optional[str] = Field(default=None, description="Exit date (ISO format)")
    exit_reason: str = Field(default="", description="Reason for exit")
    lessons: str = Field(default="", description="Lessons learned")
    grade: str = Field(default="", description="Self-assessment grade")


class UpdateNoteRequest(BaseModel):
    """Request to update a note."""

    content: str = Field(..., description="New content")


@app.get("/api/notes", response_model=ApiResponse, tags=["Notes"])
async def list_notes(
    note_type: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 100,
):
    """
    List notes with optional filters.

    Args:
        note_type: Filter by type (thesis, trade, company, etc.)
        tags: Comma-separated list of tags
        limit: Maximum results
    """
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        tag_list = tags.split(",") if tags else None
        notes = app_state.note_manager.list_notes(
            note_type=note_type, tags=tag_list, limit=limit
        )

        return create_response(data=[n.to_dict() for n in notes])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing notes: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/notes/search", response_model=ApiResponse, tags=["Notes"])
async def search_notes(query: str, limit: int = 50):
    """
    Full-text search across notes.

    Args:
        query: Search query
        limit: Maximum results
    """
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        results = app_state.note_manager.search(query, limit)
        return create_response(data=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/notes/graph", response_model=ApiResponse, tags=["Notes"])
async def get_notes_graph():
    """Get the note graph for visualization."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        graph = app_state.note_manager.get_graph()
        return create_response(data=graph)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting notes graph: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/notes/{name}", response_model=ApiResponse, tags=["Notes"])
async def get_note(name: str):
    """Get a specific note by name."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        note = app_state.note_manager.get_note(name)
        if not note:
            raise HTTPException(status_code=404, detail=f"Note not found: {name}")

        return create_response(data={
            **note.to_dict(),
            "content": note.content,
            "frontmatter": note.frontmatter.to_yaml(),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting note {name}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/notes/{name}/backlinks", response_model=ApiResponse, tags=["Notes"])
async def get_note_backlinks(name: str):
    """Get all notes that link to the given note."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        backlinks = app_state.note_manager.get_backlinks(name)
        return create_response(data=[n.to_dict() for n in backlinks])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backlinks for {name}: {e}")
        return create_response(error=str(e), success=False)


@app.put("/api/notes/{name}", response_model=ApiResponse, tags=["Notes"])
async def update_note(name: str, request: UpdateNoteRequest):
    """Update a note's content."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        note = app_state.note_manager.update_note(name, request.content)
        return create_response(data=note.to_dict())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating note {name}: {e}")
        return create_response(error=str(e), success=False)


@app.delete("/api/notes/{name}", response_model=ApiResponse, tags=["Notes"])
async def delete_note(name: str):
    """Delete a note."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        deleted = app_state.note_manager.delete_note(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Note not found: {name}")

        return create_response(data={"deleted": name})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note {name}: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Thesis Endpoints
# =============================================================================


@app.get("/api/theses", response_model=ApiResponse, tags=["Notes"])
async def list_theses(status: Optional[str] = None, symbol: Optional[str] = None):
    """
    List investment theses.

    Args:
        status: Filter by status (research, watchlist, active, closed, invalidated)
        symbol: Filter by symbol
    """
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        theses = app_state.note_manager.get_theses(status=status, symbol=symbol)
        return create_response(data=[t.to_dict() for t in theses])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing theses: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/theses", response_model=ApiResponse, tags=["Notes"])
async def create_thesis(request: CreateThesisRequest):
    """Create a new investment thesis."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        thesis = app_state.note_manager.create_thesis(
            symbol=request.symbol,
            company_name=request.company_name,
            sector=request.sector,
            conviction=request.conviction,
            content=request.content,
        )

        return create_response(data=thesis.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating thesis: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Trade Journal Endpoints
# =============================================================================


@app.get("/api/trades", response_model=ApiResponse, tags=["Notes"])
async def list_trades(status: Optional[str] = None, symbol: Optional[str] = None):
    """
    List trade journal entries.

    Args:
        status: Filter by status (open, closed, partial)
        symbol: Filter by symbol
    """
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        trades = app_state.note_manager.get_trades(status=status, symbol=symbol)
        return create_response(data=[t.to_dict() for t in trades])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing trades: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/trades", response_model=ApiResponse, tags=["Notes"])
async def create_trade(request: CreateTradeRequest):
    """Create a new trade journal entry."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        trade = app_state.note_manager.create_trade(
            symbol=request.symbol,
            direction=request.direction,
            entry_price=request.entry_price,
            shares=request.shares,
            entry_date=request.entry_date,
            content=request.content,
        )

        return create_response(data=trade.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating trade: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/trades/{name}/close", response_model=ApiResponse, tags=["Notes"])
async def close_trade(name: str, request: CloseTradeRequest):
    """Close an open trade."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        trade = app_state.note_manager.close_trade(
            trade_name=name,
            exit_price=request.exit_price,
            exit_date=request.exit_date,
            exit_reason=request.exit_reason,
            lessons=request.lessons,
            grade=request.grade,
        )

        return create_response(data=trade.to_dict())

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error closing trade {name}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/trades/stats", response_model=ApiResponse, tags=["Notes"])
async def get_trade_stats():
    """Get aggregate trade statistics."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        stats = app_state.note_manager.get_trade_stats()
        return create_response(data=stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trade stats: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Run server (for development)
# =============================================================================


def run_dev_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the development server."""
    import uvicorn

    uvicorn.run(
        "stanley.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    run_dev_server()
