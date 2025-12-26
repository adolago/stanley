"""
Stanley REST API

FastAPI-based REST API for the Stanley institutional investment analysis platform.
Provides endpoints for the stanley-ui (Tauri/React) application.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..accounting import (
    AccountingAnalyzer,
    EarningsQualityAnalyzer,
    RedFlagScorer,
    AnomalyAggregator,
    EdgarAdapter,
    FinancialStatements,
)
from ..analytics.institutional import InstitutionalAnalyzer
from ..analytics.money_flow import MoneyFlowAnalyzer
from ..commodities import CommoditiesAnalyzer
from ..data.data_manager import DataManager
from ..etf import ETFAnalyzer
from ..notes import NoteManager
from ..options import OptionsAnalyzer
from ..portfolio import PortfolioAnalyzer
from ..research import ResearchAnalyzer
from ..signals import SignalGenerator, SignalBacktester, PerformanceTracker

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
        self.options_analyzer: Optional[OptionsAnalyzer] = None
        self.etf_analyzer: Optional[ETFAnalyzer] = None
        self.note_manager: Optional[NoteManager] = None
        self.accounting_analyzer: Optional[AccountingAnalyzer] = None
        self.earnings_quality_analyzer: Optional[EarningsQualityAnalyzer] = None
        self.red_flag_scorer: Optional[RedFlagScorer] = None
        self.anomaly_aggregator: Optional[AnomalyAggregator] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.signal_backtester: Optional[SignalBacktester] = None
        self.performance_tracker: Optional[PerformanceTracker] = None


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
    app_state.options_analyzer = OptionsAnalyzer(app_state.data_manager)
    app_state.etf_analyzer = ETFAnalyzer(app_state.data_manager)

    # Initialize accounting analyzers
    try:
        # SEC requires identification for API access
        # Can be set via SEC_IDENTITY env var or defaults to config value
        sec_identity = os.environ.get("SEC_IDENTITY", "stanley-research@example.com")

        # Create EdgarAdapter with identity
        edgar_adapter = EdgarAdapter(identity=sec_identity)
        edgar_adapter.initialize()

        # Create FinancialStatements with the same adapter
        financial_statements = FinancialStatements(edgar_adapter=edgar_adapter)

        # Initialize all accounting analyzers with proper identity/statements
        app_state.accounting_analyzer = AccountingAnalyzer(edgar_identity=sec_identity)
        app_state.earnings_quality_analyzer = EarningsQualityAnalyzer(
            financial_statements=financial_statements
        )
        app_state.red_flag_scorer = RedFlagScorer(edgar_adapter=edgar_adapter)
        app_state.anomaly_aggregator = AnomalyAggregator(edgar_adapter=edgar_adapter)
        logger.info(
            f"Accounting analyzers initialized with SEC identity: {sec_identity}"
        )
    except Exception as e:
        logger.warning(f"Accounting analyzers initialization failed: {e}")

    # Initialize note manager
    try:
        app_state.note_manager = NoteManager()
        logger.info("Note manager initialized")
    except Exception as e:
        logger.warning(f"Note manager initialization failed: {e}")

    # Initialize signal generator
    try:
        app_state.signal_generator = SignalGenerator(
            money_flow_analyzer=app_state.money_flow_analyzer,
            institutional_analyzer=app_state.institutional_analyzer,
            research_analyzer=app_state.research_analyzer,
            portfolio_analyzer=app_state.portfolio_analyzer,
            data_manager=app_state.data_manager,
        )
        app_state.signal_backtester = SignalBacktester(app_state.data_manager)
        app_state.performance_tracker = PerformanceTracker(app_state.data_manager)
        logger.info("Signal generator initialized")
    except Exception as e:
        logger.warning(f"Signal generator initialization failed: {e}")

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


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def create_response(
    data: Any = None, error: Optional[str] = None, success: bool = True
) -> ApiResponse:
    """Create a standardized API response."""
    # Convert numpy types to native Python types
    converted_data = _convert_numpy_types(data) if data is not None else None

    return ApiResponse(
        success=success and error is None,
        data=converted_data,
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
        "options_analyzer": False,
        "etf_analyzer": False,
        "accounting_analyzer": False,
        "earnings_quality_analyzer": False,
        "red_flag_scorer": False,
        "anomaly_aggregator": False,
        "signal_generator": False,
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
            components["research_analyzer"] = app_state.research_analyzer.health_check()
        if app_state.commodities_analyzer:
            components["commodities_analyzer"] = (
                app_state.commodities_analyzer.health_check()
            )
        if app_state.options_analyzer:
            components["options_analyzer"] = app_state.options_analyzer.health_check()
        if app_state.etf_analyzer:
            components["etf_analyzer"] = app_state.etf_analyzer.health_check()
        if app_state.accounting_analyzer:
            components["accounting_analyzer"] = (
                app_state.accounting_analyzer.health_check()
            )
        if app_state.earnings_quality_analyzer:
            components["earnings_quality_analyzer"] = True
        if app_state.red_flag_scorer:
            components["red_flag_scorer"] = True
        if app_state.anomaly_aggregator:
            components["anomaly_aggregator"] = True
        if app_state.signal_generator:
            components["signal_generator"] = app_state.signal_generator.health_check()
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
            stock_data = pd.DataFrame(
                {
                    "date": pd.date_range(start=start_date, end=end_date, freq="D"),
                    "close": [150.0 + np.random.uniform(-5, 5) for _ in range(6)],
                    "volume": [
                        np.random.randint(10000000, 100000000) for _ in range(6)
                    ],
                }
            )

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
            raise HTTPException(
                status_code=503, detail="Portfolio analyzer not initialized"
            )

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
            dark_pool_data.append(
                {
                    "symbol": symbol,
                    "date": row["date"].isoformat() if pd.notna(row["date"]) else None,
                    "darkPoolVolume": int(row.get("dark_pool_volume", 0)),
                    "totalVolume": int(row.get("total_volume", 0)),
                    "darkPoolPercentage": round(
                        float(row.get("dark_pool_percentage", 0)) * 100, 2
                    ),
                    "largeBlockActivity": round(
                        float(row.get("large_block_activity", 0)) * 100, 2
                    ),
                    "signal": (
                        "bullish"
                        if row.get("dark_pool_signal", 0) > 0
                        else (
                            "bearish"
                            if row.get("dark_pool_signal", 0) < 0
                            else "neutral"
                        )
                    ),
                }
            )

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

        return create_response(
            data={
                "symbol": flow_data.get("symbol", symbol),
                "moneyFlowScore": round(float(flow_data.get("money_flow_score", 0)), 3),
                "institutionalSentiment": round(
                    float(flow_data.get("institutional_sentiment", 0)), 3
                ),
                "smartMoneyActivity": round(
                    float(flow_data.get("smart_money_activity", 0)), 3
                ),
                "shortPressure": round(float(flow_data.get("short_pressure", 0)), 3),
                "accumulationDistribution": round(
                    float(flow_data.get("accumulation_distribution", 0)), 3
                ),
                "confidence": round(float(flow_data.get("confidence", 0)), 3),
            }
        )

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


@app.get(
    "/api/commodities/{symbol}/macro", response_model=ApiResponse, tags=["Commodities"]
)
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


@app.get(
    "/api/commodities/correlations", response_model=ApiResponse, tags=["Commodities"]
)
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


# =============================================================================
# Options Flow Endpoints
# =============================================================================


class OptionsFlowResponse(BaseModel):
    """Options flow analysis response."""

    symbol: str
    totalCallVolume: int
    totalPutVolume: int
    totalCallPremium: float
    totalPutPremium: float
    putCallRatio: float
    premiumPutCallRatio: float
    netPremiumFlow: float
    unusualActivityCount: int
    smartMoneyTrades: int
    sentiment: str
    confidence: float
    timestamp: str


class GammaExposureResponse(BaseModel):
    """Gamma exposure response."""

    symbol: str
    totalGex: float
    callGex: float
    putGex: float
    netGex: float
    flipPoint: Optional[float]
    maxGammaStrike: float
    timestamp: str


class UnusualActivityResponse(BaseModel):
    """Unusual options activity response."""

    symbol: str
    strike: float
    expiration: str
    optionType: str
    volume: int
    openInterest: int
    volOiRatio: float
    premium: float
    impliedVolatility: float
    tradeType: str
    sentiment: str


@app.get("/api/options/{symbol}/flow", response_model=ApiResponse, tags=["Options"])
async def get_options_flow(symbol: str, lookback_days: int = 5):
    """
    Get comprehensive options flow analysis for a symbol.

    Returns volume, premium, put/call ratios, gamma exposure, and sentiment.

    Args:
        symbol: Stock ticker symbol
        lookback_days: Number of days to analyze
    """
    try:
        symbol = symbol.upper()

        if not app_state.options_analyzer:
            raise HTTPException(
                status_code=503, detail="Options analyzer not initialized"
            )

        flow_data = await app_state.options_analyzer.get_options_flow(
            symbol, lookback_days
        )

        response = OptionsFlowResponse(
            symbol=flow_data["symbol"],
            totalCallVolume=flow_data["total_call_volume"],
            totalPutVolume=flow_data["total_put_volume"],
            totalCallPremium=flow_data["total_call_premium"],
            totalPutPremium=flow_data["total_put_premium"],
            putCallRatio=flow_data["put_call_ratio"],
            premiumPutCallRatio=flow_data["premium_put_call_ratio"],
            netPremiumFlow=flow_data["net_premium_flow"],
            unusualActivityCount=flow_data["unusual_activity_count"],
            smartMoneyTrades=flow_data["smart_money_trades"],
            sentiment=flow_data["sentiment"],
            confidence=flow_data["confidence"],
            timestamp=flow_data["timestamp"],
        )

        return create_response(data=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching options flow for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/options/{symbol}/gamma", response_model=ApiResponse, tags=["Options"])
async def get_gamma_exposure(symbol: str):
    """
    Get gamma exposure (GEX) analysis for a symbol.

    Returns aggregate gamma exposure, flip point, and dealer positioning.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.options_analyzer:
            raise HTTPException(
                status_code=503, detail="Options analyzer not initialized"
            )

        gex_data = await app_state.options_analyzer.calculate_gamma_exposure(symbol)

        response = GammaExposureResponse(
            symbol=gex_data["symbol"],
            totalGex=gex_data["total_gex"],
            callGex=gex_data["call_gex"],
            putGex=gex_data["put_gex"],
            netGex=gex_data["net_gex"],
            flipPoint=gex_data["flip_point"],
            maxGammaStrike=gex_data["max_gamma_strike"],
            timestamp=gex_data["timestamp"],
        )

        return create_response(data=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating gamma exposure for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/options/{symbol}/unusual", response_model=ApiResponse, tags=["Options"])
async def get_unusual_options_activity(
    symbol: str, volume_threshold: float = 2.0, min_premium: float = 50000
):
    """
    Get unusual options activity for a symbol.

    Detects options with volume/OI ratio above threshold and significant premium.

    Args:
        symbol: Stock ticker symbol
        volume_threshold: Minimum volume/OI ratio (default: 2.0)
        min_premium: Minimum premium in dollars (default: $50k)
    """
    try:
        symbol = symbol.upper()

        if not app_state.options_analyzer:
            raise HTTPException(
                status_code=503, detail="Options analyzer not initialized"
            )

        unusual_df = await app_state.options_analyzer.detect_unusual_activity(
            symbol, volume_threshold, min_premium
        )

        if unusual_df.empty:
            return create_response(data=[])

        unusual_list = []
        for _, row in unusual_df.iterrows():
            unusual_list.append(
                UnusualActivityResponse(
                    symbol=symbol,
                    strike=float(row["strike"]),
                    expiration=str(row["expiration"]),
                    optionType=str(row["option_type"]),
                    volume=int(row["volume"]),
                    openInterest=int(row["open_interest"]),
                    volOiRatio=round(float(row["vol_oi_ratio"]), 2),
                    premium=round(float(row["premium"]), 2),
                    impliedVolatility=round(float(row["implied_volatility"]), 4),
                    tradeType=str(row["trade_type"]),
                    sentiment=str(row["sentiment"]),
                ).model_dump()
            )

        return create_response(data=unusual_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting unusual activity for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/options/{symbol}/put-call", response_model=ApiResponse, tags=["Options"])
async def get_put_call_analysis(symbol: str):
    """
    Get put/call flow analysis for a symbol.

    Returns volume and premium-weighted ratios with strike distribution.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.options_analyzer:
            raise HTTPException(
                status_code=503, detail="Options analyzer not initialized"
            )

        pcr_data = await app_state.options_analyzer.analyze_put_call_flow(symbol)

        return create_response(
            data={
                "symbol": pcr_data["symbol"],
                "putCallRatio": pcr_data["put_call_ratio"],
                "premiumPutCallRatio": pcr_data["premium_put_call_ratio"],
                "oiPutCallRatio": pcr_data["oi_put_call_ratio"],
                "totalCallVolume": pcr_data["total_call_volume"],
                "totalPutVolume": pcr_data["total_put_volume"],
                "totalCallPremium": pcr_data["total_call_premium"],
                "totalPutPremium": pcr_data["total_put_premium"],
                "callOpenInterest": pcr_data["call_open_interest"],
                "putOpenInterest": pcr_data["put_open_interest"],
                "itmCallVolume": pcr_data["itm_call_volume"],
                "otmCallVolume": pcr_data["otm_call_volume"],
                "itmPutVolume": pcr_data["itm_put_volume"],
                "otmPutVolume": pcr_data["otm_put_volume"],
                "weightedCallStrike": pcr_data["weighted_call_strike"],
                "weightedPutStrike": pcr_data["weighted_put_strike"],
                "underlyingPrice": pcr_data["underlying_price"],
                "sentiment": pcr_data["sentiment"],
                "timestamp": pcr_data["timestamp"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing put/call flow for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/options/{symbol}/smart-money", response_model=ApiResponse, tags=["Options"]
)
async def get_smart_money_trades(symbol: str, min_premium: float = 100000):
    """
    Get smart money options activity for a symbol.

    Tracks block trades, sweep orders, and other institutional activity.

    Args:
        symbol: Stock ticker symbol
        min_premium: Minimum premium threshold (default: $100k)
    """
    try:
        symbol = symbol.upper()

        if not app_state.options_analyzer:
            raise HTTPException(
                status_code=503, detail="Options analyzer not initialized"
            )

        smart_money_df = await app_state.options_analyzer.track_smart_money(
            symbol, min_premium
        )

        if smart_money_df.empty:
            return create_response(data=[])

        trades_list = []
        for _, row in smart_money_df.iterrows():
            trades_list.append(
                {
                    "symbol": symbol,
                    "strike": float(row["strike"]),
                    "expiration": str(row["expiration"]),
                    "optionType": str(row["option_type"]),
                    "premium": round(float(row["premium"]), 2),
                    "volume": int(row["volume"]),
                    "tradeType": str(row["trade_type"]),
                    "side": str(row["side"]),
                    "sentiment": str(row["sentiment"]),
                    "timestamp": str(row["timestamp"]),
                }
            )

        return create_response(data=trades_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking smart money for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/options/{symbol}/max-pain", response_model=ApiResponse, tags=["Options"])
async def get_max_pain(symbol: str, expiration: Optional[str] = None):
    """
    Get max pain analysis for a symbol.

    Returns the strike price where option holders would experience maximum loss.

    Args:
        symbol: Stock ticker symbol
        expiration: Optional expiration date (YYYY-MM-DD)
    """
    try:
        symbol = symbol.upper()

        if not app_state.options_analyzer:
            raise HTTPException(
                status_code=503, detail="Options analyzer not initialized"
            )

        analysis = await app_state.options_analyzer.analyze_expiration_flow(
            symbol, expiration
        )

        return create_response(
            data={
                "symbol": symbol,
                "expiration": analysis["expiration"],
                "maxPain": analysis["max_pain"],
                "totalCallOI": analysis["total_call_oi"],
                "totalPutOI": analysis["total_put_oi"],
                "totalCallVolume": analysis["total_call_volume"],
                "totalPutVolume": analysis["total_put_volume"],
                "gammaConcentration": analysis["gamma_concentration"],
                "pinRisk": analysis["pin_risk"],
                "daysToExpiry": analysis["days_to_expiry"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating max pain for {symbol}: {e}")
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
    entry_date: Optional[str] = Field(
        default=None, description="Entry date (ISO format)"
    )
    content: Optional[str] = Field(default=None, description="Custom content")


class CloseTradeRequest(BaseModel):
    """Request to close a trade."""

    exit_price: float = Field(..., ge=0, description="Exit price")
    exit_date: Optional[str] = Field(default=None, description="Exit date (ISO format)")
    exit_reason: str = Field(default="", description="Reason for exit")
    lessons: str = Field(default="", description="Lessons learned")
    grade: str = Field(default="", description="Self-assessment grade")


class CreateEventRequest(BaseModel):
    """Request to create an event note."""

    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(default="", description="Company name")
    event_type: str = Field(
        default="conference",
        description="Event type (earnings_call, investor_day, conference, etc.)",
    )
    event_date: Optional[str] = Field(
        default=None, description="Event date (ISO format)"
    )
    host: str = Field(default="", description="Bank/broker hosting the event")
    participants: List[str] = Field(default=[], description="List of participant names")
    content: Optional[str] = Field(default=None, description="Custom content")


class CreatePersonRequest(BaseModel):
    """Request to create a person/executive profile."""

    full_name: str = Field(..., description="Person's full name")
    current_role: str = Field(default="", description="Current role (CEO, CFO, etc.)")
    current_company: str = Field(default="", description="Current company name")
    linkedin_url: str = Field(default="", description="LinkedIn profile URL")
    content: Optional[str] = Field(default=None, description="Custom content")


class CreateSectorRequest(BaseModel):
    """Request to create a sector overview."""

    sector_name: str = Field(..., description="Sector name")
    sub_sectors: List[str] = Field(default=[], description="List of sub-sectors")
    companies: List[str] = Field(default=[], description="List of companies covered")
    content: Optional[str] = Field(default=None, description="Custom content")


# =============================================================================
# Bloomberg-Beating Analytics - Request Models
# =============================================================================


class BatchSmartMoneyRequest(BaseModel):
    """Request for batch smart money analysis."""

    symbols: List[str] = Field(
        ..., description="List of stock symbols to analyze", min_length=1, max_length=50
    )


# =============================================================================
# Bloomberg-Beating Analytics - Response Models
# =============================================================================


class WhaleHolder(BaseModel):
    """Whale holder data."""

    managerName: str
    managerCik: str
    sharesHeld: int
    valueHeld: float
    ownershipPercentage: float
    isNewPosition: bool = False
    positionChangePercent: Optional[float] = None
    lastFilingDate: Optional[str] = None


class WhaleMovement(BaseModel):
    """Whale position movement."""

    managerName: str
    managerCik: str
    changeType: str  # "increase", "decrease", "new", "exit"
    sharesBefore: int
    sharesAfter: int
    sharesChange: int
    percentChange: float
    filingDate: str
    estimatedValue: float


class WhaleAlert(BaseModel):
    """Whale activity alert."""

    symbol: str
    alertType: (
        str  # "large_accumulation", "large_distribution", "new_whale", "whale_exit"
    )
    managerName: str
    significance: str  # "high", "medium", "low"
    sharesChanged: int
    valueChanged: float
    percentOfFloat: float
    timestamp: str
    description: str


class OptionsFlowData(BaseModel):
    """Options flow analysis data."""

    symbol: str
    callVolume: int
    putVolume: int
    callPutRatio: float
    totalPremium: float
    netCallPremium: float
    netPutPremium: float
    dominantFlow: str  # "bullish", "bearish", "neutral"
    largeOrderCount: int
    timestamp: str


class UnusualOptionsActivity(BaseModel):
    """Unusual options activity."""

    symbol: str
    optionType: str  # "call" or "put"
    strike: float
    expiry: str
    volume: int
    openInterest: int
    volumeOiRatio: float
    impliedVolatility: float
    premium: float
    unusualScore: float
    sentiment: str  # "bullish", "bearish"
    timestamp: str


class OptionsSentiment(BaseModel):
    """Options-based sentiment analysis."""

    symbol: str
    overallSentiment: str  # "bullish", "bearish", "neutral"
    sentimentScore: float  # -1.0 to 1.0
    callPutRatio: float
    ivRank: float
    ivPercentile: float
    skew: float  # Put/Call IV skew
    termStructure: str  # "contango", "backwardation", "flat"
    smartMoneyBias: str  # "bullish", "bearish", "neutral"
    confidence: float


class SectorRotationData(BaseModel):
    """Sector rotation analysis."""

    sector: str
    etfSymbol: str
    relativeStrength: float
    momentumScore: float
    flowScore: float
    rotationPhase: str  # "leading", "weakening", "lagging", "improving"
    recommendation: str  # "overweight", "neutral", "underweight"
    oneMonthReturn: float
    threeMonthReturn: float


class RotationSignal(BaseModel):
    """Sector rotation signal."""

    signalType: str  # "rotate_into", "rotate_out", "maintain"
    fromSector: Optional[str] = None
    toSector: Optional[str] = None
    strength: str  # "strong", "moderate", "weak"
    confidence: float
    rationale: str
    timestamp: str


class RiskRegime(BaseModel):
    """Risk-on/Risk-off regime status."""

    regime: str  # "risk_on", "risk_off", "transitioning"
    regimeScore: float  # -1.0 (full risk-off) to 1.0 (full risk-on)
    indicators: Dict[str, float]
    leadingSectors: List[str]
    laggingSectors: List[str]
    recommendation: str
    confidence: float
    lastChange: Optional[str] = None


class SmartMoneyIndex(BaseModel):
    """Smart money index for a symbol."""

    symbol: str
    smartMoneyIndex: float  # -1.0 to 1.0
    institutionalSentiment: float
    darkPoolSignal: float
    blockTradeSignal: float
    flowImbalance: float
    confidence: float
    trend: str  # "accumulating", "distributing", "neutral"
    timestamp: str


class SmartMoneyComponents(BaseModel):
    """Smart money index component breakdown."""

    symbol: str
    institutionalBuyers: int
    institutionalSellers: int
    netInstitutionalFlow: float
    darkPoolPercentage: float
    darkPoolBias: str  # "bullish", "bearish", "neutral"
    blockTradeCount: int
    blockTradeBias: str
    largeOrderImbalance: float
    retailVsInstitutional: float  # Positive = institutional dominance
    components: Dict[str, float]


class InstitutionalAlert(BaseModel):
    """Institutional position change alert."""

    symbol: str
    alertType: (
        str  # "large_increase", "large_decrease", "new_position", "exit_position"
    )
    managerName: str
    managerType: str  # "hedge_fund", "mutual_fund", "pension", "other"
    previousShares: int
    currentShares: int
    changePercent: float
    estimatedValue: float
    filingDate: str
    significance: str  # "high", "medium", "low"


class ConvictionPick(BaseModel):
    """High conviction institutional position."""

    symbol: str
    companyName: str
    topHolders: List[str]
    averagePortfolioWeight: float
    holderCount: int
    recentIncreases: int
    aggregateConviction: float  # 0-100
    institutionalOwnership: float
    avgHoldingPeriod: str  # "short", "medium", "long"
    trend: str  # "increasing", "stable", "decreasing"


class NewPosition(BaseModel):
    """Recently initiated institutional position."""

    symbol: str
    companyName: str
    managerName: str
    managerCik: str
    sharesInitiated: int
    estimatedValue: float
    portfolioWeight: float
    filingDate: str
    managerAum: Optional[float] = None
    managerStyle: Optional[str] = None  # "value", "growth", "quant", etc.


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

        return create_response(
            data={
                **note.to_dict(),
                "content": note.content,
                "frontmatter": note.frontmatter.to_yaml(),
            }
        )

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
# Event Endpoints
# =============================================================================


@app.get("/api/events", response_model=ApiResponse, tags=["Notes"])
async def list_events(
    event_type: Optional[str] = None,
    symbol: Optional[str] = None,
    company: Optional[str] = None,
):
    """
    List event notes (conference calls, investor days, etc.).

    Args:
        event_type: Filter by type (earnings_call, conference, investor_day, etc.)
        symbol: Filter by stock symbol
        company: Filter by company name
    """
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        events = app_state.note_manager.get_events(
            event_type=event_type, symbol=symbol, company=company
        )
        return create_response(data=[e.to_dict() for e in events])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing events: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/events", response_model=ApiResponse, tags=["Notes"])
async def create_event(request: CreateEventRequest):
    """Create a new event note."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        event = app_state.note_manager.create_event(
            symbol=request.symbol,
            company_name=request.company_name,
            event_type=request.event_type,
            event_date=request.event_date,
            host=request.host,
            participants=request.participants,
            content=request.content,
        )

        return create_response(data=event.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# People Endpoints
# =============================================================================


@app.get("/api/people", response_model=ApiResponse, tags=["Notes"])
async def list_people(
    company: Optional[str] = None,
    role: Optional[str] = None,
):
    """
    List person/executive profile notes.

    Args:
        company: Filter by company name
        role: Filter by role (CEO, CFO, etc.)
    """
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        people = app_state.note_manager.get_people(company=company, role=role)
        return create_response(data=[p.to_dict() for p in people])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing people: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/people", response_model=ApiResponse, tags=["Notes"])
async def create_person(request: CreatePersonRequest):
    """Create a new person/executive profile."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        person = app_state.note_manager.create_person(
            full_name=request.full_name,
            current_role=request.current_role,
            current_company=request.current_company,
            linkedin_url=request.linkedin_url,
            content=request.content,
        )

        return create_response(data=person.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating person: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Sector Endpoints
# =============================================================================


@app.get("/api/sectors", response_model=ApiResponse, tags=["Notes"])
async def list_sectors():
    """List all sector overview notes."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        sectors = app_state.note_manager.get_sectors()
        return create_response(data=[s.to_dict() for s in sectors])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing sectors: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/sectors", response_model=ApiResponse, tags=["Notes"])
async def create_sector(request: CreateSectorRequest):
    """Create a new sector overview."""
    try:
        if not app_state.note_manager:
            raise HTTPException(status_code=503, detail="Note manager not initialized")

        sector = app_state.note_manager.create_sector(
            sector_name=request.sector_name,
            sub_sectors=request.sub_sectors,
            companies=request.companies,
            content=request.content,
        )

        return create_response(data=sector.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating sector: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# ETF Flow Analytics Endpoints
# =============================================================================


class ETFFlowRequest(BaseModel):
    """Request body for ETF flow analysis."""

    symbols: Optional[List[str]] = Field(
        default=None, description="List of ETF symbols (all if not specified)"
    )
    lookback_days: int = Field(
        default=90, ge=1, le=365, description="Number of days to analyze"
    )


# =============================================================================
# Enhanced Institutional Analytics Endpoints (Bloomberg-Beating Features)
# =============================================================================


@app.get(
    "/api/institutional/{symbol}/changes",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_13f_changes(symbol: str, conviction_threshold: float = 0.05):
    """
    Get enhanced 13F change detection with conviction scoring.

    Compares current vs previous quarter to detect new/closed positions,
    significant changes, and calculates conviction scores.

    Args:
        symbol: Stock ticker symbol
        conviction_threshold: Minimum change to consider significant (default: 5%)
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        changes_df = app_state.institutional_analyzer.detect_13f_changes(
            symbol, conviction_threshold
        )

        if changes_df.empty:
            return create_response(data=[])

        changes_list = []
        for _, row in changes_df.iterrows():
            changes_list.append(
                {
                    "managerName": str(row.get("manager_name", "")),
                    "changeType": str(row.get("change_type", "")),
                    "sharesCurrent": int(row.get("shares_current", 0)),
                    "sharesPrevious": int(row.get("shares_previous", 0)),
                    "changeMagnitude": int(row.get("change_magnitude", 0)),
                    "changePercentage": round(
                        float(row.get("change_percentage", 0)), 4
                    ),
                    "convictionScore": round(float(row.get("conviction_score", 0)), 4),
                }
            )

        return create_response(data=changes_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching 13F changes for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/{symbol}/whales",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_whale_accumulation(
    symbol: str,
    min_position_change: float = 0.10,
    min_aum: float = 1e9,
):
    """
    Detect whale activity alerts for large institutional position changes.

    Focuses on "smart money" by filtering institutions by AUM and
    detecting significant position changes.

    Args:
        symbol: Stock ticker symbol
        min_position_change: Minimum position change to trigger alert (default: 10%)
        min_aum: Minimum AUM in dollars to consider "whale" (default: $1B)
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        whales_df = app_state.institutional_analyzer.detect_whale_accumulation(
            symbol, min_position_change, min_aum
        )

        if whales_df.empty:
            return create_response(data=[])

        whales_list = []
        for _, row in whales_df.iterrows():
            whales_list.append(
                {
                    "institutionName": str(row.get("institution_name", "")),
                    "changeType": str(row.get("change_type", "")),
                    "magnitude": round(float(row.get("magnitude", 0)), 4),
                    "estimatedAum": float(row.get("estimated_aum", 0)),
                    "alertLevel": str(row.get("alert_level", "")),
                    "sharesChanged": int(row.get("shares_changed", 0)),
                }
            )

        return create_response(data=whales_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting whale accumulation for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/{symbol}/sentiment",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_institutional_sentiment_score(symbol: str):
    """
    Calculate multi-factor institutional sentiment score.

    Combines ownership_trend, buyer_seller_ratio, concentration_change,
    and filing_momentum into a unified sentiment score.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        sentiment = app_state.institutional_analyzer.calculate_sentiment_score(symbol)

        return create_response(
            data={
                "symbol": symbol,
                "score": sentiment["score"],
                "classification": sentiment["classification"],
                "confidence": sentiment["confidence"],
                "contributingFactors": sentiment["contributing_factors"],
                "weightsUsed": sentiment["weights_used"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating sentiment for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/{symbol}/clusters",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_position_clusters(symbol: str, n_clusters: int = 4):
    """
    Position clustering analysis for institutional holdings.

    Groups institutions by position sizes using quartile-based clustering
    to identify smart money accumulation/distribution patterns.

    Args:
        symbol: Stock ticker symbol
        n_clusters: Number of clusters (default: 4 for quartiles)
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        clusters = app_state.institutional_analyzer.cluster_positions(
            symbol, n_clusters
        )

        # Convert cluster_labels DataFrame to list if not empty
        cluster_labels = []
        if not clusters["cluster_labels"].empty:
            for _, row in clusters["cluster_labels"].iterrows():
                cluster_labels.append(
                    {
                        "managerName": str(row.get("manager_name", "")),
                        "cluster": int(row.get("cluster", 0)),
                        "clusterName": str(row.get("cluster_name", "")),
                    }
                )

        return create_response(
            data={
                "symbol": symbol,
                "clusterLabels": cluster_labels,
                "clusterStats": clusters["cluster_stats"],
                "smartMoneyDirection": clusters["smart_money_direction"],
                "clusterSummary": clusters["cluster_summary"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clustering positions for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/{symbol}/cross-filing",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_cross_filing_analysis(symbol: str, min_filers: int = 3):
    """
    Cross-filing pattern analysis across multiple 13F filers.

    Analyzes the same stock across multiple institutional filers to detect
    coordinated buying/selling patterns.

    Args:
        symbol: Stock ticker symbol
        min_filers: Minimum number of filers required for analysis (default: 3)
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        cross_analysis = app_state.institutional_analyzer.analyze_cross_filing(
            symbol, min_filers
        )

        return create_response(
            data={
                "symbol": symbol,
                "institutionCount": cross_analysis["institution_count"],
                "institutionAgreement": cross_analysis["institution_agreement"],
                "consensusDirection": cross_analysis["consensus_direction"],
                "crossFilingScore": cross_analysis["cross_filing_score"],
                "filingBreakdown": cross_analysis["filing_breakdown"],
                "buyersCount": cross_analysis.get("buyers_count", 0),
                "sellersCount": cross_analysis.get("sellers_count", 0),
                "holdersCount": cross_analysis.get("holders_count", 0),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing cross-filing for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/{symbol}/momentum",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_smart_money_momentum(
    symbol: str,
    window_quarters: int = 4,
    weight_by_performance: bool = True,
):
    """
    Track smart money momentum with rolling calculations.

    Tracks institutional momentum over configurable windows and detects
    momentum acceleration/deceleration patterns.

    Args:
        symbol: Stock ticker symbol
        window_quarters: Number of quarters for momentum calculation (default: 4)
        weight_by_performance: Weight by manager performance scores (default: True)
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        momentum = app_state.institutional_analyzer.track_smart_money_momentum(
            symbol, window_quarters, weight_by_performance
        )

        return create_response(
            data={
                "symbol": symbol,
                "momentumScore": momentum["momentum_score"],
                "trendDirection": momentum["trend_direction"],
                "acceleration": momentum["acceleration"],
                "quarterlyMomentum": momentum["quarterly_momentum"],
                "topMovers": momentum["top_movers"],
                "signalStrength": momentum["signal_strength"],
                "windowQuarters": momentum["window_quarters"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking momentum for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/{symbol}/smart-money-flow",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_smart_money_flow(symbol: str):
    """
    Calculate net buying/selling by top-performing managers.

    Analyzes flow patterns weighted by manager performance to identify
    coordinated smart money activity.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        flow = app_state.institutional_analyzer.calculate_smart_money_flow(symbol)

        # Convert DataFrames to lists
        buying_list = []
        if not flow["buying_activity"].empty:
            for _, row in flow["buying_activity"].iterrows():
                buying_list.append(
                    {
                        "managerName": str(row.get("manager_name", "")),
                        "managerCik": str(row.get("manager_cik", "")),
                        "sharesAdded": int(row.get("shares_added", 0)),
                        "valueAdded": float(row.get("value_added", 0)),
                        "performanceScore": float(row.get("performance_score", 0)),
                    }
                )

        selling_list = []
        if not flow["selling_activity"].empty:
            for _, row in flow["selling_activity"].iterrows():
                selling_list.append(
                    {
                        "managerName": str(row.get("manager_name", "")),
                        "managerCik": str(row.get("manager_cik", "")),
                        "sharesSold": int(row.get("shares_sold", 0)),
                        "valueSold": float(row.get("value_sold", 0)),
                        "performanceScore": float(row.get("performance_score", 0)),
                    }
                )

        return create_response(
            data={
                "symbol": symbol,
                "netFlow": flow["net_flow"],
                "weightedFlow": flow["weighted_flow"],
                "signal": flow["signal"],
                "signalStrength": flow["signal_strength"],
                "buyersCount": flow["buyers_count"],
                "sellersCount": flow["sellers_count"],
                "buyingActivity": buying_list,
                "sellingActivity": selling_list,
                "coordinatedBuying": flow["coordinated_buying"],
                "coordinatedSelling": flow["coordinated_selling"],
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating smart money flow for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/alerts/new-positions",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_new_positions_alerts(lookback_days: int = 45, min_value: float = 1e7):
    """
    Alert on new institutional positions initiated recently.

    Args:
        lookback_days: Days to look back for new positions (default: 45)
        min_value: Minimum position value to report (default: $10M)
    """
    try:
        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        alerts_df = app_state.institutional_analyzer.get_new_positions_alert(
            lookback_days, min_value
        )

        if alerts_df.empty:
            return create_response(data=[])

        alerts_list = []
        for _, row in alerts_df.iterrows():
            alerts_list.append(
                {
                    "symbol": str(row.get("symbol", "")),
                    "managerName": str(row.get("manager_name", "")),
                    "managerCik": str(row.get("manager_cik", "")),
                    "positionValue": float(row.get("position_value", 0)),
                    "shares": int(row.get("shares", 0)),
                    "weight": round(float(row.get("weight", 0)), 4),
                    "managerPerformanceScore": float(
                        row.get("manager_performance_score", 0)
                    ),
                    "alertType": str(row.get("alert_type", "")),
                    "significance": round(float(row.get("significance", 0)), 4),
                }
            )

        return create_response(data=alerts_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching new position alerts: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/alerts/coordinated-buying",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_coordinated_buying_alerts(min_buyers: int = 3, lookback_days: int = 45):
    """
    Alert on stocks being bought by multiple top managers.

    Args:
        min_buyers: Minimum number of coordinated buyers to report (default: 3)
        lookback_days: Days to look back for coordinated activity (default: 45)
    """
    try:
        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        alerts_df = app_state.institutional_analyzer.get_coordinated_buying_alert(
            min_buyers, lookback_days
        )

        if alerts_df.empty:
            return create_response(data=[])

        alerts_list = []
        for _, row in alerts_df.iterrows():
            alerts_list.append(
                {
                    "symbol": str(row.get("symbol", "")),
                    "buyersCount": int(row.get("buyers_count", 0)),
                    "totalValueAdded": float(row.get("total_value_added", 0)),
                    "avgBuyerPerformance": round(
                        float(row.get("avg_buyer_performance", 0)), 4
                    ),
                    "signalStrength": round(float(row.get("signal_strength", 0)), 4),
                    "buyers": row.get("buyers", []),
                }
            )

        return create_response(data=alerts_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching coordinated buying alerts: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/conviction-picks",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_conviction_picks(min_weight: float = 0.05, top_n_managers: int = 50):
    """
    Get high-conviction positions (stocks with significant portfolio weight).

    Args:
        min_weight: Minimum portfolio weight to consider (default: 5%)
        top_n_managers: Number of top managers to analyze (default: 50)
    """
    try:
        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        picks_df = app_state.institutional_analyzer.get_conviction_picks(
            min_weight, top_n_managers
        )

        if picks_df.empty:
            return create_response(data=[])

        picks_list = []
        for _, row in picks_df.iterrows():
            picks_list.append(
                {
                    "symbol": str(row.get("symbol", "")),
                    "holders": row.get("holders", []),
                    "avgWeight": round(float(row.get("avg_weight", 0)), 4),
                    "maxWeight": round(float(row.get("max_weight", 0)), 4),
                    "holderCount": int(row.get("holder_count", 0)),
                    "totalValue": float(row.get("total_value", 0)),
                    "avgManagerScore": round(float(row.get("avg_manager_score", 0)), 4),
                }
            )

        return create_response(data=picks_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conviction picks: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/institutional/filing-calendar",
    response_model=ApiResponse,
    tags=["Institutional Analytics"],
)
async def get_13f_filing_calendar(quarters_ahead: int = 4):
    """
    Get upcoming 13F filing deadlines.

    Args:
        quarters_ahead: Number of quarters to look ahead (default: 4)
    """
    try:
        if not app_state.institutional_analyzer:
            raise HTTPException(
                status_code=503, detail="Institutional analyzer not initialized"
            )

        calendar_df = app_state.institutional_analyzer.get_13f_filing_calendar(
            quarters_ahead
        )

        if calendar_df.empty:
            return create_response(data=[])

        calendar_list = []
        for _, row in calendar_df.iterrows():
            calendar_list.append(
                {
                    "quarter": str(row.get("quarter", "")),
                    "quarterEnd": (
                        row["quarter_end"].isoformat()
                        if pd.notna(row.get("quarter_end"))
                        else None
                    ),
                    "filingDeadline": (
                        row["filing_deadline"].isoformat()
                        if pd.notna(row.get("filing_deadline"))
                        else None
                    ),
                    "status": str(row.get("status", "")),
                    "daysUntilDeadline": int(row.get("days_until_deadline", 0)),
                }
            )

        return create_response(data=calendar_list)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching filing calendar: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/flows", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_etf_flows(
    symbols: Optional[str] = None,
    lookback_days: int = 90,
):
    """
    Get comprehensive ETF flow analysis.

    Returns creation/redemption flows, momentum, and institutional activity.

    Args:
        symbols: Comma-separated list of ETF symbols (all tracked if not specified)
        lookback_days: Number of days to analyze (default: 90)
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        symbol_list = symbols.split(",") if symbols else None
        flows = await app_state.etf_analyzer.get_etf_flows(
            symbols=symbol_list, lookback_days=lookback_days
        )

        return create_response(data=[f.to_dict() for f in flows])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ETF flows: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/flows/{symbol}", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_etf_flow_detail(symbol: str, lookback_days: int = 30):
    """
    Get detailed creation/redemption activity for a specific ETF.

    Args:
        symbol: ETF symbol
        lookback_days: Number of days to analyze (default: 30)
    """
    try:
        symbol = symbol.upper()

        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        activity = await app_state.etf_analyzer.get_creation_redemption_activity(
            symbol, lookback_days
        )

        return create_response(data=activity)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ETF flow detail for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/sector-rotation", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_sector_rotation(lookback_days: int = 63):
    """
    Get sector ETF rotation signals.

    Returns momentum rankings, relative strength, and rotation recommendations.

    Args:
        lookback_days: Days for momentum calculation (default: 63 ~ 3 months)
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        signals = await app_state.etf_analyzer.get_sector_rotation(lookback_days)

        return create_response(data=[s.to_dict() for s in signals])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sector rotation: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/sector-heatmap", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_sector_heatmap(period: str = "1m"):
    """
    Get sector performance heatmap data.

    Args:
        period: Time period (1d, 1w, 1m, 3m, ytd)
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        heatmap = await app_state.etf_analyzer.get_sector_heatmap(period)

        return create_response(data=heatmap)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sector heatmap: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/smart-beta", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_smart_beta_flows(lookback_days: int = 63):
    """
    Get smart beta factor flow analysis.

    Analyzes flows into value, growth, momentum, quality, low-vol, and size factors.

    Args:
        lookback_days: Days for flow analysis (default: 63)
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        flows = await app_state.etf_analyzer.get_smart_beta_flows(lookback_days)

        return create_response(data=[f.to_dict() for f in flows])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching smart beta flows: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/factor-rotation", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_factor_rotation():
    """
    Get factor rotation signals for tactical allocation.

    Returns recommendations for rotating between value, growth, and other factors.
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        signals = await app_state.etf_analyzer.get_factor_rotation_signals()

        return create_response(data=signals)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching factor rotation: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/thematic", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_thematic_flows(lookback_days: int = 90):
    """
    Get thematic ETF flow analysis.

    Analyzes flows into clean energy, AI, cybersecurity, and other themes.

    Args:
        lookback_days: Days for flow analysis (default: 90)
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        flows = await app_state.etf_analyzer.get_thematic_flows(lookback_days)

        return create_response(data=[f.to_dict() for f in flows])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching thematic flows: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/theme-dashboard", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_theme_dashboard():
    """
    Get thematic investment dashboard.

    Returns hot themes, cooling themes, and overall thematic sentiment.
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        dashboard = await app_state.etf_analyzer.get_theme_dashboard()

        return create_response(data=dashboard)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching theme dashboard: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/institutional", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_institutional_etf_positioning(symbols: Optional[str] = None):
    """
    Get institutional ETF positioning analysis.

    Analyzes 13F institutional holdings in major ETFs.

    Args:
        symbols: Comma-separated list of ETF symbols (major ETFs if not specified)
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        symbol_list = symbols.split(",") if symbols else None
        positioning = await app_state.etf_analyzer.get_institutional_etf_positioning(
            symbol_list
        )

        return create_response(data=positioning)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching institutional positioning: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/etf/overview", response_model=ApiResponse, tags=["ETF Analytics"])
async def get_etf_flow_overview():
    """
    Get comprehensive ETF flow market overview.

    Returns aggregate flows, sentiment, top inflows/outflows, and rotation signals.
    """
    try:
        if not app_state.etf_analyzer:
            raise HTTPException(status_code=503, detail="ETF analyzer not initialized")

        overview = await app_state.etf_analyzer.get_flow_overview()

        return create_response(data=overview)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ETF overview: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Enhanced Money Flow Endpoints
# =============================================================================


@app.get(
    "/api/money-flow/{symbol}/alerts",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def get_dark_pool_alerts(symbol: str, lookback_days: int = 20):
    """
    Get dark pool alerts for a symbol.

    Detects and returns alerts for:
    - Dark pool surges (>35% activity)
    - Dark pool declines (<15% activity)
    - Unusual dark pool activity changes

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

        alerts = app_state.money_flow_analyzer.detect_dark_pool_alerts(
            symbol, lookback_days
        )

        return create_response(data=[alert.to_dict() for alert in alerts])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting dark pool alerts for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/{symbol}/block-trades",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def get_block_trades(
    symbol: str, price: Optional[float] = None, lookback_days: int = 20
):
    """
    Detect block trades for a symbol.

    Returns block trades classified by size:
    - Small: 10K-50K shares or $200K-$1M
    - Medium: 50K-100K shares or $1M-$5M
    - Large: 100K-500K shares or $5M-$25M
    - Mega: 500K+ shares or $25M+

    Args:
        symbol: Stock ticker symbol
        price: Current stock price (optional, for notional value calculation)
        lookback_days: Number of days to analyze (default: 20)
    """
    try:
        symbol = symbol.upper()

        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        block_trades = app_state.money_flow_analyzer.detect_block_trades(
            symbol, price=price, lookback_days=lookback_days
        )

        return create_response(data=[bt.to_dict() for bt in block_trades])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting block trades for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/sector-rotation",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def get_money_flow_sector_rotation(
    sectors: Optional[str] = None, lookback_days: int = 21
):
    """
    Detect sector rotation patterns based on money flow.

    Analyzes relative strength and momentum across sectors to identify
    rotation patterns used by institutional investors.

    Args:
        sectors: Comma-separated list of sector ETF symbols (default: major sectors)
        lookback_days: Number of days for momentum calculation (default: 21)
    """
    try:
        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        sector_list = None
        if sectors:
            sector_list = [s.strip().upper() for s in sectors.split(",")]

        signal = app_state.money_flow_analyzer.detect_sector_rotation(
            sectors=sector_list, lookback_days=lookback_days
        )

        return create_response(data=signal.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting sector rotation: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/{symbol}/smart-money",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def track_smart_money(symbol: str, lookback_days: int = 21):
    """
    Track smart money activity for a symbol.

    Aggregates institutional flow data to identify smart money
    accumulation or distribution patterns.

    Returns:
    - Institutional net flow and direction
    - Ownership metrics
    - Smart money score (-1 to 1)
    - Smart money trend (accumulating/distributing/neutral)

    Args:
        symbol: Stock ticker symbol
        lookback_days: Number of days to analyze (default: 21)
    """
    try:
        symbol = symbol.upper()

        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        metrics = app_state.money_flow_analyzer.track_smart_money(symbol, lookback_days)

        return create_response(data=metrics.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking smart money for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/{symbol}/unusual-volume",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def detect_unusual_volume(symbol: str, lookback_days: int = 20):
    """
    Detect unusual volume activity for a symbol.

    Uses statistical analysis to identify volume anomalies that may
    indicate institutional activity.

    Returns:
    - Current vs average volume
    - Volume ratio and z-score
    - Percentile rank
    - Likely direction (accumulation/distribution)

    Args:
        symbol: Stock ticker symbol
        lookback_days: Number of days for baseline calculation (default: 20)
    """
    try:
        symbol = symbol.upper()

        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        signal = app_state.money_flow_analyzer.detect_unusual_volume(
            symbol, lookback_days
        )

        return create_response(data=signal.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting unusual volume for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/{symbol}/momentum",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def calculate_flow_momentum(symbol: str, lookback_days: int = 21):
    """
    Calculate flow momentum indicators for a symbol.

    Analyzes the rate of change and acceleration of money flows
    to identify momentum shifts.

    Returns:
    - Flow momentum and acceleration
    - Trend direction and strength
    - Moving averages (5, 10, 20 day)
    - MA crossover signals
    - Momentum divergence detection

    Args:
        symbol: Stock ticker symbol
        lookback_days: Number of days for analysis (default: 21)
    """
    try:
        symbol = symbol.upper()

        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        indicator = app_state.money_flow_analyzer.calculate_flow_momentum(
            symbol, lookback_days
        )

        return create_response(data=indicator.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating flow momentum for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/{symbol}/comprehensive",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def get_comprehensive_money_flow(symbol: str, lookback_days: int = 21):
    """
    Get comprehensive money flow analysis for a symbol.

    Combines all enhanced money flow features into a single analysis report:
    - Basic money flow metrics
    - Dark pool alerts
    - Block trades detected
    - Smart money tracking
    - Unusual volume signals
    - Flow momentum indicators
    - All active alerts

    This is the recommended endpoint for a complete money flow overview.

    Args:
        symbol: Stock ticker symbol
        lookback_days: Number of days to analyze (default: 21)
    """
    try:
        symbol = symbol.upper()

        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        analysis = app_state.money_flow_analyzer.get_comprehensive_analysis(
            symbol, lookback_days
        )

        return create_response(data=analysis)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comprehensive analysis for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/alerts",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def get_all_money_flow_alerts(
    symbol: Optional[str] = None,
    alert_type: Optional[str] = None,
):
    """
    Get all active money flow alerts.

    Returns alerts across all analyzed symbols, optionally filtered.

    Args:
        symbol: Filter by stock symbol (optional)
        alert_type: Filter by alert type (optional). Valid types:
            - dark_pool_surge
            - dark_pool_decline
            - block_trade
            - unusual_volume
            - sector_rotation
            - smart_money_inflow
            - smart_money_outflow
            - flow_momentum_shift
    """
    from ..analytics.alerts import AlertType as AT

    try:
        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        # Parse alert type if provided
        at = None
        if alert_type:
            alert_type_map = {
                "dark_pool_surge": AT.DARK_POOL_SURGE,
                "dark_pool_decline": AT.DARK_POOL_DECLINE,
                "block_trade": AT.BLOCK_TRADE,
                "unusual_volume": AT.UNUSUAL_VOLUME,
                "sector_rotation": AT.SECTOR_ROTATION,
                "smart_money_inflow": AT.SMART_MONEY_INFLOW,
                "smart_money_outflow": AT.SMART_MONEY_OUTFLOW,
                "flow_momentum_shift": AT.FLOW_MOMENTUM_SHIFT,
            }
            at = alert_type_map.get(alert_type.lower())
            if alert_type and at is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid alert_type. Valid types: {list(alert_type_map.keys())}",
                )

        alerts = app_state.money_flow_analyzer.get_alerts(
            symbol=symbol.upper() if symbol else None,
            alert_type=at,
        )

        return create_response(data=[alert.to_dict() for alert in alerts])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting money flow alerts: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/money-flow/alerts/summary",
    response_model=ApiResponse,
    tags=["Money Flow"],
)
async def get_alert_summary():
    """
    Get summary of all active money flow alerts.

    Returns aggregate counts by:
    - Alert type
    - Severity level
    - Symbol
    """
    try:
        if not app_state.money_flow_analyzer:
            raise HTTPException(
                status_code=503, detail="Money flow analyzer not initialized"
            )

        summary = app_state.money_flow_analyzer.alert_aggregator.get_alert_summary()

        return create_response(data=summary)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting alert summary: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Accounting Quality Endpoints
# =============================================================================


class EarningsQualityResponse(BaseModel):
    """Earnings quality analysis response."""

    symbol: str
    overallQuality: str
    mScore: float
    mScoreRisk: str
    fScore: int
    fScoreRating: str
    zScore: float
    zScoreZone: str
    bankruptcyProbability: float
    accrualRatio: float
    manipulationFlags: List[str]
    timestamp: str


class RedFlagResponse(BaseModel):
    """Red flag analysis response."""

    symbol: str
    totalScore: float
    riskLevel: str
    redFlags: List[Dict[str, Any]]
    categorySummary: Dict[str, int]
    timestamp: str


class AnomalyResponse(BaseModel):
    """Anomaly detection response."""

    symbol: str
    totalAnomalies: int
    anomalies: List[Dict[str, Any]]
    benfordScore: Optional[float]
    disclosureQuality: Optional[float]
    timestamp: str


@app.get(
    "/api/accounting/earnings-quality/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_earnings_quality(symbol: str, manufacturing: bool = False):
    """
    Get comprehensive earnings quality analysis for a symbol.

    Includes Beneish M-Score, Piotroski F-Score, Altman Z-Score, and accrual analysis.

    Args:
        symbol: Stock ticker symbol
        manufacturing: Whether to use manufacturing-specific Z-Score formula
    """
    try:
        symbol = symbol.upper()

        if not app_state.earnings_quality_analyzer:
            raise HTTPException(
                status_code=503, detail="Earnings quality analyzer not initialized"
            )

        result = app_state.earnings_quality_analyzer.analyze(
            symbol, manufacturing=manufacturing
        )

        return create_response(
            data={
                "symbol": symbol,
                "overallRating": (
                    result.overall_rating.value
                    if hasattr(result.overall_rating, "value")
                    else str(result.overall_rating)
                ),
                "overallScore": result.overall_score,
                "mScore": result.m_score.m_score if result.m_score else None,
                "mScoreRisk": (
                    result.m_score.risk_level.value
                    if result.m_score and hasattr(result.m_score.risk_level, "value")
                    else None
                ),
                "isLikelyManipulator": (
                    result.m_score.is_likely_manipulator if result.m_score else False
                ),
                "fScore": result.f_score.f_score if result.f_score else None,
                "fScoreCategory": (result.f_score.category if result.f_score else None),
                "zScore": result.z_score.z_score if result.z_score else None,
                "zScoreZone": result.z_score.zone if result.z_score else None,
                "accrualRatio": result.accrual_ratio,
                "cashConversion": result.cash_conversion,
                "earningsPersistence": result.earnings_persistence,
                "redFlags": result.red_flags,
                "timestamp": get_timestamp(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing earnings quality for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/m-score/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_beneish_m_score(symbol: str):
    """
    Get Beneish M-Score analysis for earnings manipulation detection.

    M-Score < -2.22 suggests low manipulation risk.
    M-Score > -2.22 suggests higher manipulation risk.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.earnings_quality_analyzer:
            raise HTTPException(
                status_code=503, detail="Earnings quality analyzer not initialized"
            )

        result = app_state.earnings_quality_analyzer.m_score_calc.calculate(symbol)

        return create_response(
            data={
                "symbol": symbol,
                "mScore": result.m_score,
                "isLikelyManipulator": result.is_likely_manipulator,
                "components": result.components,
                "riskLevel": result.risk_level.value,
                "threshold": -1.78,
                "timestamp": get_timestamp(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating M-Score for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/f-score/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_piotroski_f_score(symbol: str):
    """
    Get Piotroski F-Score for financial strength assessment.

    F-Score 8-9: Strong fundamentals
    F-Score 5-7: Neutral
    F-Score 0-4: Weak fundamentals

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.earnings_quality_analyzer:
            raise HTTPException(
                status_code=503, detail="Earnings quality analyzer not initialized"
            )

        result = app_state.earnings_quality_analyzer.f_score_calc.calculate(symbol)

        return create_response(
            data={
                "symbol": symbol,
                "fScore": result.f_score,
                "category": result.category,
                "signals": result.signals,
                "timestamp": get_timestamp(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating F-Score for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/z-score/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_altman_z_score(symbol: str, manufacturing: bool = False):
    """
    Get Altman Z-Score for bankruptcy risk assessment.

    Manufacturing: Z > 2.99 Safe, 1.81-2.99 Gray Zone, < 1.81 Distress
    Service: Z' > 2.6 Safe, 1.1-2.6 Gray Zone, < 1.1 Distress

    Args:
        symbol: Stock ticker symbol
        manufacturing: Use manufacturing formula (default: service/non-manufacturing)
    """
    try:
        symbol = symbol.upper()

        if not app_state.earnings_quality_analyzer:
            raise HTTPException(
                status_code=503, detail="Earnings quality analyzer not initialized"
            )

        result = app_state.earnings_quality_analyzer.z_score_calc.calculate(
            symbol, manufacturing=manufacturing
        )

        return create_response(
            data={
                "symbol": symbol,
                "zScore": result.z_score,
                "zone": result.zone,
                "components": result.components,
                "formulaType": "manufacturing" if manufacturing else "service",
                "timestamp": get_timestamp(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating Z-Score for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/red-flags/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_red_flags(symbol: str):
    """
    Get comprehensive red flag analysis for a symbol.

    Detects revenue manipulation, expense irregularities, accrual issues,
    off-balance-sheet concerns, and cash flow anomalies.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.red_flag_scorer:
            raise HTTPException(
                status_code=503, detail="Red flag scorer not initialized"
            )

        result = app_state.red_flag_scorer.score(symbol)

        # Convert red flags to serializable format
        flags_data = []
        for flag in result.red_flags:
            flags_data.append(
                {
                    "category": flag.category,
                    "description": flag.description,
                    "severity": (
                        flag.severity.value
                        if hasattr(flag.severity, "value")
                        else str(flag.severity)
                    ),
                    "metric": flag.metric,
                    "value": flag.value,
                    "threshold": flag.threshold,
                }
            )

        response = RedFlagResponse(
            symbol=symbol,
            totalScore=result.total_score,
            riskLevel=result.risk_level,
            redFlags=flags_data,
            categorySummary=result.category_summary,
            timestamp=get_timestamp(),
        )

        return create_response(data=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing red flags for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/red-flags/{symbol}/peers",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_red_flags_peer_comparison(symbol: str, peers: Optional[str] = None):
    """
    Compare red flags against peer companies.

    Args:
        symbol: Stock ticker symbol
        peers: Comma-separated list of peer symbols (auto-detected if not provided)
    """
    try:
        symbol = symbol.upper()

        if not app_state.red_flag_scorer:
            raise HTTPException(
                status_code=503, detail="Red flag scorer not initialized"
            )

        peer_list = peers.split(",") if peers else None
        comparison = app_state.red_flag_scorer.compare_peer_red_flags(symbol, peer_list)

        if comparison.empty:
            return create_response(data={"symbol": symbol, "comparison": {}})

        return create_response(
            data={
                "symbol": symbol,
                "comparison": comparison.to_dict(orient="index"),
                "timestamp": get_timestamp(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing red flags for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/anomalies/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_anomalies(symbol: str):
    """
    Get comprehensive anomaly detection analysis for a symbol.

    Includes time-series anomalies, Benford's Law analysis, peer comparison,
    footnote analysis, disclosure quality, and seasonal patterns.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.anomaly_aggregator:
            raise HTTPException(
                status_code=503, detail="Anomaly aggregator not initialized"
            )

        result = app_state.anomaly_aggregator.detect_all_anomalies(symbol)

        # Convert anomalies to serializable format
        anomalies_data = []
        for anomaly in result.anomalies:
            anomalies_data.append(
                {
                    "type": anomaly.anomaly_type,
                    "description": anomaly.description,
                    "severity": anomaly.severity,
                    "metric": anomaly.metric,
                    "value": anomaly.value,
                    "expectedRange": anomaly.expected_range,
                }
            )

        response = AnomalyResponse(
            symbol=symbol,
            totalAnomalies=len(result.anomalies),
            anomalies=anomalies_data,
            benfordScore=result.benford_score,
            disclosureQuality=result.disclosure_quality,
            timestamp=get_timestamp(),
        )

        return create_response(data=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting anomalies for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/accruals/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_accrual_analysis(symbol: str):
    """
    Get detailed accrual analysis for a symbol.

    Includes Sloan accrual ratio, Richardson decomposition, and working capital accruals.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.earnings_quality_analyzer:
            raise HTTPException(
                status_code=503, detail="Earnings quality analyzer not initialized"
            )

        accrual_ratio = (
            app_state.earnings_quality_analyzer.accrual_calc.calculate_accrual_ratio(
                symbol
            )
        )
        cash_conversion = (
            app_state.earnings_quality_analyzer.accrual_calc.calculate_cash_conversion(
                symbol
            )
        )

        # Flag high accruals (>10%) as lower quality
        quality_flag = (
            abs(accrual_ratio) > 0.10 if not np.isnan(accrual_ratio) else False
        )

        return create_response(
            data={
                "symbol": symbol,
                "accrualRatio": accrual_ratio if not np.isnan(accrual_ratio) else None,
                "cashConversion": (
                    cash_conversion if not np.isnan(cash_conversion) else None
                ),
                "qualityFlag": quality_flag,
                "timestamp": get_timestamp(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing accruals for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/accounting/comprehensive/{symbol}",
    response_model=ApiResponse,
    tags=["Accounting Quality"],
)
async def get_comprehensive_accounting_quality(
    symbol: str, manufacturing: bool = False
):
    """
    Get comprehensive accounting quality report combining all analyses.

    Includes earnings quality, red flags, anomalies, and overall assessment.
    This is the primary endpoint for full fundamental quality analysis.

    Args:
        symbol: Stock ticker symbol
        manufacturing: Whether to use manufacturing-specific formulas
    """
    try:
        symbol = symbol.upper()

        # Collect all available analyses
        report = {"symbol": symbol, "timestamp": get_timestamp()}

        # Earnings Quality
        if app_state.earnings_quality_analyzer:
            try:
                eq_result = app_state.earnings_quality_analyzer.analyze(
                    symbol, manufacturing=manufacturing
                )
                report["earningsQuality"] = {
                    "overallRating": (
                        eq_result.overall_rating.value
                        if hasattr(eq_result.overall_rating, "value")
                        else str(eq_result.overall_rating)
                    ),
                    "overallScore": eq_result.overall_score,
                    "mScore": {
                        "score": (
                            eq_result.m_score.m_score if eq_result.m_score else None
                        ),
                        "riskLevel": (
                            eq_result.m_score.risk_level.value
                            if eq_result.m_score
                            else None
                        ),
                        "isLikelyManipulator": (
                            eq_result.m_score.is_likely_manipulator
                            if eq_result.m_score
                            else False
                        ),
                    },
                    "fScore": {
                        "score": (
                            eq_result.f_score.f_score if eq_result.f_score else None
                        ),
                        "category": (
                            eq_result.f_score.category if eq_result.f_score else None
                        ),
                    },
                    "zScore": {
                        "score": (
                            eq_result.z_score.z_score if eq_result.z_score else None
                        ),
                        "zone": eq_result.z_score.zone if eq_result.z_score else None,
                    },
                    "accrualRatio": eq_result.accrual_ratio,
                    "cashConversion": eq_result.cash_conversion,
                    "redFlags": eq_result.red_flags,
                }
            except Exception as e:
                logger.warning(f"Earnings quality analysis failed: {e}")
                report["earningsQuality"] = {"error": str(e)}

        # Red Flags
        if app_state.red_flag_scorer:
            try:
                rf_result = app_state.red_flag_scorer.score(symbol)
                flags_data = []
                for flag in rf_result.red_flags:
                    flags_data.append(
                        {
                            "category": flag.category,
                            "description": flag.description,
                            "severity": (
                                flag.severity.value
                                if hasattr(flag.severity, "value")
                                else str(flag.severity)
                            ),
                        }
                    )
                report["redFlags"] = {
                    "totalScore": rf_result.total_score,
                    "riskLevel": rf_result.risk_level,
                    "flagCount": len(rf_result.red_flags),
                    "flags": flags_data[:10],  # Top 10 flags
                    "categorySummary": rf_result.category_summary,
                }
            except Exception as e:
                logger.warning(f"Red flag analysis failed: {e}")
                report["redFlags"] = {"error": str(e)}

        # Anomalies
        if app_state.anomaly_aggregator:
            try:
                an_result = app_state.anomaly_aggregator.detect_all_anomalies(symbol)
                anomalies_data = []
                for anomaly in an_result.anomalies[:10]:  # Top 10 anomalies
                    anomalies_data.append(
                        {
                            "type": anomaly.anomaly_type,
                            "description": anomaly.description,
                            "severity": anomaly.severity,
                        }
                    )
                report["anomalies"] = {
                    "totalCount": len(an_result.anomalies),
                    "benfordScore": an_result.benford_score,
                    "disclosureQuality": an_result.disclosure_quality,
                    "anomalies": anomalies_data,
                }
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")
                report["anomalies"] = {"error": str(e)}

        # Overall Assessment
        quality_score = 100
        risk_factors = []

        # Deduct for M-Score risk
        if "earningsQuality" in report and "error" not in report["earningsQuality"]:
            eq = report["earningsQuality"]
            if eq.get("mScore", {}).get("risk") == "High":
                quality_score -= 30
                risk_factors.append("High earnings manipulation risk (M-Score)")
            elif eq.get("mScore", {}).get("risk") == "Moderate":
                quality_score -= 15
                risk_factors.append("Moderate earnings manipulation risk")

            # Deduct for weak F-Score
            if eq.get("fScore", {}).get("score", 9) <= 3:
                quality_score -= 25
                risk_factors.append("Weak financial fundamentals (F-Score)")
            elif eq.get("fScore", {}).get("score", 9) <= 5:
                quality_score -= 10

            # Deduct for distress zone
            if eq.get("zScore", {}).get("zone") == "Distress":
                quality_score -= 30
                risk_factors.append("High bankruptcy risk (Z-Score)")
            elif eq.get("zScore", {}).get("zone") == "Gray Zone":
                quality_score -= 15

        # Deduct for red flags
        if "redFlags" in report and "error" not in report["redFlags"]:
            rf = report["redFlags"]
            if rf.get("riskLevel") == "Critical":
                quality_score -= 30
                risk_factors.append("Critical red flags detected")
            elif rf.get("riskLevel") == "High":
                quality_score -= 20
                risk_factors.append("High red flag risk")
            elif rf.get("riskLevel") == "Medium":
                quality_score -= 10

        # Deduct for anomalies
        if "anomalies" in report and "error" not in report["anomalies"]:
            an = report["anomalies"]
            if an.get("totalCount", 0) > 10:
                quality_score -= 15
                risk_factors.append("High number of financial anomalies")
            elif an.get("totalCount", 0) > 5:
                quality_score -= 8

        quality_score = max(0, min(100, quality_score))

        if quality_score >= 80:
            overall_rating = "Excellent"
        elif quality_score >= 60:
            overall_rating = "Good"
        elif quality_score >= 40:
            overall_rating = "Fair"
        elif quality_score >= 20:
            overall_rating = "Poor"
        else:
            overall_rating = "Critical"

        report["overallAssessment"] = {
            "qualityScore": quality_score,
            "rating": overall_rating,
            "riskFactors": risk_factors,
        }

        return create_response(data=report)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comprehensive report for {symbol}: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Signal Generation Endpoints
# =============================================================================


class SignalRequest(BaseModel):
    """Request for signal generation."""

    symbols: List[str] = Field(
        ..., description="List of stock symbols to generate signals for"
    )
    min_conviction: float = Field(
        default=0.3, ge=0, le=1, description="Minimum conviction threshold"
    )


class BacktestRequest(BaseModel):
    """Request for signal backtesting."""

    symbols: List[str] = Field(..., description="Symbols to backtest")
    start_date: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    holding_period_days: int = Field(
        default=30, ge=1, le=365, description="Holding period"
    )


@app.get("/api/signals/{symbol}", response_model=ApiResponse, tags=["Signals"])
async def get_signal(symbol: str):
    """
    Generate investment signal for a single symbol.

    Returns a multi-factor signal with conviction indicators,
    price targets, and factor breakdown.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)
    """
    try:
        symbol = symbol.upper()

        if not app_state.signal_generator:
            raise HTTPException(
                status_code=503, detail="Signal generator not initialized"
            )

        signal = await app_state.signal_generator.generate_signal(symbol)

        # Record for tracking
        if app_state.performance_tracker:
            app_state.performance_tracker.record_signal(signal)

        return create_response(data=signal.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.post("/api/signals", response_model=ApiResponse, tags=["Signals"])
async def generate_signals(request: SignalRequest):
    """
    Generate signals for multiple symbols.

    Returns signals for all symbols that meet the conviction threshold,
    sorted by conviction score.

    Args:
        request: SignalRequest with symbols and filters
    """
    try:
        if not app_state.signal_generator:
            raise HTTPException(
                status_code=503, detail="Signal generator not initialized"
            )

        symbols = [s.upper() for s in request.symbols]

        result = await app_state.signal_generator.generate_universe_signals(
            universe=symbols,
            min_conviction=request.min_conviction,
        )

        # Convert DataFrame to list of dicts
        signals_data = result.to_dict(orient="records") if not result.empty else []

        return create_response(
            data={
                "signals": signals_data,
                "totalRequested": len(symbols),
                "signalsGenerated": len(signals_data),
                "filters": {
                    "minConviction": request.min_conviction,
                },
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/signals/{symbol}/composite", response_model=ApiResponse, tags=["Signals"]
)
async def get_composite_score(symbol: str):
    """
    Get detailed composite score breakdown for a symbol.

    Returns individual factor scores and their contributions
    to the overall signal.

    Args:
        symbol: Stock ticker symbol
    """
    try:
        symbol = symbol.upper()

        if not app_state.signal_generator:
            raise HTTPException(
                status_code=503, detail="Signal generator not initialized"
            )

        composite = await app_state.signal_generator.get_composite_score(symbol)

        return create_response(data=composite.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting composite score for {symbol}: {e}")
        return create_response(error=str(e), success=False)


@app.get("/api/signals/performance/stats", response_model=ApiResponse, tags=["Signals"])
async def get_signal_performance_stats():
    """
    Get aggregate performance statistics for tracked signals.

    Returns win rate, average return, and factor performance metrics.
    """
    try:
        if not app_state.performance_tracker:
            raise HTTPException(
                status_code=503, detail="Performance tracker not initialized"
            )

        stats = app_state.performance_tracker.get_performance_stats()

        return create_response(data=stats.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        return create_response(error=str(e), success=False)


@app.get(
    "/api/signals/performance/history", response_model=ApiResponse, tags=["Signals"]
)
async def get_signal_history(
    symbol: Optional[str] = None,
    limit: int = 100,
):
    """
    Get signal history with outcomes.

    Args:
        symbol: Filter by symbol (optional)
        limit: Maximum records to return
    """
    try:
        if not app_state.performance_tracker:
            raise HTTPException(
                status_code=503, detail="Performance tracker not initialized"
            )

        history = app_state.performance_tracker.get_signal_history(
            symbol=symbol.upper() if symbol else None,
            limit=limit,
        )

        # Convert DataFrame to list of dicts
        history_data = history.to_dict(orient="records") if not history.empty else []

        return create_response(
            data={
                "history": history_data,
                "count": len(history_data),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal history: {e}")
        return create_response(error=str(e), success=False)


@app.post(
    "/api/signals/{signal_id}/outcome", response_model=ApiResponse, tags=["Signals"]
)
async def record_signal_outcome(
    signal_id: str,
    exit_price: float,
    exit_reason: str = "manual",
):
    """
    Record the outcome of a signal.

    Args:
        signal_id: Signal ID to update
        exit_price: Exit price
        exit_reason: Reason for exit (e.g., target, stop_loss, manual)
    """
    try:
        if not app_state.performance_tracker:
            raise HTTPException(
                status_code=503, detail="Performance tracker not initialized"
            )

        record = app_state.performance_tracker.record_outcome(
            signal_id=signal_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

        if record is None:
            raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")

        return create_response(data=record.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording outcome for {signal_id}: {e}")
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
