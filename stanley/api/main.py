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
from ..data.data_manager import DataManager

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
        if not app_state.data_manager:
            raise HTTPException(status_code=503, detail="Data manager not initialized")

        if not request.holdings:
            return create_response(data=None, error="No holdings provided")

        # Calculate portfolio analytics
        end_date = datetime.now()
        start_date = datetime(end_date.year, end_date.month, 1)  # Start of month

        holdings_data = []
        total_value = 0.0
        total_cost = 0.0

        # Fetch current prices for all holdings
        for holding in request.holdings:
            symbol = holding.symbol.upper()

            try:
                stock_data = await app_state.data_manager.get_stock_data(
                    symbol, start_date, end_date
                )
                if not stock_data.empty:
                    current_price = float(stock_data.iloc[-1]["close"])
                else:
                    # Fallback mock price
                    current_price = 100.0 + np.random.uniform(-20, 50)
            except Exception as e:
                logger.warning(f"Failed to fetch price for {symbol}: {e}")
                current_price = 100.0 + np.random.uniform(-20, 50)

            average_cost = holding.average_cost if holding.average_cost else current_price
            market_value = holding.shares * current_price
            cost_basis = holding.shares * average_cost

            holdings_data.append({
                "symbol": symbol,
                "shares": holding.shares,
                "averageCost": average_cost,
                "currentPrice": current_price,
                "marketValue": market_value,
                "costBasis": cost_basis,
            })

            total_value += market_value
            total_cost += cost_basis

        # Calculate weights
        for h in holdings_data:
            h["weight"] = (h["marketValue"] / total_value * 100) if total_value > 0 else 0

        # Calculate returns
        total_return = total_value - total_cost
        total_return_percent = (
            (total_return / total_cost * 100) if total_cost > 0 else 0
        )

        # Calculate portfolio metrics (simplified/mock for now)
        # In production, these would use proper risk calculations
        portfolio_beta = 1.0 + np.random.uniform(-0.3, 0.3)
        var_95 = total_value * 0.02  # 2% VaR estimate
        var_99 = total_value * 0.035  # 3.5% VaR estimate

        # Mock sector exposure based on common sector allocations
        sector_exposure = {
            "Technology": np.random.uniform(20, 40),
            "Healthcare": np.random.uniform(10, 20),
            "Financial": np.random.uniform(10, 20),
            "Consumer": np.random.uniform(5, 15),
            "Industrial": np.random.uniform(5, 15),
            "Other": 0,
        }
        # Normalize to 100%
        total_sector = sum(sector_exposure.values())
        sector_exposure = {k: round(v / total_sector * 100, 2) for k, v in sector_exposure.items()}

        # Sort holdings by market value for top holdings
        sorted_holdings = sorted(holdings_data, key=lambda x: x["marketValue"], reverse=True)

        top_holdings = [
            PortfolioHolding(
                symbol=h["symbol"],
                shares=h["shares"],
                averageCost=round(h["averageCost"], 2),
                currentPrice=round(h["currentPrice"], 2),
                marketValue=round(h["marketValue"], 2),
                weight=round(h["weight"], 2),
            ).model_dump()
            for h in sorted_holdings[:10]  # Top 10 holdings
        ]

        analytics = PortfolioAnalytics(
            totalValue=round(total_value, 2),
            totalReturn=round(total_return, 2),
            totalReturnPercent=round(total_return_percent, 2),
            beta=round(portfolio_beta, 3),
            var95=round(var_95, 2),
            var99=round(var_99, 2),
            sectorExposure=sector_exposure,
            topHoldings=top_holdings,
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
