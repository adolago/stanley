"""
Stanley REST API - Refactored Main Module
==========================================

FastAPI-based REST API for the Stanley institutional investment analysis platform.
This is the refactored entry point using modular routers and middleware integration.

Features:
    - Modular router architecture for better organization
    - Integrated authentication and rate limiting middleware
    - Dependency injection container for services
    - CORS configuration for frontend clients
    - Global exception handling
    - Structured logging

Usage:
    # Development
    uvicorn stanley.api.main_new:app --reload --port 8000

    # Production
    uvicorn stanley.api.main_new:app --host 0.0.0.0 --port 8000 --workers 4

Environment Variables:
    SEC_IDENTITY: Email for SEC EDGAR API identification
    STANLEY_AUTH_JWT_SECRET_KEY: JWT signing key (32+ chars)
    STANLEY_AUTH_CORS_ORIGINS: Comma-separated CORS origins
    STANLEY_AUTH_RATE_LIMIT_ENABLED: Enable rate limiting (default: true)
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from stanley.api.dependencies import get_container, Container
from stanley.api.routers.registration import register_routers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Handles startup initialization of services and graceful shutdown.
    The container is stored in app.state for access by endpoints.
    """
    # Startup
    logger.info("Stanley API starting up...")

    container = get_container()
    await container.initialize()

    # Store container in app state for access by routers
    app.state.app_state = container

    logger.info("Stanley API ready")

    yield

    # Shutdown
    logger.info("Stanley API shutting down...")
    await container.close()
    logger.info("Stanley API shutdown complete")


# =============================================================================
# Application Factory
# =============================================================================


def create_app() -> FastAPI:
    """
    Application factory for creating the FastAPI instance.

    This factory pattern allows for:
    - Easy testing with custom configurations
    - Multiple app instances if needed
    - Clear separation of configuration

    Returns:
        Configured FastAPI application instance
    """
    # Try to load auth settings, with fallback for development
    try:
        from stanley.api.auth import get_auth_settings, RateLimitMiddleware
        settings = get_auth_settings()
        cors_origins = settings.CORS_ORIGINS
        rate_limit_enabled = settings.RATE_LIMIT_ENABLED
        has_auth = True
    except Exception as e:
        logger.warning(f"Auth settings not configured: {e}")
        logger.warning("Running without authentication - set STANLEY_AUTH_JWT_SECRET_KEY for auth")
        cors_origins = [
            "http://localhost:1420",  # Tauri dev server
            "http://127.0.0.1:1420",
            "http://localhost:3000",  # React dev server
            "http://localhost:8080",  # Alternative dev server
            "tauri://localhost",      # Tauri production
            "https://tauri.localhost",
        ]
        rate_limit_enabled = False
        has_auth = False

    app = FastAPI(
        title="Stanley API",
        description="Institutional Investment Analysis Platform - REST API",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "System", "description": "Health checks and system status"},
            {"name": "Settings", "description": "User preferences and configuration"},
            {"name": "Market Data", "description": "Real-time and historical market data"},
            {"name": "Institutional", "description": "13F holdings and institutional analysis"},
            {"name": "Analytics", "description": "Money flow and sector rotation"},
            {"name": "Research", "description": "Valuation, earnings, and peer analysis"},
            {"name": "Options", "description": "Options flow and Greeks analysis"},
            {"name": "ETF", "description": "ETF flows and sector rotation"},
            {"name": "Macro", "description": "Economic indicators and regime detection"},
            {"name": "Commodities", "description": "Commodity prices and correlations"},
            {"name": "Accounting", "description": "SEC filings and financial analysis"},
            {"name": "Signals", "description": "Signal generation and backtesting"},
            {"name": "Notes", "description": "Research vault and trade journal"},
            {"name": "Portfolio", "description": "Portfolio analytics and risk metrics"},
        ],
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware (if auth is configured)
    if has_auth and rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware)
        logger.info("Rate limiting middleware enabled")

    # Register all routers
    register_routers(app)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions with structured response."""
        logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if os.environ.get("DEBUG") else None,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )

    # HTTP exception handler (for custom formatting)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured response."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            headers=exc.headers,
        )

    return app


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()


# =============================================================================
# Utility Functions (for backward compatibility during migration)
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
    data: Any = None,
    error: Optional[str] = None,
    success: bool = True
) -> dict:
    """Create a standardized API response."""
    converted_data = _convert_numpy_types(data) if data is not None else None

    return {
        "success": success and error is None,
        "data": converted_data,
        "error": error,
        "timestamp": get_timestamp(),
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "app",
    "create_app",
    "get_timestamp",
    "create_response",
    "_convert_numpy_types",
]
