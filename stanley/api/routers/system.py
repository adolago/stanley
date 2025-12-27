"""
Stanley System Router
======================

System-level API endpoints for health checks, version information,
and system status monitoring.

Endpoints:
    GET /api/health     - Health check with component status
    GET /api/version    - Version and build information
    GET /api/status     - Detailed system status and metrics
"""

import logging
import platform
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


# =============================================================================
# Response Models
# =============================================================================


class ComponentHealth(BaseModel):
    """Health status of an individual component."""

    name: str = Field(..., description="Component name")
    healthy: bool = Field(..., description="Whether component is healthy")
    latency_ms: Optional[float] = Field(None, description="Response latency in milliseconds")
    details: Optional[str] = Field(None, description="Additional details or error message")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall status: healthy, degraded, or unhealthy")
    version: str = Field(..., description="API version")
    components: Dict[str, bool] = Field(..., description="Component health status")
    timestamp: str = Field(..., description="ISO timestamp of the check")


class VersionInfo(BaseModel):
    """Version and build information."""

    version: str = Field(..., description="API semantic version")
    api_version: str = Field(default="v1", description="API version prefix")
    python_version: str = Field(..., description="Python runtime version")
    platform: str = Field(..., description="Operating system platform")
    build_date: Optional[str] = Field(None, description="Build timestamp")
    git_commit: Optional[str] = Field(None, description="Git commit hash")


class SystemMetrics(BaseModel):
    """System performance metrics."""

    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    request_count: int = Field(default=0, description="Total requests processed")
    active_connections: int = Field(default=0, description="Current active connections")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, description="CPU usage percentage")


class SystemStatus(BaseModel):
    """Comprehensive system status."""

    status: str = Field(..., description="Overall system status")
    version: VersionInfo = Field(..., description="Version information")
    components: list[ComponentHealth] = Field(default=[], description="Component health details")
    metrics: Optional[SystemMetrics] = Field(None, description="Performance metrics")
    timestamp: str = Field(..., description="Status check timestamp")


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = Field(..., description="Whether the request succeeded")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: str = Field(..., description="Response timestamp")


# =============================================================================
# Module State
# =============================================================================

# Track server start time for uptime calculation
_server_start_time: Optional[datetime] = None


def get_server_start_time() -> datetime:
    """Get or initialize server start time."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = datetime.utcnow()
    return _server_start_time


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.utcnow().isoformat() + "Z"


def create_response(
    data: Any = None,
    error: Optional[str] = None,
    success: bool = True
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


@router.get("/api/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the API and its components. This endpoint is
    designed for load balancer health checks and monitoring systems.

    Returns:
        HealthResponse with component status and overall health

    Status Codes:
        200: All components healthy
        200: Some components degraded (status: "degraded")
        503: Critical components unhealthy
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

    # Check if app_state is available via dependency injection
    # For now, we'll check if the state exists on the app
    try:
        app_state = getattr(request.app.state, "app_state", None)
        if app_state:
            if hasattr(app_state, "data_manager") and app_state.data_manager:
                try:
                    components["data_manager"] = await app_state.data_manager.health_check()
                except Exception:
                    pass
            if hasattr(app_state, "money_flow_analyzer") and app_state.money_flow_analyzer:
                components["money_flow_analyzer"] = app_state.money_flow_analyzer.health_check()
            if hasattr(app_state, "institutional_analyzer") and app_state.institutional_analyzer:
                components["institutional_analyzer"] = app_state.institutional_analyzer.health_check()
            if hasattr(app_state, "portfolio_analyzer") and app_state.portfolio_analyzer:
                components["portfolio_analyzer"] = app_state.portfolio_analyzer.health_check()
            if hasattr(app_state, "research_analyzer") and app_state.research_analyzer:
                components["research_analyzer"] = app_state.research_analyzer.health_check()
            if hasattr(app_state, "commodities_analyzer") and app_state.commodities_analyzer:
                components["commodities_analyzer"] = app_state.commodities_analyzer.health_check()
            if hasattr(app_state, "options_analyzer") and app_state.options_analyzer:
                components["options_analyzer"] = app_state.options_analyzer.health_check()
            if hasattr(app_state, "etf_analyzer") and app_state.etf_analyzer:
                components["etf_analyzer"] = app_state.etf_analyzer.health_check()
            if hasattr(app_state, "accounting_analyzer") and app_state.accounting_analyzer:
                components["accounting_analyzer"] = app_state.accounting_analyzer.health_check()
            if hasattr(app_state, "earnings_quality_analyzer") and app_state.earnings_quality_analyzer:
                components["earnings_quality_analyzer"] = True
            if hasattr(app_state, "red_flag_scorer") and app_state.red_flag_scorer:
                components["red_flag_scorer"] = True
            if hasattr(app_state, "anomaly_aggregator") and app_state.anomaly_aggregator:
                components["anomaly_aggregator"] = True
            if hasattr(app_state, "signal_generator") and app_state.signal_generator:
                components["signal_generator"] = app_state.signal_generator.health_check()
    except Exception as e:
        logger.error(f"Health check error: {e}")

    all_healthy = all(components.values())
    api_healthy = components.get("api", False)

    if all_healthy:
        status = "healthy"
    elif api_healthy:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version="2.0.0",
        components=components,
        timestamp=get_timestamp(),
    )


@router.get("/api/version", response_model=ApiResponse)
async def get_version() -> ApiResponse:
    """
    Get version and build information.

    Returns detailed version information about the API, including
    the Python runtime and operating system platform.

    Returns:
        ApiResponse containing VersionInfo
    """
    version_info = VersionInfo(
        version="2.0.0",
        api_version="v1",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.system(),
        build_date=None,  # Can be set from environment variable during CI/CD
        git_commit=None,  # Can be set from environment variable during CI/CD
    )

    return create_response(data=version_info.model_dump())


@router.get("/api/status", response_model=ApiResponse)
async def get_system_status(request: Request) -> ApiResponse:
    """
    Get comprehensive system status.

    Returns detailed system status including version information,
    component health, and performance metrics.

    Returns:
        ApiResponse containing SystemStatus
    """
    # Calculate uptime
    start_time = get_server_start_time()
    uptime_seconds = (datetime.utcnow() - start_time).total_seconds()

    # Version info
    version_info = VersionInfo(
        version="2.0.0",
        api_version="v1",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.system(),
    )

    # Component health checks
    component_health: list[ComponentHealth] = []
    overall_healthy = True

    try:
        app_state = getattr(request.app.state, "app_state", None)

        # API is always healthy if we got here
        component_health.append(ComponentHealth(
            name="api",
            healthy=True,
            details="FastAPI server running"
        ))

        if app_state:
            # Check data manager
            if hasattr(app_state, "data_manager") and app_state.data_manager:
                try:
                    is_healthy = await app_state.data_manager.health_check()
                    component_health.append(ComponentHealth(
                        name="data_manager",
                        healthy=is_healthy,
                        details="OpenBB data adapter" if is_healthy else "Connection issues"
                    ))
                    if not is_healthy:
                        overall_healthy = False
                except Exception as e:
                    component_health.append(ComponentHealth(
                        name="data_manager",
                        healthy=False,
                        details=str(e)
                    ))
                    overall_healthy = False

            # Check analyzers (simplified - just check if initialized)
            analyzers = [
                ("money_flow_analyzer", "Money flow analysis"),
                ("institutional_analyzer", "Institutional holdings"),
                ("portfolio_analyzer", "Portfolio analytics"),
                ("research_analyzer", "Research reports"),
                ("commodities_analyzer", "Commodities data"),
                ("options_analyzer", "Options analytics"),
                ("etf_analyzer", "ETF analytics"),
                ("accounting_analyzer", "SEC filings"),
                ("signal_generator", "Signal generation"),
            ]

            for attr_name, description in analyzers:
                analyzer = getattr(app_state, attr_name, None)
                is_healthy = analyzer is not None
                if hasattr(analyzer, "health_check"):
                    try:
                        is_healthy = analyzer.health_check()
                    except Exception:
                        is_healthy = False

                component_health.append(ComponentHealth(
                    name=attr_name,
                    healthy=is_healthy,
                    details=description if is_healthy else "Not initialized"
                ))
    except Exception as e:
        logger.error(f"Status check error: {e}")
        overall_healthy = False

    # Metrics
    metrics = SystemMetrics(
        uptime_seconds=uptime_seconds,
        request_count=0,  # Would need to track this via middleware
        active_connections=0,
    )

    # Try to get memory usage
    try:
        import resource
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, this is in KB; on macOS, it's in bytes
        if platform.system() == "Darwin":
            metrics.memory_usage_mb = mem_usage / (1024 * 1024)
        else:
            metrics.memory_usage_mb = mem_usage / 1024
    except Exception:
        pass

    status = SystemStatus(
        status="healthy" if overall_healthy else "degraded",
        version=version_info,
        components=component_health,
        metrics=metrics,
        timestamp=get_timestamp(),
    )

    return create_response(data=status.model_dump())


@router.get("/api/ping")
async def ping() -> Dict[str, str]:
    """
    Simple ping endpoint for connectivity testing.

    Returns a minimal response to verify API connectivity.
    Useful for quick health checks from load balancers.

    Returns:
        Simple dict with "pong" response and timestamp
    """
    return {"status": "pong", "timestamp": get_timestamp()}
