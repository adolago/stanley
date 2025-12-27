# Stanley API Versioning and Governance Strategy

## Executive Summary

This document outlines the API versioning strategy, backward compatibility approach, deprecation policy, and governance recommendations for the Stanley institutional investment analysis platform. The strategy is designed to support long-term API evolution while maintaining stability for existing clients, particularly the Rust GUI (`stanley-gui`).

---

## 1. API Versioning Strategy

### Recommendation: URL Path Versioning (Primary) with Header Support (Secondary)

**Chosen Approach: URL Path Versioning**

```
/api/v1/market/{symbol}
/api/v1/institutional/{symbol}
/api/v2/market/{symbol}
```

**Rationale:**

| Factor | URL Versioning | Header Versioning |
|--------|---------------|-------------------|
| Visibility | High - version is explicit in URL | Low - hidden in headers |
| Caching | Excellent - CDN/proxy friendly | Poor - requires Vary header |
| Client Complexity | Low - simple string concatenation | Medium - header manipulation |
| Debugging | Easy - visible in logs/browsers | Harder - requires header inspection |
| API Gateway Support | Excellent | Variable |
| Rust Client Compatibility | Simple URL changes | Requires middleware layer |

**Implementation Plan:**

### Phase 1: Introduce Versioned Routes (Current -> v1)

```python
# stanley/api/main.py - Router structure
from fastapi import APIRouter

# Version 1 router
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

# Legacy router for backward compatibility
legacy_router = APIRouter(prefix="/api", tags=["legacy"])

# Mount both routers
app.include_router(v1_router)
app.include_router(legacy_router)  # Deprecated, will be removed in future
```

### Phase 2: Add Version Header Support (Optional)

```python
# Middleware for header-based version selection
from fastapi import Request

@app.middleware("http")
async def version_middleware(request: Request, call_next):
    # Support Accept header versioning as secondary option
    accept = request.headers.get("Accept", "")
    if "application/vnd.stanley.v2+json" in accept:
        request.state.api_version = "v2"
    else:
        request.state.api_version = "v1"
    return await call_next(request)
```

### Rust Client Updates Required

```rust
// stanley-gui/src/api.rs - Updated client
pub struct StanleyClient {
    base_url: String,
    api_version: String,  // Add version tracking
    client: reqwest::Client,
}

impl StanleyClient {
    pub fn new() -> Self {
        Self::with_version("http://localhost:8000".to_string(), "v1".to_string())
    }

    pub fn with_version(base_url: String, version: String) -> Self {
        Self {
            base_url,
            api_version: version,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_market_data(&self, symbol: &str) -> Result<ApiResponse<MarketData>, ApiError> {
        let url = format!("{}/api/{}/market/{}", self.base_url, self.api_version, symbol);
        // ... rest of implementation
    }
}
```

---

## 2. Backward Compatibility Approach

### Compatibility Levels

| Level | Description | Duration |
|-------|-------------|----------|
| **Full** | Old clients work without modification | Minimum 12 months |
| **Partial** | Core functionality preserved, edge cases may differ | 6-12 months |
| **Breaking** | Requires client updates | Announced 6 months ahead |

### Compatibility Preservation Strategies

#### 2.1 Response Envelope Stability

The current response envelope MUST remain stable:

```python
class ApiResponse(BaseModel):
    """Standard API response wrapper - NEVER CHANGE STRUCTURE"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str
```

#### 2.2 Field Addition Strategy (Non-Breaking)

```python
# v1 response
class MarketDataV1(BaseModel):
    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int
    marketCap: Optional[float] = None
    timestamp: str

# v2 response - adds fields, doesn't remove
class MarketDataV2(MarketDataV1):
    bid: Optional[float] = None
    ask: Optional[float] = None
    lastTradeTime: Optional[str] = None
    extendedHoursPrice: Optional[float] = None
```

#### 2.3 Field Renaming Strategy

```python
# Use aliases for backward compatibility
class InstitutionalHolding(BaseModel):
    manager_name: str = Field(..., alias="managerName")
    shares_held: int = Field(..., alias="sharesHeld")

    class Config:
        populate_by_name = True  # Accept both snake_case and camelCase
```

#### 2.4 Response Transformation Layer

```python
from typing import Callable, Dict, Any

class ResponseTransformer:
    """Transform responses based on API version"""

    transformers: Dict[str, Dict[str, Callable]] = {
        "v1": {
            "market": lambda data: {
                k: v for k, v in data.items()
                if k in ["symbol", "price", "change", "changePercent", "volume", "marketCap", "timestamp"]
            },
        },
        "v2": {
            "market": lambda data: data,  # Full response
        }
    }

    @classmethod
    def transform(cls, version: str, endpoint: str, data: Any) -> Any:
        if version in cls.transformers and endpoint in cls.transformers[version]:
            return cls.transformers[version][endpoint](data)
        return data
```

---

## 3. Deprecation Policy

### Deprecation Lifecycle

```
Active -> Deprecated (6 months) -> Sunset Warning (3 months) -> Removed
```

### Deprecation Markers

#### 3.1 Response Headers

```python
from fastapi import Response

@app.get("/api/market/{symbol}")  # Legacy endpoint
async def get_market_data_legacy(symbol: str, response: Response):
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "2025-12-31T23:59:59Z"
    response.headers["Link"] = '</api/v1/market/{symbol}>; rel="successor-version"'
    # ... rest of handler
```

#### 3.2 Response Body Warnings

```python
class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str
    warnings: Optional[List[str]] = None  # Deprecation warnings

# Usage
return ApiResponse(
    success=True,
    data=market_data,
    timestamp=get_timestamp(),
    warnings=["This endpoint is deprecated. Please migrate to /api/v1/market/{symbol} by 2025-12-31"]
)
```

#### 3.3 OpenAPI Documentation

```python
@app.get(
    "/api/market/{symbol}",
    deprecated=True,  # Mark as deprecated in OpenAPI
    description="**DEPRECATED**: Use /api/v1/market/{symbol} instead. Will be removed 2025-12-31.",
    tags=["Legacy - Deprecated"]
)
async def get_market_data_legacy(symbol: str):
    pass
```

### Deprecation Registry

```python
# stanley/api/deprecation.py
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel

class DeprecationInfo(BaseModel):
    deprecated_at: datetime
    sunset_date: datetime
    successor: Optional[str] = None
    reason: str

DEPRECATION_REGISTRY: Dict[str, DeprecationInfo] = {
    "/api/market/{symbol}": DeprecationInfo(
        deprecated_at=datetime(2025, 1, 1),
        sunset_date=datetime(2025, 12, 31),
        successor="/api/v1/market/{symbol}",
        reason="Migrating to versioned API"
    ),
    # Add more as needed
}
```

---

## 4. Breaking Change Handling

### Definition of Breaking Changes

| Change Type | Breaking? | Example |
|-------------|-----------|---------|
| Remove field | YES | Removing `marketCap` from response |
| Change field type | YES | `volume: int` -> `volume: string` |
| Rename field | YES | `changePercent` -> `change_percent` |
| Add required parameter | YES | New required query param |
| Change error codes | YES | 404 -> 410 for same condition |
| Add optional field | NO | Adding `bid` to response |
| Add optional parameter | NO | New optional query param |
| Add new endpoint | NO | New `/api/v1/screener` |

### Breaking Change Protocol

#### Step 1: Impact Assessment (2 weeks)

```yaml
# docs/breaking-changes/BC-001-field-rename.yaml
id: BC-001
title: Rename camelCase to snake_case in responses
impact:
  endpoints_affected:
    - /api/market/{symbol}
    - /api/institutional/{symbol}
  clients_affected:
    - stanley-gui (Rust)
    - external_integrations
  severity: high
  migration_effort: medium
timeline:
  announcement: 2025-01-15
  v2_available: 2025-02-01
  v1_deprecated: 2025-08-01
  v1_removed: 2026-02-01
```

#### Step 2: Dual Support Period

```python
# Support both versions simultaneously
@v1_router.get("/market/{symbol}")
async def get_market_v1(symbol: str):
    """Returns camelCase response"""
    data = await fetch_market_data(symbol)
    return to_camel_case(data)

@v2_router.get("/market/{symbol}")
async def get_market_v2(symbol: str):
    """Returns snake_case response"""
    data = await fetch_market_data(symbol)
    return data  # Native snake_case
```

#### Step 3: Client Migration Support

```rust
// stanley-gui/src/api.rs - Support both response formats
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum MarketDataResponse {
    V1(MarketDataV1),  // camelCase
    V2(MarketDataV2),  // snake_case
}

impl From<MarketDataResponse> for MarketData {
    fn from(response: MarketDataResponse) -> Self {
        match response {
            MarketDataResponse::V1(v1) => v1.into(),
            MarketDataResponse::V2(v2) => v2.into(),
        }
    }
}
```

---

## 5. API Documentation (OpenAPI)

### OpenAPI Configuration

```python
# stanley/api/main.py
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Stanley API",
        version="1.0.0",
        description="""
# Stanley Institutional Investment Analysis API

## Overview
Stanley provides institutional-grade investment analytics including:
- Market data and real-time quotes
- 13F institutional holdings analysis
- Money flow and dark pool analytics
- Options flow and gamma exposure
- Research and valuation analysis
- Commodity market analysis
- Macro-economic indicators

## Authentication
Currently, the API is designed for local use with the Stanley GUI.
Future versions will include API key authentication.

## Rate Limits
- Development: No limits
- Production: TBD

## Versioning
API versions are specified in the URL path: `/api/v1/...`

## Support
- GitHub: https://github.com/your-org/stanley
- Documentation: https://docs.stanley.io
        """,
        routes=app.routes,
        servers=[
            {"url": "http://localhost:8000", "description": "Local development"},
            {"url": "https://api.stanley.io", "description": "Production (future)"},
        ],
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication (future)"
        }
    }

    # Add tags with descriptions
    openapi_schema["tags"] = [
        {"name": "System", "description": "Health checks and system status"},
        {"name": "Market Data", "description": "Real-time and historical market data"},
        {"name": "Institutional", "description": "13F holdings and institutional analytics"},
        {"name": "Analytics", "description": "Money flow, dark pool, and sector analysis"},
        {"name": "Options", "description": "Options flow, gamma exposure, and unusual activity"},
        {"name": "Research", "description": "Fundamental research and valuation"},
        {"name": "Commodities", "description": "Commodity prices and macro linkages"},
        {"name": "Notes", "description": "Investment notes and thesis management"},
        {"name": "Portfolio", "description": "Portfolio analytics and risk metrics"},
        {"name": "Signals", "description": "Trading signals and backtesting"},
        {"name": "Legacy - Deprecated", "description": "Deprecated endpoints - do not use"},
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### Per-Endpoint Documentation Standards

```python
from typing import List
from pydantic import Field

@app.get(
    "/api/v1/market/{symbol}",
    response_model=ApiResponse,
    tags=["Market Data"],
    summary="Get market data for a symbol",
    description="""
Retrieves current market data for a given stock symbol.

## Response Fields

| Field | Type | Description |
|-------|------|-------------|
| symbol | string | Stock ticker symbol |
| price | float | Current price |
| change | float | Dollar change from previous close |
| changePercent | float | Percentage change from previous close |
| volume | int | Trading volume |
| marketCap | float | Market capitalization (nullable) |

## Example Response

```json
{
    "success": true,
    "data": {
        "symbol": "AAPL",
        "price": 185.50,
        "change": 2.30,
        "changePercent": 1.25,
        "volume": 45000000,
        "marketCap": 2900000000000
    },
    "timestamp": "2025-01-15T14:30:00Z"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 404 | Symbol not found |
| 503 | Data manager not initialized |
    """,
    responses={
        200: {
            "description": "Successful response with market data",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "symbol": "AAPL",
                            "price": 185.50,
                            "change": 2.30,
                            "changePercent": 1.25,
                            "volume": 45000000,
                            "marketCap": 2900000000000
                        },
                        "timestamp": "2025-01-15T14:30:00Z"
                    }
                }
            }
        },
        404: {
            "description": "Symbol not found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "error": "No data found for INVALID",
                        "timestamp": "2025-01-15T14:30:00Z"
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable - data manager not initialized"
        }
    }
)
async def get_market_data(
    symbol: str = Path(
        ...,
        description="Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
        example="AAPL",
        min_length=1,
        max_length=10
    )
):
    """Get market data for a symbol."""
    pass
```

---

## 6. Client SDK Generation

### OpenAPI-Based SDK Generation

#### Python SDK (Auto-generated)

```bash
# Generate Python client
pip install openapi-python-client

openapi-python-client generate \
    --url http://localhost:8000/openapi.json \
    --output-path ./sdk/python \
    --config sdk-config.yaml
```

```yaml
# sdk-config.yaml
project_name_override: stanley-client
package_name_override: stanley
```

#### Rust SDK (Manual with Type Sharing)

```rust
// sdk/rust/src/lib.rs
//! Stanley API Client for Rust
//!
//! Auto-generated types with handcrafted client logic

pub mod types;
pub mod client;
pub mod error;

pub use client::StanleyClient;
pub use error::ApiError;

// Re-export all types
pub use types::*;
```

```rust
// sdk/rust/src/types.rs
//! API types - keep in sync with OpenAPI schema

use serde::{Deserialize, Serialize};

/// Standard API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: String,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Market data response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub change: f64,
    pub change_percent: f64,
    pub volume: i64,
    pub market_cap: Option<f64>,
    pub timestamp: String,
}

// ... more types
```

#### TypeScript SDK (For Future Web Clients)

```bash
# Generate TypeScript client
npx @openapitools/openapi-generator-cli generate \
    -i http://localhost:8000/openapi.json \
    -g typescript-fetch \
    -o ./sdk/typescript \
    --additional-properties=supportsES6=true,npmName=stanley-client
```

### SDK Versioning

```
sdk-python/
  stanley-client-1.0.0/  # For API v1
  stanley-client-2.0.0/  # For API v2 (when available)

sdk-rust/
  stanley-client-1.0.0/
  stanley-client-2.0.0/
```

---

## 7. API Changelog

### Changelog Format (CHANGELOG-API.md)

```markdown
# Stanley API Changelog

All notable changes to the Stanley API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

## [1.0.0] - 2025-01-15

### Added
- Initial versioned API release
- `/api/v1/health` - Health check endpoint
- `/api/v1/market/{symbol}` - Market data endpoint
- `/api/v1/institutional/{symbol}` - 13F holdings
- `/api/v1/institutional/{symbol}/changes` - 13F change detection
- `/api/v1/money-flow` - Sector money flow analysis
- `/api/v1/portfolio-analytics` - Portfolio VaR and beta
- `/api/v1/dark-pool/{symbol}` - Dark pool activity
- `/api/v1/equity-flow/{symbol}` - Equity money flow
- `/api/v1/research/{symbol}` - Research reports
- `/api/v1/valuation/{symbol}` - Valuation analysis
- `/api/v1/earnings/{symbol}` - Earnings analysis
- `/api/v1/peers/{symbol}` - Peer comparison
- `/api/v1/commodities` - Commodity overview
- `/api/v1/commodities/{symbol}` - Commodity detail
- `/api/v1/commodities/{symbol}/macro` - Macro linkages
- `/api/v1/commodities/correlations` - Correlation matrix
- `/api/v1/options/{symbol}/flow` - Options flow
- `/api/v1/options/{symbol}/gamma` - Gamma exposure
- `/api/v1/options/{symbol}/unusual` - Unusual activity
- `/api/v1/options/{symbol}/put-call` - Put/call analysis
- `/api/v1/options/{symbol}/smart-money` - Smart money trades
- `/api/v1/options/{symbol}/max-pain` - Max pain analysis
- `/api/v1/notes` - Notes CRUD operations
- `/api/v1/theses` - Investment theses
- `/api/v1/trades` - Trade journal
- `/api/v1/events` - Event notes
- `/api/v1/people` - Executive profiles
- `/api/v1/sectors` - Sector notes
- `/api/v1/signals/*` - Signal generation and backtesting
- `/api/v1/accounting/*` - SEC filings and accounting analysis

### Deprecated
- `/api/*` (non-versioned) - Use `/api/v1/*` instead
  - Sunset date: 2025-12-31

### Security
- Added CORS configuration for Tauri development
- SEC identity configuration for EDGAR API access

---

## Migration Guides

### Migrating from Non-Versioned to v1

1. Update all API calls to include `/v1/` after `/api/`
2. Example: `/api/market/AAPL` -> `/api/v1/market/AAPL`
3. No response format changes in v1

---

## API Stability Guarantees

### v1.x
- Response envelope structure is frozen
- Existing fields will not be removed
- New optional fields may be added
- Error codes will remain consistent
```

### Automated Changelog Updates

```python
# scripts/update_changelog.py
import yaml
from datetime import datetime

def add_changelog_entry(change_type: str, description: str, endpoints: list = None):
    """Add entry to API changelog"""
    entry = {
        "date": datetime.now().isoformat(),
        "type": change_type,
        "description": description,
        "endpoints": endpoints or []
    }

    # Append to CHANGELOG-API.yaml for tracking
    with open("docs/CHANGELOG-API.yaml", "a") as f:
        yaml.dump([entry], f)
```

---

## 8. Migration Guides

### Guide Template

```markdown
# Migration Guide: v1 to v2

## Overview
This guide covers migrating from Stanley API v1 to v2.

**Timeline:**
- v2 Available: 2025-06-01
- v1 Deprecated: 2025-12-01
- v1 Removed: 2026-06-01

## Breaking Changes

### 1. Response Field Naming Convention

**v1 (camelCase):**
```json
{
  "changePercent": 1.25,
  "marketCap": 2900000000000
}
```

**v2 (snake_case):**
```json
{
  "change_percent": 1.25,
  "market_cap": 2900000000000
}
```

### Migration Steps

#### Python Clients

```python
# Before (v1)
response = client.get_market_data("AAPL")
print(response.data["changePercent"])

# After (v2)
response = client.get_market_data("AAPL")
print(response.data["change_percent"])
```

#### Rust Clients (stanley-gui)

```rust
// Before (v1)
#[derive(Deserialize)]
pub struct MarketData {
    #[serde(rename = "changePercent")]
    pub change_percent: f64,
}

// After (v2)
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MarketData {
    pub change_percent: f64,
}
```

## New Features in v2

### 1. Extended Market Data
- `bid` and `ask` prices
- `last_trade_time` timestamp
- `extended_hours_price` for pre/post market

### 2. Pagination Support
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 150,
    "total_pages": 3
  }
}
```

## Compatibility Mode

During the transition period, you can request v1-style responses from v2 endpoints:

```http
GET /api/v2/market/AAPL
Accept: application/vnd.stanley.v1+json
```

## Testing Your Migration

```bash
# Test v1 compatibility
curl http://localhost:8000/api/v1/market/AAPL

# Test v2
curl http://localhost:8000/api/v2/market/AAPL

# Compare responses
diff <(curl -s http://localhost:8000/api/v1/market/AAPL | jq -S) \
     <(curl -s http://localhost:8000/api/v2/market/AAPL | jq -S)
```

## Support

- Migration issues: [GitHub Issues](https://github.com/your-org/stanley/issues)
- Timeline questions: api-support@stanley.io
```

---

## 9. Feature Flags for API

### Feature Flag System

```python
# stanley/api/feature_flags.py
from enum import Enum
from typing import Dict, Set
from functools import wraps
from fastapi import HTTPException, Request

class ApiFeature(Enum):
    """API feature flags"""
    REAL_TIME_STREAMING = "real_time_streaming"
    ADVANCED_ANALYTICS = "advanced_analytics"
    BATCH_REQUESTS = "batch_requests"
    WEBHOOKS = "webhooks"
    GRAPHQL = "graphql"
    RATE_LIMITING = "rate_limiting"
    API_KEYS = "api_keys"
    AUDIT_LOGGING = "audit_logging"

class FeatureFlags:
    """Feature flag management"""

    _enabled: Set[ApiFeature] = {
        ApiFeature.REAL_TIME_STREAMING,  # Currently enabled
        ApiFeature.ADVANCED_ANALYTICS,
    }

    _beta_users: Dict[str, Set[ApiFeature]] = {
        # API keys that have access to beta features
        "beta-key-001": {ApiFeature.BATCH_REQUESTS, ApiFeature.GRAPHQL},
    }

    @classmethod
    def is_enabled(cls, feature: ApiFeature, api_key: str = None) -> bool:
        """Check if feature is enabled globally or for specific user"""
        if feature in cls._enabled:
            return True
        if api_key and api_key in cls._beta_users:
            return feature in cls._beta_users[api_key]
        return False

    @classmethod
    def enable(cls, feature: ApiFeature):
        """Enable a feature globally"""
        cls._enabled.add(feature)

    @classmethod
    def disable(cls, feature: ApiFeature):
        """Disable a feature globally"""
        cls._enabled.discard(feature)

def require_feature(feature: ApiFeature):
    """Decorator to require a feature flag"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            api_key = request.headers.get("X-API-Key") if request else None
            if not FeatureFlags.is_enabled(feature, api_key):
                raise HTTPException(
                    status_code=403,
                    detail=f"Feature '{feature.value}' is not enabled"
                )
            return await func(*args, request=request, **kwargs)
        return wrapper
    return decorator
```

### Feature-Flagged Endpoint Example

```python
from stanley.api.feature_flags import require_feature, ApiFeature

@app.post(
    "/api/v1/batch",
    response_model=ApiResponse,
    tags=["Advanced"],
    summary="Batch multiple requests (Beta)"
)
@require_feature(ApiFeature.BATCH_REQUESTS)
async def batch_requests(request: Request, batch: BatchRequest):
    """
    Execute multiple API requests in a single call.

    **Note:** This feature is in beta and requires feature flag access.
    """
    pass
```

### Feature Discovery Endpoint

```python
@app.get("/api/v1/features", response_model=ApiResponse, tags=["System"])
async def get_features(request: Request):
    """
    List available API features and their status.
    """
    api_key = request.headers.get("X-API-Key")

    features = {}
    for feature in ApiFeature:
        features[feature.value] = {
            "enabled": FeatureFlags.is_enabled(feature, api_key),
            "beta": feature not in FeatureFlags._enabled,
        }

    return create_response(data=features)
```

---

## 10. Rate Limiting Tiers

### Rate Limiting Architecture

```python
# stanley/api/rate_limiting.py
from enum import Enum
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

class RateTier(Enum):
    """API rate limiting tiers"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"  # For stanley-gui

class RateLimits:
    """Rate limit configurations per tier"""

    LIMITS: Dict[RateTier, Dict[str, int]] = {
        RateTier.FREE: {
            "requests_per_minute": 10,
            "requests_per_hour": 100,
            "requests_per_day": 500,
            "max_batch_size": 5,
            "max_symbols_per_request": 10,
        },
        RateTier.BASIC: {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "max_batch_size": 20,
            "max_symbols_per_request": 50,
        },
        RateTier.PROFESSIONAL: {
            "requests_per_minute": 300,
            "requests_per_hour": 10000,
            "requests_per_day": 100000,
            "max_batch_size": 100,
            "max_symbols_per_request": 200,
        },
        RateTier.ENTERPRISE: {
            "requests_per_minute": 1000,
            "requests_per_hour": 50000,
            "requests_per_day": 1000000,
            "max_batch_size": 500,
            "max_symbols_per_request": 500,
        },
        RateTier.INTERNAL: {
            "requests_per_minute": -1,  # Unlimited
            "requests_per_hour": -1,
            "requests_per_day": -1,
            "max_batch_size": 1000,
            "max_symbols_per_request": 1000,
        },
    }

    # Endpoint-specific rate limits (override tier limits)
    ENDPOINT_LIMITS: Dict[str, Dict[RateTier, int]] = {
        "/api/v1/research/{symbol}": {
            RateTier.FREE: 5,  # Per minute - expensive operation
            RateTier.BASIC: 30,
            RateTier.PROFESSIONAL: 100,
        },
        "/api/v1/signals/generate": {
            RateTier.FREE: 1,
            RateTier.BASIC: 10,
            RateTier.PROFESSIONAL: 50,
        },
    }

class RateLimiter:
    """In-memory rate limiter (replace with Redis for production)"""

    def __init__(self):
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(
        self,
        identifier: str,
        tier: RateTier,
        endpoint: Optional[str] = None
    ) -> tuple[bool, Dict[str, any]]:
        """
        Check if request is allowed under rate limits.

        Returns:
            (is_allowed, rate_info)
        """
        async with self._lock:
            now = datetime.now()
            limits = RateLimits.LIMITS[tier]

            # Clean old requests
            self._requests[identifier] = [
                req for req in self._requests[identifier]
                if now - req < timedelta(days=1)
            ]

            requests = self._requests[identifier]

            # Check limits
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)

            requests_minute = sum(1 for r in requests if r > minute_ago)
            requests_hour = sum(1 for r in requests if r > hour_ago)
            requests_day = len(requests)

            limit_minute = limits["requests_per_minute"]
            limit_hour = limits["requests_per_hour"]
            limit_day = limits["requests_per_day"]

            # Check endpoint-specific limits
            if endpoint and endpoint in RateLimits.ENDPOINT_LIMITS:
                endpoint_limits = RateLimits.ENDPOINT_LIMITS[endpoint]
                if tier in endpoint_limits:
                    limit_minute = min(limit_minute, endpoint_limits[tier])

            rate_info = {
                "tier": tier.value,
                "requests_minute": requests_minute,
                "limit_minute": limit_minute,
                "requests_hour": requests_hour,
                "limit_hour": limit_hour,
                "requests_day": requests_day,
                "limit_day": limit_day,
                "reset_minute": (minute_ago + timedelta(minutes=1)).isoformat(),
                "reset_hour": (hour_ago + timedelta(hours=1)).isoformat(),
            }

            # Unlimited for internal tier
            if limit_minute == -1:
                self._requests[identifier].append(now)
                return True, rate_info

            # Check limits
            if requests_minute >= limit_minute:
                rate_info["error"] = "Rate limit exceeded (per minute)"
                return False, rate_info
            if requests_hour >= limit_hour:
                rate_info["error"] = "Rate limit exceeded (per hour)"
                return False, rate_info
            if requests_day >= limit_day:
                rate_info["error"] = "Rate limit exceeded (per day)"
                return False, rate_info

            self._requests[identifier].append(now)
            return True, rate_info

# Global rate limiter instance
rate_limiter = RateLimiter()
```

### Rate Limiting Middleware

```python
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limiting to API requests"""

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        # Determine tier from API key or default
        api_key = request.headers.get("X-API-Key")
        if api_key:
            tier = await self._get_tier_for_key(api_key)
            identifier = api_key
        else:
            # Default tier for unauthenticated requests
            # Internal requests from stanley-gui get special treatment
            if self._is_internal_request(request):
                tier = RateTier.INTERNAL
                identifier = "internal"
            else:
                tier = RateTier.FREE
                identifier = request.client.host

        # Check rate limit
        allowed, rate_info = await rate_limiter.is_allowed(
            identifier,
            tier,
            request.url.path
        )

        if not allowed:
            return Response(
                content=json.dumps({
                    "success": False,
                    "error": rate_info["error"],
                    "rate_limit": rate_info,
                    "timestamp": datetime.now().isoformat()
                }),
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit_minute"]),
                    "X-RateLimit-Remaining": str(max(0, rate_info["limit_minute"] - rate_info["requests_minute"])),
                    "X-RateLimit-Reset": rate_info["reset_minute"],
                    "Retry-After": "60",
                }
            )

        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit_minute"])
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, rate_info["limit_minute"] - rate_info["requests_minute"])
        )
        response.headers["X-RateLimit-Reset"] = rate_info["reset_minute"]

        return response

    def _is_internal_request(self, request: Request) -> bool:
        """Check if request is from internal client (stanley-gui)"""
        origin = request.headers.get("Origin", "")
        user_agent = request.headers.get("User-Agent", "")

        internal_origins = [
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "tauri://localhost",
            "https://tauri.localhost",
        ]

        return (
            origin in internal_origins or
            "Stanley-GUI" in user_agent or
            "Tauri" in user_agent
        )

    async def _get_tier_for_key(self, api_key: str) -> RateTier:
        """Look up tier for API key (would query database in production)"""
        # Placeholder - would query database
        tier_mapping = {
            "demo-free-key": RateTier.FREE,
            "demo-basic-key": RateTier.BASIC,
            "demo-pro-key": RateTier.PROFESSIONAL,
            "demo-enterprise-key": RateTier.ENTERPRISE,
        }
        return tier_mapping.get(api_key, RateTier.FREE)

# Add middleware to app
# app.add_middleware(RateLimitMiddleware)
```

### Rate Limit Status Endpoint

```python
@app.get("/api/v1/rate-limit", response_model=ApiResponse, tags=["System"])
async def get_rate_limit_status(request: Request):
    """
    Get current rate limit status for the requesting client.
    """
    api_key = request.headers.get("X-API-Key")

    if api_key:
        tier = await rate_limiter._get_tier_for_key(api_key)
        identifier = api_key
    else:
        identifier = request.client.host
        tier = RateTier.FREE

    _, rate_info = await rate_limiter.is_allowed(identifier, tier)

    return create_response(data={
        "tier": tier.value,
        "limits": RateLimits.LIMITS[tier],
        "current_usage": {
            "requests_minute": rate_info["requests_minute"],
            "requests_hour": rate_info["requests_hour"],
            "requests_day": rate_info["requests_day"],
        },
        "remaining": {
            "minute": max(0, rate_info["limit_minute"] - rate_info["requests_minute"]),
            "hour": max(0, rate_info["limit_hour"] - rate_info["requests_hour"]),
            "day": max(0, rate_info["limit_day"] - rate_info["requests_day"]),
        },
        "reset": {
            "minute": rate_info["reset_minute"],
            "hour": rate_info["reset_hour"],
        }
    })
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create versioned router structure (`/api/v1/`)
- [ ] Add deprecation headers to legacy endpoints
- [ ] Enhance OpenAPI documentation
- [ ] Set up API changelog

### Phase 2: Client Updates (Weeks 3-4)
- [ ] Update `stanley-gui` Rust client to use versioned endpoints
- [ ] Generate Python SDK from OpenAPI
- [ ] Create migration guide for external users

### Phase 3: Governance (Weeks 5-6)
- [ ] Implement feature flag system
- [ ] Add rate limiting middleware
- [ ] Create deprecation registry
- [ ] Set up API health monitoring

### Phase 4: Advanced Features (Weeks 7-8)
- [ ] API key authentication system
- [ ] Usage analytics and metering
- [ ] Webhook support for async operations
- [ ] GraphQL gateway (optional)

---

## Summary of Recommendations

| Area | Recommendation | Priority |
|------|----------------|----------|
| Versioning | URL path versioning (`/api/v1/`) | High |
| Compatibility | Add fields, never remove in minor versions | High |
| Deprecation | 12-month deprecation cycle with headers | High |
| Documentation | Enhanced OpenAPI with examples | High |
| SDK Generation | Auto-generate Python, manual Rust | Medium |
| Feature Flags | Implement for beta features | Medium |
| Rate Limiting | Tier-based with internal bypass | Medium |
| Changelog | Automated updates on each release | Low |

---

*Document Version: 1.0.0*
*Last Updated: 2025-01-15*
*Author: Stanley API Team*
