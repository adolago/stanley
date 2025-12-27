# Enhanced Money Flow Analysis Architecture

> **Related Documents**: See [System Architecture](./system_architecture.md) for overall platform architecture, [ML Architecture](../ml_architecture_roadmap.md) for ML integration plans.

## Executive Summary

This document outlines the architecture for enhancing `stanley/analytics/money_flow.py` to provide Bloomberg-competitive money flow analysis capabilities.

## Current State Analysis

The existing `MoneyFlowAnalyzer` class provides:
- `analyze_sector_flow()` - Basic sector ETF flow analysis
- `analyze_equity_flow()` - Basic equity money flow metrics
- `get_dark_pool_activity()` - Placeholder dark pool analysis

## Enhancement Objectives

1. **Real-time Dark Pool Alerts** - Detect significant dark pool activity patterns
2. **Block Trade Detection** - Identify and classify large institutional trades
3. **Sector Rotation Signals** - Generate economic cycle-aware rotation signals
4. **Smart Money Tracking** - Aggregate institutional manager activity
5. **Unusual Volume Detection** - Identify abnormal volume patterns
6. **Flow Momentum Indicators** - Calculate normalized flow momentum

## Architecture Overview

```
                    +---------------------------+
                    |  EnhancedMoneyFlowAnalyzer |
                    +---------------------------+
                              |
         +--------------------+--------------------+
         |                    |                    |
    +---------+         +---------+         +---------+
    | Dark    |         | Block   |         | Sector  |
    | Pool    |         | Trade   |         | Rotation|
    | Module  |         | Module  |         | Module  |
    +---------+         +---------+         +---------+
         |                    |                    |
         +--------------------+--------------------+
                              |
                    +------------------+
                    |   DataManager    |
                    +------------------+
                              |
              +---------------+---------------+
              |               |               |
         +---------+    +---------+    +---------+
         | OpenBB  |    | Dark    |    | 13F     |
         | Adapter |    | Pool    |    | Data    |
         +---------+    | Source  |    +---------+
                        +---------+
```

## Data Structures

### Alert Configuration (`AlertConfig`)

Centralized configuration for all alert thresholds:

```python
@dataclass
class AlertConfig:
    # Dark Pool Thresholds
    dark_pool_volume_zscore: float = 2.0
    dark_pool_percentage_high: float = 0.40

    # Block Trade Thresholds
    block_trade_min_value_usd: float = 1_000_000.0
    block_trade_min_shares: int = 10_000

    # Volume Thresholds
    unusual_volume_multiplier: float = 2.5

    # Smart Money Thresholds
    smart_money_min_aum: float = 1_000_000_000

    # Flow Momentum Thresholds
    flow_momentum_breakout_zscore: float = 2.0
```

### Alert Types

```python
class AlertType(Enum):
    DARK_POOL_SURGE = auto()
    BLOCK_TRADE = auto()
    UNUSUAL_VOLUME = auto()
    SECTOR_ROTATION = auto()
    SMART_MONEY_ACCUMULATION = auto()
    SMART_MONEY_DISTRIBUTION = auto()
    FLOW_MOMENTUM_BREAKOUT = auto()
    FLOW_DIVERGENCE = auto()
```

### Block Trade Classification

```python
class BlockTradeType(Enum):
    ACCUMULATION = auto()   # Large buy, minimal impact
    DISTRIBUTION = auto()   # Large sell, minimal impact
    MOMENTUM = auto()       # Trade aligned with trend
    CONTRARIAN = auto()     # Trade against trend
    CROSS = auto()          # Matched institutional trade
    ICEBERG = auto()        # Hidden order detected
    SWEEP = auto()          # Aggressive order
```

## Key Methods

### Dark Pool Analysis

| Method | Description | Returns |
|--------|-------------|---------|
| `analyze_dark_pool_realtime()` | Real-time dark pool monitoring | Dict with metrics |
| `get_dark_pool_alerts()` | Get dark pool alerts | List[MoneyFlowAlert] |
| `detect_dark_pool_accumulation()` | Detect stealth accumulation | Optional[Alert] |

### Block Trade Detection

| Method | Description | Returns |
|--------|-------------|---------|
| `detect_block_trades()` | Detect and classify block trades | List[BlockTrade] |
| `classify_block_trade()` | Classify single block trade | BlockTradeType |
| `get_block_trade_summary()` | Summary statistics | Dict |

### Sector Rotation

| Method | Description | Returns |
|--------|-------------|---------|
| `generate_sector_rotation_signal()` | Generate rotation signal | SectorRotationSignal |
| `determine_rotation_phase()` | Determine economic phase | SectorRotationPhase |
| `get_sector_rotation_alerts()` | Get rotation alerts | List[MoneyFlowAlert] |

### Smart Money Tracking

| Method | Description | Returns |
|--------|-------------|---------|
| `track_smart_money()` | Track for single symbol | SmartMoneyActivity |
| `aggregate_smart_money_flow()` | Aggregate across universe | DataFrame |
| `get_smart_money_conviction_picks()` | High conviction picks | List[Dict] |

### Unusual Volume

| Method | Description | Returns |
|--------|-------------|---------|
| `detect_unusual_volume()` | Detect for single symbol | Optional[Event] |
| `scan_unusual_volume()` | Scan universe | List[Event] |
| `correlate_volume_with_flow()` | Correlation analysis | Dict |

### Flow Momentum

| Method | Description | Returns |
|--------|-------------|---------|
| `calculate_flow_momentum()` | Calculate momentum | FlowMomentumIndicator |
| `detect_flow_divergence()` | Detect price/flow divergence | Optional[Alert] |
| `get_flow_momentum_leaders()` | Top momentum stocks | DataFrame |

## API Endpoints

New REST endpoints:

```
GET /api/money-flow/dark-pool/{symbol}
GET /api/money-flow/dark-pool/{symbol}/alerts
GET /api/money-flow/block-trades/{symbol}
GET /api/money-flow/sector-rotation
GET /api/money-flow/smart-money/{symbol}
GET /api/money-flow/smart-money/conviction
GET /api/money-flow/unusual-volume
GET /api/money-flow/momentum/{symbol}
GET /api/money-flow/momentum/leaders
GET /api/money-flow/comprehensive/{symbol}
GET /api/money-flow/dashboard
GET /api/money-flow/alerts
WebSocket /ws/money-flow/alerts
```

## Integration with DataManager

Required new methods in `DataManager`:

```python
async def get_realtime_trades(symbol, since) -> pd.DataFrame
async def get_dark_pool_prints(symbol, lookback_days) -> pd.DataFrame
async def get_institutional_flow(symbol, lookback_days) -> pd.DataFrame
async def get_smart_money_managers(min_aum) -> pd.DataFrame
```

## Event-Driven Alert Pattern

```
┌─────────────────┐
│ Alert Generated │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AlertEventBus  │
└────────┬────────┘
         │
    ┌────┴────┬─────────────┐
    │         │             │
    ▼         ▼             ▼
┌───────┐ ┌───────┐    ┌───────┐
│Webhook│ │WebSock│    │ Log   │
│Handler│ │Stream │    │Handler│
└───────┘ └───────┘    └───────┘
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- Data structures (dataclasses, enums)
- AlertConfig and threshold system
- Basic alert generation
- Integration with existing MoneyFlowAnalyzer

### Phase 2: Detection Algorithms (Week 2)
- Unusual volume detection
- Block trade detection/classification
- Dark pool analysis enhancement
- Flow momentum calculation

### Phase 3: Advanced Analysis (Week 3)
- Sector rotation signals
- Smart money tracking
- Flow divergence detection
- Comprehensive analysis methods

### Phase 4: Real-time Capabilities (Week 4)
- Alert event bus and streaming
- WebSocket endpoint
- Real-time monitoring loop
- Alert handlers

### Phase 5: API and Integration (Week 5)
- New API endpoints
- DataManager integration
- GUI integration
- Testing and documentation

## Dependencies

**Required:**
- pandas >= 1.5.0
- numpy >= 1.24.0
- asyncio (stdlib)
- dataclasses (stdlib)

**External Data Sources:**
- Dark pool data: FINRA ADF, IEX
- Block trade data: TAQ or consolidated tape
- 13F data: SEC EDGAR (already integrated)
- Real-time quotes: OpenBB

## Files Modified/Created

**Modified:**
- `/home/artur/Repositories/stanley/stanley/analytics/money_flow.py` - Add new methods
- `/home/artur/Repositories/stanley/stanley/data/data_manager.py` - Add data methods
- `/home/artur/Repositories/stanley/stanley/api/main.py` - Add endpoints

**Created:**
- `/home/artur/Repositories/stanley/stanley/analytics/alerts.py` - Alert system
- `/home/artur/Repositories/stanley/stanley/analytics/block_trades.py` - Block trade logic
- `/home/artur/Repositories/stanley/stanley/analytics/sector_rotation.py` - Rotation logic
