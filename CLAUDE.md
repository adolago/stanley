# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stanley is an institutional investment analysis platform focused on money flow analysis, institutional positioning, and fundamental research. The platform explicitly avoids technical indicators (RSI, MACD, Fibonacci, etc.) in favor of institutional data sources like 13F filings, dark pool activity, and options flow.

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run a single test
pytest tests/test_core.py::test_basic -v

# Linting
flake8 stanley/

# Format code
black stanley/

# Type checking
mypy stanley/
```

## Architecture

### Core Components

- **`stanley/core/core.py`**: Main `Stanley` class that coordinates all functionality and provides the primary API entry point
- **`stanley/data/data_manager.py`**: `DataManager` class handles all data fetching from external sources (OpenBB, SEC, options flow providers). Uses async methods for data retrieval.
- **`stanley/analytics/money_flow.py`**: `MoneyFlowAnalyzer` for sector and equity money flow analysis, dark pool activity tracking
- **`stanley/analytics/institutional.py`**: `InstitutionalAnalyzer` for 13F filing analysis, institutional sentiment, and smart money tracking

### Data Flow

1. `Stanley` class is the main entry point
2. Analytics modules (`MoneyFlowAnalyzer`, `InstitutionalAnalyzer`) receive a `DataManager` instance
3. `DataManager` fetches from external APIs (OpenBB, SEC filings, options flow providers)
4. Results are returned as pandas DataFrames or dictionaries

### Configuration

Configuration is in `config/stanley.yaml` with sections for:
- Data source API keys (OpenBB, options flow, dark pool providers)
- Database connections (PostgreSQL, Redis)
- Risk parameters (position limits, VaR settings, stress scenarios)
- Analytics settings (lookback periods, thresholds)

### Module Status

Active/implemented:
- `core/` - Main Stanley class
- `analytics/` - Money flow and institutional analysis
- `data/` - Data management layer

Placeholder/scaffolded:
- `api/` - FastAPI endpoints (not implemented)
- `portfolio/` - Portfolio analytics (not implemented)
- `research/` - Fundamental research tools (not implemented)

## Key Design Principles

- No technical indicators or chart patterns - focus on institutional data
- Data sources: SEC filings (13F, Forms 4), ETF flows, dark pool volume, options flow
- All analytics methods return pandas DataFrames or typed dictionaries
- Async methods in DataManager for external API calls

## Rust GUI (stanley-gui)

A GPUI-based graphical interface for Stanley using Zed's UI framework.

### Building the GUI

```bash
cd stanley-gui
cargo build --release
cargo run
```

### GUI Architecture

- `src/main.rs` - Application entry point and window setup
- `src/app.rs` - Main `StanleyApp` state and rendering logic
- `src/theme.rs` - Dark/light theme color definitions
- `src/components/` - Reusable UI components (sidebar, charts, tables)
- `src/api.rs` - HTTP client for Python backend communication

### GPUI Resources

- [GPUI Documentation](https://www.gpui.rs/)
- [Zed GPUI Examples](https://github.com/zed-industries/zed/tree/main/crates/gpui/examples)
