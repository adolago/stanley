# Stanley GUI

GPUI-based graphical interface for the Stanley institutional investment analysis platform.

## Overview

Stanley GUI provides a native desktop interface for Stanley, built using [GPUI](https://github.com/zed-industries/zed) - the high-performance GPU-accelerated UI framework from the Zed editor team.

## Features

- Real-time market data display
- Portfolio analytics visualization
- Money flow analysis dashboards
- Institutional holdings explorer
- Options flow and gamma analysis
- ETF sector rotation tracking
- Commodities and macro analysis
- Research note management
- Comprehensive keyboard navigation
- Vim-style shortcuts support

## Requirements

- Rust 1.70+
- Linux with Wayland (X11 not supported)
- Running Stanley Python backend on port 8000

## Building

```bash
cd stanley-gui
cargo build --release
```

The binary will be located at `target/release/stanley-gui`.

**Note**: This build requires Wayland. X11 support has been removed.

## Running

1. Start the Stanley Python backend:
```bash
cd ..
python -m stanley.api.main
# or
uvicorn stanley.api.main:app --port 8000
```

2. Run the GUI:
```bash
./target/release/stanley-gui
```

## Architecture

```
stanley-gui/
├── src/
│   ├── main.rs           # Application entry point
│   ├── app.rs            # Main application state and rendering
│   ├── api.rs            # HTTP client for Python backend
│   ├── navigation.rs     # View routing and navigation system (40+ views)
│   ├── keyboard.rs       # Keyboard shortcuts and input handling
│   ├── theme.rs          # Theme configuration (dark/light)
│   ├── dashboard.rs      # Dashboard view implementation
│   ├── portfolio.rs      # Portfolio analytics view
│   ├── commodities.rs    # Commodities market view
│   ├── comparison.rs     # Asset comparison view
│   ├── settings.rs       # Settings and preferences view
│   └── components/       # Reusable UI components
│       ├── mod.rs
│       ├── header.rs     # Top navigation header
│       ├── sidebar.rs    # Navigation sidebar
│       ├── tables.rs     # Data tables
│       ├── charts.rs     # Chart components
│       ├── modals.rs     # Modal dialogs
│       ├── dashboard.rs  # Dashboard widgets
│       └── forms/        # Form components
│           ├── mod.rs
│           ├── validation.rs
│           ├── text_input.rs
│           └── number_input.rs
├── Cargo.toml            # Dependencies
└── README.md
```

## Views

The GUI implements 40+ views organized into navigation sections:

### Markets & Overview
| View | Description |
|------|-------------|
| Dashboard | Main overview with key metrics |
| Watchlist | Custom symbol watchlists |

### Flow Analytics
| View | Description |
|------|-------------|
| MoneyFlow | Sector money flow analysis |
| EquityFlow | Individual equity flow tracking |
| DarkPool | Dark pool activity monitoring |

### Institutional
| View | Description |
|------|-------------|
| Institutional | Institutional holdings overview |
| ThirteenF | 13F filing analysis |

### Options
| View | Description |
|------|-------------|
| OptionsFlow | Options order flow |
| OptionsGamma | Gamma exposure analysis |
| OptionsUnusual | Unusual options activity |
| MaxPain | Max pain calculations |

### Research
| View | Description |
|------|-------------|
| Research | Comprehensive research reports |
| Valuation | DCF and valuation models |
| Earnings | Earnings analysis and history |
| Peers | Peer comparison |

### Portfolio
| View | Description |
|------|-------------|
| Portfolio | Holdings and performance |
| PortfolioRisk | VaR and risk metrics |
| Sectors | Sector exposure analysis |

### Macro & Commodities
| View | Description |
|------|-------------|
| Macro | Macroeconomic indicators |
| Commodities | Commodity market overview |
| CommodityCorrelations | Correlation analysis |

### ETF Analytics
| View | Description |
|------|-------------|
| ETFOverview | ETF market overview |
| ETFFlows | ETF flow tracking |
| SectorRotation | Sector rotation signals |
| FactorRotation | Factor rotation analysis |
| Thematic | Thematic ETF trends |

### Accounting & Filings
| View | Description |
|------|-------------|
| Accounting | SEC filings overview |
| FinancialStatements | Balance sheet, income, cash flow |
| EarningsQuality | M-score, F-score, Z-score |
| RedFlags | Accounting red flags detection |

### Signals & Alerts
| View | Description |
|------|-------------|
| Signals | Trading signals dashboard |
| SignalBacktest | Backtest results |
| SignalPerformance | Signal performance tracking |

### Notes & Research
| View | Description |
|------|-------------|
| Notes | Research notes management |
| Theses | Investment thesis tracker |
| Trades | Trade journal |
| Events | Event calendar |

### Comparison
| View | Description |
|------|-------------|
| Comparison | Multi-asset comparison tool |

### Settings
| View | Description |
|------|-------------|
| Settings | Application settings |
| Preferences | User preferences |
| ApiConfig | API configuration |
| Appearance | Theme and display settings |

## Keyboard Shortcuts

### View Navigation
| Shortcut | Action |
|----------|--------|
| `1` | Dashboard |
| `2` | Money Flow |
| `3` | Institutional |
| `4` | Dark Pool |
| `5` | Options |
| `6` | Research |
| `7` | Portfolio |
| `8` | Commodities |
| `9` | Macro |

### Symbol Search
| Shortcut | Action |
|----------|--------|
| `Ctrl+K` / `Cmd+K` | Open symbol search |
| `Ctrl+P` | Quick symbol lookup |

### Vim-Style Navigation
| Shortcut | Action |
|----------|--------|
| `j` | Move down |
| `k` | Move up |
| `h` | Move left |
| `l` | Move right |
| `Ctrl+U` | Half page up |
| `Ctrl+D` | Half page down |
| `g g` | Go to top |
| `G` | Go to bottom |

### Table Navigation
| Shortcut | Action |
|----------|--------|
| Arrow keys | Navigate cells |
| `PageUp` / `PageDown` | Page scroll |
| `Home` | Go to first row |
| `End` | Go to last row |
| `Space` | Select/toggle row |
| `Enter` | Open detail view |

### Data Actions
| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Refresh current data |
| `Ctrl+Shift+R` | Refresh all data |
| `Ctrl+E` | Export data |
| `Ctrl+S` | Save changes |

### UI Controls
| Shortcut | Action |
|----------|--------|
| `Ctrl+T` | Toggle theme (dark/light) |
| `Ctrl+B` | Toggle sidebar |
| `F11` | Toggle fullscreen |
| `Escape` | Close modal/cancel |

### Help
| Shortcut | Action |
|----------|--------|
| `Shift+?` | Show keyboard shortcuts |
| `F1` | Open help |

## API Client

The GUI communicates with the Python backend via REST API. All endpoints are prefixed with `/api/`:

```rust
// src/api.rs
pub struct StanleyClient {
    base_url: String,
    client: reqwest::Client,
}

impl StanleyClient {
    // Core endpoints
    pub async fn health_check(&self) -> Result<HealthResponse>;
    pub async fn get_market_data(&self, symbol: &str) -> Result<MarketData>;

    // Flow analytics
    pub async fn get_sector_money_flow(&self, sectors: Vec<String>) -> Result<Vec<MoneyFlowData>>;
    pub async fn get_equity_flow(&self, symbol: &str) -> Result<EquityFlowData>;
    pub async fn get_dark_pool(&self, symbol: &str) -> Result<DarkPoolData>;

    // Institutional
    pub async fn get_institutional(&self, symbol: &str) -> Result<Vec<InstitutionalHolding>>;

    // Research
    pub async fn get_research(&self, symbol: &str) -> Result<ResearchData>;

    // Options
    pub async fn get_options_flow(&self, symbol: &str) -> Result<OptionsFlowData>;

    // Notes
    pub async fn get_theses(&self) -> Result<Vec<Thesis>>;
}
```

## Configuration

The backend URL can be configured via environment variable:

```bash
STANLEY_API_URL=http://localhost:8000 ./stanley-gui
```

Default: `http://localhost:8000`

## Platform Support

| Platform | Status |
|----------|--------|
| Linux (Wayland) | Supported |
| Linux (X11) | Not supported |
| macOS | Untested |
| Windows | Not supported |

## Dependencies

- `gpui`: Zed's GPU-accelerated UI framework
- `serde`: Serialization/deserialization
- `reqwest`: Async HTTP client
- `tokio`: Async runtime

## Development

```bash
# Run in development mode
cargo run

# Run with logging
RUST_LOG=debug cargo run

# Format code
cargo fmt

# Check for issues
cargo clippy

# Run tests
cargo test
```

## Theme System

The GUI supports dark and light themes with comprehensive color tokens:

- Background colors (primary, secondary, elevated)
- Text colors (primary, secondary, muted)
- Accent colors (brand, success, warning, error)
- Border and separator colors

Toggle themes with `Ctrl+T` or through Settings > Appearance.

## License

AGPL-3.0-only - Same as the main Stanley project.
