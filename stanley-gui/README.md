# Stanley GUI

GPUI-based graphical interface for the Stanley institutional investment analysis platform.

## Overview

Stanley GUI provides a native desktop interface for Stanley, built using [GPUI](https://github.com/zed-industries/zed) - the high-performance UI framework from the Zed editor team.

## Features

- Real-time market data display
- Portfolio analytics visualization
- Money flow analysis dashboards
- Institutional holdings explorer
- Research note management

## Requirements

- Rust 1.70+
- Linux with Wayland (X11 not supported)
- Running Stanley Python backend

## Building

```bash
cd stanley-gui
cargo build --release
```

The binary will be located at `target/release/stanley-gui`.

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
│   ├── main.rs         # Application entry point
│   ├── app.rs          # Main application state and rendering
│   ├── api.rs          # HTTP client for Python backend
│   ├── components/     # UI components
│   │   ├── mod.rs
│   │   ├── header.rs
│   │   ├── sidebar.rs
│   │   └── charts.rs
│   └── theme.rs        # Theme configuration
├── Cargo.toml          # Dependencies
└── README.md
```

## API Client

The GUI communicates with the Python backend via REST API. All endpoints are prefixed with `/api/`:

```rust
// src/api.rs
pub struct StanleyClient {
    base_url: String,
}

impl StanleyClient {
    pub async fn get_market_data(&self, symbol: &str) -> Result<MarketData>;
    pub async fn get_portfolio_analytics(&self, holdings: Vec<Holding>) -> Result<PortfolioAnalytics>;
    pub async fn get_money_flow(&self, sectors: Vec<String>) -> Result<Vec<MoneyFlowData>>;
    pub async fn get_institutional(&self, symbol: &str) -> Result<Vec<InstitutionalHolding>>;
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

- `gpui`: Zed's UI framework
- `serde`: Serialization
- `reqwest`: HTTP client
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
```

## License

AGPL-3.0-only - Same as the main Stanley project.
