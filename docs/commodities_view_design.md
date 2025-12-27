# Commodities View Design - Stanley GUI

## Overview

This document describes the GPUI Rust implementation for the Commodities view in Stanley GUI. The design follows existing patterns from `app.rs` and integrates with the Python backend's commodities endpoints.

## Files

- `/home/artur/Repositories/stanley/stanley-gui/src/commodities.rs` - Main Commodities view implementation

## Architecture

### Data Flow

```
Python Backend                    Rust GUI
--------------                    --------
/api/commodities         ->      CommoditiesOverview
/api/commodities/{sym}   ->      CommoditySummary
/api/commodities/{sym}/macro ->  MacroAnalysis
/api/commodities/correlations -> CorrelationMatrix
```

### State Management

The `CommoditiesState` struct holds all view state:

```rust
pub struct CommoditiesState {
    pub sub_view: CommoditiesSubView,        // Current sub-tab
    pub selected_commodity: Option<String>,  // Selected commodity symbol
    pub overview: LoadState<CommoditiesOverview>,
    pub detail: LoadState<CommoditySummary>,
    pub macro_analysis: LoadState<MacroAnalysis>,
    pub correlations: LoadState<CorrelationMatrix>,
    pub selected_category: Option<String>,
}
```

### Sub-Views

1. **Overview** - Market grid by category (Energy, Precious Metals, etc.)
2. **Detail** - Individual commodity analysis (price, trends, supply/demand)
3. **Correlations** - Correlation matrix heatmap
4. **Macro Linkages** - Macro-commodity relationship analysis

## Integration Steps

### 1. Add to ActiveView enum in app.rs

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveView {
    #[default]
    Dashboard,
    MoneyFlow,
    Institutional,
    DarkPool,
    Options,
    Research,
    Commodities,  // ADD THIS
}
```

### 2. Add commodities module to main.rs

```rust
mod api;
mod app;
mod commodities;  // ADD THIS
mod theme;
```

### 3. Add state to StanleyApp

```rust
use crate::commodities::CommoditiesState;

pub struct StanleyApp {
    // ... existing fields ...
    commodities: CommoditiesState,
}
```

### 4. Add navigation item

In `render_nav()`:

```rust
.child(self.nav_item("Commodities", ActiveView::Commodities, cx))
```

### 5. Add content routing

In `render_content()`:

```rust
fn render_content(&self, cx: &mut Context<Self>) -> impl IntoElement {
    match self.active_view {
        // ... existing cases ...
        ActiveView::Commodities => self.render_commodities(cx),
    }
}
```

### 6. Add render method

```rust
fn render_commodities(&self, cx: &mut Context<Self>) -> Div {
    commodities::render_commodities(&self.theme, &self.commodities, cx)
}
```

### 7. Add data loading

```rust
fn load_commodities_data(&mut self, cx: &mut Context<Self>) {
    self.commodities.overview = LoadState::Loading;
    let client = self.api_client.clone();

    cx.spawn(async move |this, cx: &mut AsyncApp| {
        let result = client.get_commodities_overview().await;
        let _ = cx.update(|cx| {
            if let Some(entity) = this.upgrade() {
                entity.update(cx, |app, cx| {
                    app.commodities.overview = match result {
                        Ok(r) if r.success => r.data.map(LoadState::Loaded)
                            .unwrap_or(LoadState::Error("No data".into())),
                        Ok(r) => LoadState::Error(r.error.unwrap_or("Unknown error".into())),
                        Err(e) => LoadState::Error(e.to_string()),
                    };
                    cx.notify();
                });
            }
        });
    }).detach();
}
```

## UI Components

### Category Grid

Each commodity category displays as a section with:
- Category header with icon and average change
- Grid of commodity cards (4 per row)

### Commodity Card

Individual card shows:
- Symbol and name
- Current price with trend indicator
- Change percentage badge
- Day range bar (low to high)
- Volume and open interest

### Correlation Matrix

Heatmap visualization:
- Color-coded cells (-1 to +1)
- Green for positive correlation
- Red for negative correlation
- Intensity reflects strength

### Macro Linkages

Each linkage displays:
- Macro indicator name
- Correlation value
- Lead/lag relationship
- Strength badge
- Relationship description

## Color Scheme

### Category Colors

| Category | Color |
|----------|-------|
| Energy | Orange (0.08, 0.8, 0.5) |
| Precious Metals | Gold (0.14, 0.9, 0.6) |
| Base Metals | Teal (0.55, 0.6, 0.5) |
| Agriculture | Green (0.33, 0.7, 0.45) |
| Softs | Brown (0.08, 0.6, 0.35) |
| Livestock | Red (0.0, 0.6, 0.5) |

### Trend Colors

- Positive: `theme.positive` (green)
- Negative: `theme.negative` (red)
- Neutral: `theme.text_muted`

### Strength Indicators

- Strong: `theme.positive`
- Moderate: `theme.accent`
- Weak: `theme.text_muted`

## API Client Extensions

The following methods are added to `StanleyClient`:

```rust
impl StanleyClient {
    pub async fn get_commodities_overview(&self) -> Result<ApiResponse<CommoditiesOverview>, ApiError>;
    pub async fn get_commodity_detail(&self, symbol: &str) -> Result<ApiResponse<CommoditySummary>, ApiError>;
    pub async fn get_commodity_macro(&self, symbol: &str) -> Result<ApiResponse<MacroAnalysis>, ApiError>;
    pub async fn get_commodities_correlations(&self) -> Result<ApiResponse<CorrelationMatrix>, ApiError>;
}
```

## Additional Components

### Commodity Ticker

Horizontal scrolling ticker for dashboard:

```rust
render_commodity_ticker(theme: &Theme, prices: &[CommodityPrice])
```

### Commodity Watchlist

Sidebar watchlist component:

```rust
render_commodity_watchlist(theme: &Theme, commodities: &[CommodityPrice], selected: Option<&str>)
```

## Price Formatting

Different commodities use different price formats:

| Symbol | Format |
|--------|--------|
| GC, PL, PA | $1,950.00 |
| CL, BZ | $75.00 |
| NG | $2.500 |
| HG, SI | $24.000 |
| ZC, ZW | 450.0c |

## Testing

1. Start Python backend: `python -m stanley.api.main`
2. Build Rust GUI: `cd stanley-gui && cargo build --release`
3. Run GUI and navigate to Commodities view

## Future Enhancements

1. **Price Charts** - Add sparkline price visualization
2. **Alerts** - Price and correlation alerts
3. **Watchlist Management** - User-defined commodity watchlists
4. **Real-time Updates** - WebSocket price streaming
5. **Seasonal Patterns** - Historical seasonal analysis
