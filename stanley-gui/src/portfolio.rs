//! Portfolio View for Stanley GUI
//!
//! Displays portfolio holdings, risk metrics, and sector allocation.
//! Features:
//! - Sortable holdings table with P&L coloring
//! - Risk metrics panel (VaR, Sharpe, beta)
//! - Sector allocation visualization
//! - Async data loading with loading states

use crate::api::StanleyClient;
use crate::theme::Theme;
use gpui::prelude::*;
use gpui::*;
use serde::Deserialize;
use std::sync::Arc;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Individual holding in the portfolio
#[derive(Clone, Debug)]
pub struct Holding {
    pub symbol: String,
    pub shares: f64,
    pub avg_cost: f64,
    pub current_price: f64,
    pub market_value: f64,
    pub gain_loss: f64,
    pub gain_loss_pct: f64,
    pub weight: f64,
}

impl Holding {
    /// Create a holding from basic data, calculating derived fields
    pub fn new(
        symbol: String,
        shares: f64,
        avg_cost: f64,
        current_price: f64,
        total_portfolio_value: f64,
    ) -> Self {
        let market_value = shares * current_price;
        let cost_basis = shares * avg_cost;
        let gain_loss = market_value - cost_basis;
        let gain_loss_pct = if cost_basis > 0.0 {
            (gain_loss / cost_basis) * 100.0
        } else {
            0.0
        };
        let weight = if total_portfolio_value > 0.0 {
            (market_value / total_portfolio_value) * 100.0
        } else {
            0.0
        };

        Self {
            symbol,
            shares,
            avg_cost,
            current_price,
            market_value,
            gain_loss,
            gain_loss_pct,
            weight,
        }
    }
}

/// Risk metrics for the portfolio
#[derive(Clone, Debug, Default)]
pub struct RiskMetrics {
    /// Value at Risk (95% confidence)
    pub var_95: f64,
    /// Value at Risk (99% confidence)
    pub var_99: f64,
    /// Conditional VaR (Expected Shortfall) at 95%
    pub cvar_95: f64,
    /// Maximum historical drawdown
    pub max_drawdown: f64,
    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside risk-adjusted return)
    pub sortino_ratio: f64,
    /// Portfolio beta relative to benchmark
    pub beta: f64,
}

/// Sector allocation in the portfolio
#[derive(Clone, Debug)]
pub struct SectorAllocation {
    pub sector: String,
    pub weight: f64,
    pub value: f64,
}

/// API response types for portfolio data
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct PortfolioAnalyticsResponse {
    pub holdings: Option<Vec<HoldingData>>,
    pub risk_metrics: Option<RiskMetricsData>,
    pub sector_allocation: Option<Vec<SectorData>>,
    pub total_value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct HoldingData {
    pub symbol: String,
    pub shares: f64,
    #[serde(rename = "avgCost")]
    pub avg_cost: f64,
    #[serde(rename = "currentPrice")]
    pub current_price: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct RiskMetricsData {
    #[serde(rename = "var95")]
    pub var_95: f64,
    #[serde(rename = "var99")]
    pub var_99: f64,
    #[serde(rename = "cvar95")]
    pub cvar_95: f64,
    #[serde(rename = "maxDrawdown")]
    pub max_drawdown: f64,
    #[serde(rename = "sharpeRatio")]
    pub sharpe_ratio: f64,
    #[serde(rename = "sortinoRatio")]
    pub sortino_ratio: f64,
    pub beta: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SectorData {
    pub sector: String,
    pub weight: f64,
    pub value: f64,
}

// ============================================================================
// LOAD STATE
// ============================================================================

/// Loading state for async data
#[derive(Debug, Clone, Default)]
pub enum LoadState<T> {
    #[default]
    NotLoaded,
    Loading,
    Loaded(T),
    Error(String),
}

impl<T> LoadState<T> {
    #[allow(dead_code)]
    pub fn is_loading(&self) -> bool {
        matches!(self, LoadState::Loading)
    }

    #[allow(dead_code)]
    pub fn is_loaded(&self) -> bool {
        matches!(self, LoadState::Loaded(_))
    }
}

// ============================================================================
// SORT CONFIGURATION
// ============================================================================

/// Columns that can be sorted in the holdings table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortColumn {
    Symbol,
    Shares,
    AvgCost,
    CurrentPrice,
    MarketValue,
    GainLoss,
    GainLossPct,
    Weight,
}

impl SortColumn {
    pub fn label(&self) -> &'static str {
        match self {
            SortColumn::Symbol => "Symbol",
            SortColumn::Shares => "Shares",
            SortColumn::AvgCost => "Avg Cost",
            SortColumn::CurrentPrice => "Price",
            SortColumn::MarketValue => "Value",
            SortColumn::GainLoss => "P&L",
            SortColumn::GainLossPct => "P&L %",
            SortColumn::Weight => "Weight",
        }
    }

    pub fn width(&self) -> f32 {
        match self {
            SortColumn::Symbol => 80.0,
            SortColumn::Shares => 80.0,
            SortColumn::AvgCost => 90.0,
            SortColumn::CurrentPrice => 90.0,
            SortColumn::MarketValue => 100.0,
            SortColumn::GainLoss => 100.0,
            SortColumn::GainLossPct => 80.0,
            SortColumn::Weight => 70.0,
        }
    }
}

// ============================================================================
// PORTFOLIO VIEW
// ============================================================================

/// Main portfolio view component
pub struct PortfolioView {
    holdings: LoadState<Vec<Holding>>,
    risk_metrics: LoadState<RiskMetrics>,
    sector_allocation: LoadState<Vec<SectorAllocation>>,
    total_value: f64,
    selected_holding: Option<usize>,
    sort_column: SortColumn,
    sort_ascending: bool,
    #[allow(dead_code)]
    api_client: Arc<StanleyClient>,
    theme: Theme,
}

impl PortfolioView {
    /// Create a new portfolio view
    pub fn new(api_client: Arc<StanleyClient>) -> Self {
        Self {
            holdings: LoadState::NotLoaded,
            risk_metrics: LoadState::NotLoaded,
            sector_allocation: LoadState::NotLoaded,
            total_value: 0.0,
            selected_holding: None,
            sort_column: SortColumn::Weight,
            sort_ascending: false,
            api_client,
            theme: Theme::dark(),
        }
    }

    /// Load portfolio data from API
    pub fn load_data(&mut self, cx: &mut Context<Self>) {
        self.holdings = LoadState::Loading;
        self.risk_metrics = LoadState::Loading;
        self.sector_allocation = LoadState::Loading;

        // For now, use mock data since the API endpoint may not exist
        // In production, this would call the actual portfolio-analytics endpoint
        cx.spawn(async move |this, cx: &mut AsyncApp| {
            // Simulate API delay
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Mock portfolio data
            let mock_holdings = vec![
                ("AAPL", 100.0, 150.0, 178.50),
                ("MSFT", 50.0, 280.0, 378.90),
                ("GOOGL", 25.0, 120.0, 175.30),
                ("NVDA", 30.0, 450.0, 875.20),
                ("AMZN", 40.0, 130.0, 186.40),
                ("META", 35.0, 290.0, 505.75),
                ("TSLA", 20.0, 200.0, 248.60),
            ];

            let total_value: f64 = mock_holdings
                .iter()
                .map(|(_, shares, _, price)| shares * price)
                .sum();

            let holdings: Vec<Holding> = mock_holdings
                .into_iter()
                .map(|(symbol, shares, avg_cost, current_price)| {
                    Holding::new(
                        symbol.to_string(),
                        shares,
                        avg_cost,
                        current_price,
                        total_value,
                    )
                })
                .collect();

            let risk_metrics = RiskMetrics {
                var_95: -0.0234,
                var_99: -0.0412,
                cvar_95: -0.0356,
                max_drawdown: -0.1245,
                sharpe_ratio: 1.85,
                sortino_ratio: 2.42,
                beta: 1.12,
            };

            let sector_allocation = vec![
                SectorAllocation {
                    sector: "Technology".to_string(),
                    weight: 65.4,
                    value: total_value * 0.654,
                },
                SectorAllocation {
                    sector: "Consumer".to_string(),
                    weight: 18.2,
                    value: total_value * 0.182,
                },
                SectorAllocation {
                    sector: "Automotive".to_string(),
                    weight: 8.8,
                    value: total_value * 0.088,
                },
                SectorAllocation {
                    sector: "Other".to_string(),
                    weight: 7.6,
                    value: total_value * 0.076,
                },
            ];

            let _ = cx.update(|cx| {
                if let Some(entity) = this.upgrade() {
                    entity.update(cx, |view, cx| {
                        view.holdings = LoadState::Loaded(holdings);
                        view.risk_metrics = LoadState::Loaded(risk_metrics);
                        view.sector_allocation = LoadState::Loaded(sector_allocation);
                        view.total_value = total_value;
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    /// Sort holdings by column
    #[allow(dead_code)]
    pub fn sort_by(&mut self, column: SortColumn) {
        if self.sort_column == column {
            self.sort_ascending = !self.sort_ascending;
        } else {
            self.sort_column = column;
            self.sort_ascending = true;
        }

        if let LoadState::Loaded(ref mut holdings) = self.holdings {
            let ascending = self.sort_ascending;
            holdings.sort_by(|a, b| {
                let ord = match column {
                    SortColumn::Symbol => a.symbol.cmp(&b.symbol),
                    SortColumn::Shares => a.shares.partial_cmp(&b.shares).unwrap_or(std::cmp::Ordering::Equal),
                    SortColumn::AvgCost => a.avg_cost.partial_cmp(&b.avg_cost).unwrap_or(std::cmp::Ordering::Equal),
                    SortColumn::CurrentPrice => a.current_price.partial_cmp(&b.current_price).unwrap_or(std::cmp::Ordering::Equal),
                    SortColumn::MarketValue => a.market_value.partial_cmp(&b.market_value).unwrap_or(std::cmp::Ordering::Equal),
                    SortColumn::GainLoss => a.gain_loss.partial_cmp(&b.gain_loss).unwrap_or(std::cmp::Ordering::Equal),
                    SortColumn::GainLossPct => a.gain_loss_pct.partial_cmp(&b.gain_loss_pct).unwrap_or(std::cmp::Ordering::Equal),
                    SortColumn::Weight => a.weight.partial_cmp(&b.weight).unwrap_or(std::cmp::Ordering::Equal),
                };
                if ascending { ord } else { ord.reverse() }
            });
        }
    }

    /// Select a holding row
    #[allow(dead_code)]
    pub fn select_holding(&mut self, index: Option<usize>) {
        self.selected_holding = index;
    }

    // ========================================================================
    // RENDERING
    // ========================================================================

    /// Render the holdings table
    fn render_holdings_table(&self) -> Div {
        let theme = &self.theme;

        match &self.holdings {
            LoadState::Loading => self.render_loading_indicator(),
            LoadState::Error(e) => self.render_error_message(e),
            LoadState::Loaded(holdings) => {
                div()
                    .flex()
                    .flex_col()
                    .w_full()
                    .bg(theme.card_bg)
                    .rounded(px(10.0))
                    .border_1()
                    .border_color(theme.border)
                    .overflow_hidden()
                    // Header
                    .child(self.render_table_header())
                    // Body
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .max_h(px(400.0))
                            .overflow_hidden()
                            .children(
                                holdings.iter().enumerate().map(|(idx, holding)| {
                                    self.render_holding_row(holding, idx)
                                }).collect::<Vec<_>>()
                            )
                    )
                    // Footer with totals
                    .child(self.render_table_footer(holdings))
            }
            LoadState::NotLoaded => self.render_loading_indicator(),
        }
    }

    /// Render table header with sortable columns
    fn render_table_header(&self) -> Div {
        let theme = &self.theme;
        let columns = [
            SortColumn::Symbol,
            SortColumn::Shares,
            SortColumn::AvgCost,
            SortColumn::CurrentPrice,
            SortColumn::MarketValue,
            SortColumn::GainLoss,
            SortColumn::GainLossPct,
            SortColumn::Weight,
        ];

        div()
            .h(px(44.0))
            .flex()
            .items_center()
            .bg(theme.card_bg_elevated)
            .border_b_1()
            .border_color(theme.border)
            .children(columns.iter().map(|col| {
                let is_sorted = self.sort_column == *col;

                div()
                    .w(px(col.width()))
                    .h_full()
                    .flex()
                    .items_center()
                    .px(px(12.0))
                    .cursor_pointer()
                    .hover(|s| s.bg(theme.hover_bg))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(4.0))
                            .child(
                                div()
                                    .text_size(px(11.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(if is_sorted { theme.accent } else { theme.text_muted })
                                    .child(col.label())
                            )
                            .when(is_sorted, |el| {
                                el.child(
                                    div()
                                        .text_size(px(10.0))
                                        .text_color(theme.accent)
                                        .child(if self.sort_ascending { "^" } else { "v" })
                                )
                            })
                    )
            }).collect::<Vec<_>>())
    }

    /// Render a single holding row
    fn render_holding_row(&self, holding: &Holding, index: usize) -> Div {
        let theme = &self.theme;
        let is_selected = self.selected_holding == Some(index);
        let is_positive = holding.gain_loss >= 0.0;
        let pnl_color = if is_positive { theme.positive } else { theme.negative };

        div()
            .h(px(44.0))
            .flex()
            .items_center()
            .bg(if is_selected { theme.accent_subtle } else { transparent_black() })
            .border_b_1()
            .border_color(theme.border_subtle)
            .cursor_pointer()
            .hover(|s| s.bg(theme.hover_bg))
            // Symbol
            .child(
                div()
                    .w(px(SortColumn::Symbol.width()))
                    .px(px(12.0))
                    .text_size(px(13.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.text)
                    .child(holding.symbol.clone())
            )
            // Shares
            .child(
                div()
                    .w(px(SortColumn::Shares.width()))
                    .px(px(12.0))
                    .text_size(px(13.0))
                    .text_color(theme.text_secondary)
                    .child(format_number(holding.shares))
            )
            // Avg Cost
            .child(
                div()
                    .w(px(SortColumn::AvgCost.width()))
                    .px(px(12.0))
                    .text_size(px(13.0))
                    .text_color(theme.text_secondary)
                    .child(format!("${:.2}", holding.avg_cost))
            )
            // Current Price
            .child(
                div()
                    .w(px(SortColumn::CurrentPrice.width()))
                    .px(px(12.0))
                    .text_size(px(13.0))
                    .text_color(theme.text)
                    .child(format!("${:.2}", holding.current_price))
            )
            // Market Value
            .child(
                div()
                    .w(px(SortColumn::MarketValue.width()))
                    .px(px(12.0))
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text)
                    .child(format_currency(holding.market_value))
            )
            // Gain/Loss
            .child(
                div()
                    .w(px(SortColumn::GainLoss.width()))
                    .px(px(12.0))
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(pnl_color)
                    .child(format_currency_signed(holding.gain_loss))
            )
            // Gain/Loss %
            .child(
                div()
                    .w(px(SortColumn::GainLossPct.width()))
                    .px(px(12.0))
                    .child(
                        div()
                            .px(px(6.0))
                            .py(px(2.0))
                            .rounded(px(4.0))
                            .bg(pnl_color.opacity(0.15))
                            .text_size(px(11.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(pnl_color)
                            .child(format!("{:+.2}%", holding.gain_loss_pct))
                    )
            )
            // Weight
            .child(
                div()
                    .w(px(SortColumn::Weight.width()))
                    .px(px(12.0))
                    .text_size(px(13.0))
                    .text_color(theme.text_muted)
                    .child(format!("{:.1}%", holding.weight))
            )
    }

    /// Render table footer with totals
    fn render_table_footer(&self, holdings: &[Holding]) -> Div {
        let theme = &self.theme;
        let total_value: f64 = holdings.iter().map(|h| h.market_value).sum();
        let total_cost: f64 = holdings.iter().map(|h| h.shares * h.avg_cost).sum();
        let total_pnl = total_value - total_cost;
        let total_pnl_pct = if total_cost > 0.0 {
            (total_pnl / total_cost) * 100.0
        } else {
            0.0
        };
        let is_positive = total_pnl >= 0.0;
        let pnl_color = if is_positive { theme.positive } else { theme.negative };

        div()
            .h(px(48.0))
            .flex()
            .items_center()
            .justify_between()
            .px(px(12.0))
            .bg(theme.card_bg_elevated)
            .border_t_1()
            .border_color(theme.border)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.0))
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(theme.text_muted)
                            .child("Total:")
                    )
                    .child(
                        div()
                            .text_size(px(16.0))
                            .font_weight(FontWeight::BOLD)
                            .text_color(theme.text)
                            .child(format_currency(total_value))
                    )
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.0))
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(pnl_color)
                            .child(format_currency_signed(total_pnl))
                    )
                    .child(
                        div()
                            .px(px(8.0))
                            .py(px(4.0))
                            .rounded(px(4.0))
                            .bg(pnl_color.opacity(0.15))
                            .text_size(px(12.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(pnl_color)
                            .child(format!("{:+.2}%", total_pnl_pct))
                    )
            )
    }

    /// Render risk metrics panel
    fn render_risk_panel(&self) -> Div {
        let theme = &self.theme;

        match &self.risk_metrics {
            LoadState::Loading => self.render_loading_indicator(),
            LoadState::Error(e) => self.render_error_message(e),
            LoadState::Loaded(metrics) => {
                div()
                    .flex()
                    .flex_col()
                    .gap(px(12.0))
                    .p(px(16.0))
                    .bg(theme.card_bg)
                    .rounded(px(10.0))
                    .border_1()
                    .border_color(theme.border)
                    // Title
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.text)
                            .child("Risk Metrics")
                    )
                    // Metrics grid
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(8.0))
                            .child(self.render_risk_metric("VaR (95%)", metrics.var_95, true))
                            .child(self.render_risk_metric("VaR (99%)", metrics.var_99, true))
                            .child(self.render_risk_metric("CVaR (95%)", metrics.cvar_95, true))
                            .child(self.render_risk_metric("Max Drawdown", metrics.max_drawdown, true))
                    )
                    // Ratios section
                    .child(
                        div()
                            .pt(px(8.0))
                            .border_t_1()
                            .border_color(theme.border_subtle)
                            .flex()
                            .flex_col()
                            .gap(px(8.0))
                            .child(self.render_ratio_metric("Sharpe Ratio", metrics.sharpe_ratio))
                            .child(self.render_ratio_metric("Sortino Ratio", metrics.sortino_ratio))
                            .child(self.render_ratio_metric("Beta", metrics.beta))
                    )
            }
            LoadState::NotLoaded => self.render_loading_indicator(),
        }
    }

    /// Render a single risk metric row
    fn render_risk_metric(&self, label: &str, value: f64, is_percentage: bool) -> Div {
        let theme = &self.theme;
        let is_negative = value < 0.0;
        let color = if is_negative { theme.negative } else { theme.positive };
        let display = if is_percentage {
            format!("{:.2}%", value * 100.0)
        } else {
            format!("{:.4}", value)
        };

        div()
            .flex()
            .items_center()
            .justify_between()
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(theme.text_muted)
                    .child(label.to_string())
            )
            .child(
                div()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(color)
                    .child(display)
            )
    }

    /// Render a ratio metric row
    fn render_ratio_metric(&self, label: &str, value: f64) -> Div {
        let theme = &self.theme;
        let color = if value >= 1.0 { theme.positive } else if value >= 0.0 { theme.warning } else { theme.negative };

        div()
            .flex()
            .items_center()
            .justify_between()
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(theme.text_muted)
                    .child(label.to_string())
            )
            .child(
                div()
                    .text_size(px(14.0))
                    .font_weight(FontWeight::BOLD)
                    .text_color(color)
                    .child(format!("{:.2}", value))
            )
    }

    /// Render sector allocation chart
    fn render_sector_chart(&self) -> Div {
        let theme = &self.theme;

        match &self.sector_allocation {
            LoadState::Loading => self.render_loading_indicator(),
            LoadState::Error(e) => self.render_error_message(e),
            LoadState::Loaded(sectors) => {
                div()
                    .flex()
                    .flex_col()
                    .gap(px(12.0))
                    .p(px(16.0))
                    .bg(theme.card_bg)
                    .rounded(px(10.0))
                    .border_1()
                    .border_color(theme.border)
                    // Title
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.text)
                            .child("Sector Allocation")
                    )
                    // Sector bars
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(10.0))
                            .children(
                                sectors.iter().map(|sector| {
                                    self.render_sector_bar(sector)
                                }).collect::<Vec<_>>()
                            )
                    )
            }
            LoadState::NotLoaded => self.render_loading_indicator(),
        }
    }

    /// Render a single sector allocation bar
    fn render_sector_bar(&self, sector: &SectorAllocation) -> Div {
        let theme = &self.theme;
        // Generate a color based on sector name (simple hash)
        let hue = (sector.sector.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32)) % 360) as f32;
        let bar_color = hsla(hue / 360.0, 0.65, 0.55, 1.0);

        div()
            .flex()
            .flex_col()
            .gap(px(4.0))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(theme.text_secondary)
                            .child(sector.sector.clone())
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.0))
                            .child(
                                div()
                                    .text_size(px(11.0))
                                    .text_color(theme.text_muted)
                                    .child(format_currency(sector.value))
                            )
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(theme.text)
                                    .child(format!("{:.1}%", sector.weight))
                            )
                    )
            )
            .child(
                div()
                    .h(px(8.0))
                    .w_full()
                    .rounded(px(4.0))
                    .bg(theme.border_subtle)
                    .child(
                        div()
                            .h_full()
                            .w(relative((sector.weight / 100.0) as f32))
                            .rounded(px(4.0))
                            .bg(bar_color)
                    )
            )
    }

    /// Render loading indicator
    fn render_loading_indicator(&self) -> Div {
        let theme = &self.theme;

        div()
            .py(px(40.0))
            .flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(theme.text_dimmed)
                    .child("Loading...")
            )
    }

    /// Render error message
    fn render_error_message(&self, msg: &str) -> Div {
        let theme = &self.theme;

        div()
            .py(px(20.0))
            .px(px(16.0))
            .rounded(px(6.0))
            .bg(theme.negative.opacity(0.1))
            .text_size(px(12.0))
            .text_color(theme.negative)
            .child(msg.to_string())
    }
}

// ============================================================================
// RENDER IMPLEMENTATION
// ============================================================================

impl Render for PortfolioView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        // Three-column layout:
        // Left: Holdings table (wider)
        // Right: Risk metrics panel and sector allocation (stacked)
        div()
            .size_full()
            .p(px(24.0))
            .flex()
            .flex_col()
            .gap(px(20.0))
            .bg(theme.background)
            // Title
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_size(px(20.0))
                            .font_weight(FontWeight::BOLD)
                            .text_color(theme.text)
                            .child("Portfolio Overview")
                    )
                    .child(
                        div()
                            .text_size(px(13.0))
                            .text_color(theme.text_muted)
                            .child(format!("Total Value: {}", format_currency(self.total_value)))
                    )
            )
            // Main content
            .child(
                div()
                    .flex()
                    .gap(px(20.0))
                    .flex_grow()
                    // Left column: Holdings table
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .child(self.render_holdings_table())
                    )
                    // Right column: Risk metrics and sector allocation
                    .child(
                        div()
                            .w(px(300.0))
                            .flex()
                            .flex_col()
                            .gap(px(16.0))
                            .child(self.render_risk_panel())
                            .child(self.render_sector_chart())
                    )
            )
    }
}

// ============================================================================
// FORMATTING HELPERS
// ============================================================================

/// Format a number with appropriate precision
fn format_number(n: f64) -> String {
    if n >= 1_000_000.0 {
        format!("{:.2}M", n / 1_000_000.0)
    } else if n >= 1_000.0 {
        format!("{:.2}K", n / 1_000.0)
    } else if n == n.floor() {
        format!("{:.0}", n)
    } else {
        format!("{:.2}", n)
    }
}

/// Format currency value with abbreviations
fn format_currency(n: f64) -> String {
    if n.abs() >= 1_000_000_000.0 {
        format!("${:.2}B", n / 1_000_000_000.0)
    } else if n.abs() >= 1_000_000.0 {
        format!("${:.2}M", n / 1_000_000.0)
    } else if n.abs() >= 1_000.0 {
        format!("${:.2}K", n / 1_000.0)
    } else {
        format!("${:.2}", n)
    }
}

/// Format signed currency value
fn format_currency_signed(n: f64) -> String {
    let sign = if n >= 0.0 { "+" } else { "" };
    if n.abs() >= 1_000_000_000.0 {
        format!("{}${:.2}B", sign, n / 1_000_000_000.0)
    } else if n.abs() >= 1_000_000.0 {
        format!("{}${:.2}M", sign, n / 1_000_000.0)
    } else if n.abs() >= 1_000.0 {
        format!("{}${:.2}K", sign, n / 1_000.0)
    } else {
        format!("{}${:.2}", sign, n)
    }
}

// ============================================================================
// STANDALONE RENDER FUNCTION
// ============================================================================

/// Standalone function to render portfolio view (for use from main app)
/// This can be called from StanleyApp when the Portfolio view is active
pub fn render_portfolio_content(
    holdings: &LoadState<Vec<Holding>>,
    risk_metrics: &LoadState<RiskMetrics>,
    sector_allocation: &LoadState<Vec<SectorAllocation>>,
    total_value: f64,
    theme: &Theme,
) -> Div {
    div()
        .size_full()
        .p(px(24.0))
        .flex()
        .flex_col()
        .gap(px(20.0))
        // Title
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .child(
                    div()
                        .text_size(px(20.0))
                        .font_weight(FontWeight::BOLD)
                        .text_color(theme.text)
                        .child("Portfolio Overview")
                )
                .child(
                    div()
                        .text_size(px(13.0))
                        .text_color(theme.text_muted)
                        .child(format!("Total Value: {}", format_currency(total_value)))
                )
        )
        // Main content - two columns
        .child(
            div()
                .flex()
                .gap(px(20.0))
                .flex_grow()
                // Left: Holdings
                .child(
                    div()
                        .flex_1()
                        .child(render_holdings_standalone(holdings, theme))
                )
                // Right: Risk + Sectors
                .child(
                    div()
                        .w(px(300.0))
                        .flex()
                        .flex_col()
                        .gap(px(16.0))
                        .child(render_risk_standalone(risk_metrics, theme))
                        .child(render_sectors_standalone(sector_allocation, theme))
                )
        )
}

fn render_holdings_standalone(holdings: &LoadState<Vec<Holding>>, theme: &Theme) -> Div {
    match holdings {
        LoadState::Loading | LoadState::NotLoaded => {
            div()
                .py(px(40.0))
                .flex()
                .items_center()
                .justify_center()
                .child(
                    div()
                        .text_size(px(13.0))
                        .text_color(theme.text_dimmed)
                        .child("Loading holdings...")
                )
        }
        LoadState::Error(e) => {
            div()
                .py(px(20.0))
                .px(px(16.0))
                .rounded(px(6.0))
                .bg(theme.negative.opacity(0.1))
                .text_size(px(12.0))
                .text_color(theme.negative)
                .child(e.clone())
        }
        LoadState::Loaded(holdings) => {
            div()
                .flex()
                .flex_col()
                .bg(theme.card_bg)
                .rounded(px(10.0))
                .border_1()
                .border_color(theme.border)
                .overflow_hidden()
                // Simple header
                .child(
                    div()
                        .h(px(40.0))
                        .flex()
                        .items_center()
                        .px(px(16.0))
                        .bg(theme.card_bg_elevated)
                        .border_b_1()
                        .border_color(theme.border)
                        .child(
                            div()
                                .text_size(px(14.0))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(theme.text)
                                .child("Holdings")
                        )
                )
                // Holdings rows
                .children(
                    holdings.iter().map(|h| {
                        let is_positive = h.gain_loss >= 0.0;
                        let pnl_color = if is_positive { theme.positive } else { theme.negative };

                        div()
                            .h(px(44.0))
                            .flex()
                            .items_center()
                            .justify_between()
                            .px(px(16.0))
                            .border_b_1()
                            .border_color(theme.border_subtle)
                            .hover(|s| s.bg(theme.hover_bg))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(16.0))
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(theme.text)
                                            .child(h.symbol.clone())
                                    )
                                    .child(
                                        div()
                                            .text_size(px(12.0))
                                            .text_color(theme.text_muted)
                                            .child(format!("{} shares", format_number(h.shares)))
                                    )
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(12.0))
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .text_color(theme.text)
                                            .child(format_currency(h.market_value))
                                    )
                                    .child(
                                        div()
                                            .px(px(6.0))
                                            .py(px(2.0))
                                            .rounded(px(4.0))
                                            .bg(pnl_color.opacity(0.15))
                                            .text_size(px(11.0))
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(pnl_color)
                                            .child(format!("{:+.2}%", h.gain_loss_pct))
                                    )
                            )
                    }).collect::<Vec<_>>()
                )
        }
    }
}

fn render_risk_standalone(risk_metrics: &LoadState<RiskMetrics>, theme: &Theme) -> Div {
    match risk_metrics {
        LoadState::Loading | LoadState::NotLoaded => {
            div()
                .p(px(16.0))
                .bg(theme.card_bg)
                .rounded(px(10.0))
                .border_1()
                .border_color(theme.border)
                .child(
                    div()
                        .text_size(px(13.0))
                        .text_color(theme.text_dimmed)
                        .child("Loading risk metrics...")
                )
        }
        LoadState::Error(e) => {
            div()
                .p(px(16.0))
                .bg(theme.card_bg)
                .rounded(px(10.0))
                .border_1()
                .border_color(theme.border)
                .child(
                    div()
                        .text_size(px(12.0))
                        .text_color(theme.negative)
                        .child(e.clone())
                )
        }
        LoadState::Loaded(metrics) => {
            let risk_items = [
                ("VaR (95%)", metrics.var_95 * 100.0, true),
                ("VaR (99%)", metrics.var_99 * 100.0, true),
                ("Max Drawdown", metrics.max_drawdown * 100.0, true),
            ];

            let ratio_items = [
                ("Sharpe", metrics.sharpe_ratio),
                ("Sortino", metrics.sortino_ratio),
                ("Beta", metrics.beta),
            ];

            div()
                .p(px(16.0))
                .bg(theme.card_bg)
                .rounded(px(10.0))
                .border_1()
                .border_color(theme.border)
                .flex()
                .flex_col()
                .gap(px(12.0))
                .child(
                    div()
                        .text_size(px(14.0))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(theme.text)
                        .child("Risk Metrics")
                )
                .children(
                    risk_items.into_iter().map(|(label, value, is_pct)| {
                        let color = if value < 0.0 { theme.negative } else { theme.positive };
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(theme.text_muted)
                                    .child(label)
                            )
                            .child(
                                div()
                                    .text_size(px(13.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(color)
                                    .child(if is_pct { format!("{:.2}%", value) } else { format!("{:.2}", value) })
                            )
                    }).collect::<Vec<_>>()
                )
                .child(
                    div()
                        .h(px(1.0))
                        .w_full()
                        .bg(theme.border_subtle)
                )
                .children(
                    ratio_items.into_iter().map(|(label, value)| {
                        let color = if value >= 1.0 { theme.positive } else if value >= 0.0 { theme.warning } else { theme.negative };
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(theme.text_muted)
                                    .child(label)
                            )
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(color)
                                    .child(format!("{:.2}", value))
                            )
                    }).collect::<Vec<_>>()
                )
        }
    }
}

fn render_sectors_standalone(sectors: &LoadState<Vec<SectorAllocation>>, theme: &Theme) -> Div {
    match sectors {
        LoadState::Loading | LoadState::NotLoaded => {
            div()
                .p(px(16.0))
                .bg(theme.card_bg)
                .rounded(px(10.0))
                .border_1()
                .border_color(theme.border)
                .child(
                    div()
                        .text_size(px(13.0))
                        .text_color(theme.text_dimmed)
                        .child("Loading sectors...")
                )
        }
        LoadState::Error(e) => {
            div()
                .p(px(16.0))
                .bg(theme.card_bg)
                .rounded(px(10.0))
                .border_1()
                .border_color(theme.border)
                .child(
                    div()
                        .text_size(px(12.0))
                        .text_color(theme.negative)
                        .child(e.clone())
                )
        }
        LoadState::Loaded(sectors) => {
            div()
                .p(px(16.0))
                .bg(theme.card_bg)
                .rounded(px(10.0))
                .border_1()
                .border_color(theme.border)
                .flex()
                .flex_col()
                .gap(px(12.0))
                .child(
                    div()
                        .text_size(px(14.0))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(theme.text)
                        .child("Sector Allocation")
                )
                .children(
                    sectors.iter().map(|sector| {
                        let hue = (sector.sector.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32)) % 360) as f32;
                        let bar_color = hsla(hue / 360.0, 0.65, 0.55, 1.0);

                        div()
                            .flex()
                            .flex_col()
                            .gap(px(4.0))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .text_size(px(12.0))
                                            .text_color(theme.text_secondary)
                                            .child(sector.sector.clone())
                                    )
                                    .child(
                                        div()
                                            .text_size(px(12.0))
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(theme.text)
                                            .child(format!("{:.1}%", sector.weight))
                                    )
                            )
                            .child(
                                div()
                                    .h(px(6.0))
                                    .w_full()
                                    .rounded(px(3.0))
                                    .bg(theme.border_subtle)
                                    .child(
                                        div()
                                            .h_full()
                                            .w(relative((sector.weight / 100.0) as f32))
                                            .rounded(px(3.0))
                                            .bg(bar_color)
                                    )
                            )
                    }).collect::<Vec<_>>()
                )
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holding_calculation() {
        let holding = Holding::new(
            "AAPL".to_string(),
            100.0,
            150.0,
            180.0,
            18000.0,
        );

        assert_eq!(holding.symbol, "AAPL");
        assert_eq!(holding.shares, 100.0);
        assert_eq!(holding.market_value, 18000.0);
        assert_eq!(holding.gain_loss, 3000.0);
        assert!((holding.gain_loss_pct - 20.0).abs() < 0.01);
        assert!((holding.weight - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_format_currency() {
        assert_eq!(format_currency(1_500_000_000.0), "$1.50B");
        assert_eq!(format_currency(2_500_000.0), "$2.50M");
        assert_eq!(format_currency(5_500.0), "$5.50K");
        assert_eq!(format_currency(99.50), "$99.50");
    }

    #[test]
    fn test_format_currency_signed() {
        assert_eq!(format_currency_signed(1000.0), "+$1.00K");
        assert_eq!(format_currency_signed(-1000.0), "$-1.00K");
    }
}
