//! Main application state and rendering for Stanley GUI

use crate::theme::Theme;
use gpui::prelude::FluentBuilder;
use gpui::*;

/// Main application state
pub struct StanleyApp {
    /// Current active view/tab
    active_view: ActiveView,
    /// Theme configuration
    theme: Theme,
    /// Selected symbol for analysis
    selected_symbol: Option<String>,
    /// List of watched symbols
    watchlist: Vec<String>,
    /// Selected time period for charts
    selected_period: TimePeriod,
}

/// Available views in the application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveView {
    Dashboard,
    MoneyFlow,
    Institutional,
    DarkPool,
    Options,
    Portfolio,
    Research,
}

impl Default for ActiveView {
    fn default() -> Self {
        Self::Dashboard
    }
}

/// Available time periods for chart display
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimePeriod {
    #[default]
    OneDay,
    OneWeek,
    OneMonth,
    ThreeMonths,
    OneYear,
}

impl TimePeriod {
    pub fn label(&self) -> &'static str {
        match self {
            TimePeriod::OneDay => "1D",
            TimePeriod::OneWeek => "1W",
            TimePeriod::OneMonth => "1M",
            TimePeriod::ThreeMonths => "3M",
            TimePeriod::OneYear => "1Y",
        }
    }

    pub fn all() -> &'static [TimePeriod] {
        &[
            TimePeriod::OneDay,
            TimePeriod::OneWeek,
            TimePeriod::OneMonth,
            TimePeriod::ThreeMonths,
            TimePeriod::OneYear,
        ]
    }
}

impl StanleyApp {
    pub fn new(_cx: &mut Context<Self>) -> Self {
        Self {
            active_view: ActiveView::Dashboard,
            theme: Theme::dark(),
            selected_symbol: Some("AAPL".to_string()),
            watchlist: vec![
                "AAPL".to_string(),
                "MSFT".to_string(),
                "GOOGL".to_string(),
                "AMZN".to_string(),
                "NVDA".to_string(),
            ],
            selected_period: TimePeriod::default(),
        }
    }

    pub fn set_active_view(&mut self, view: ActiveView, cx: &mut Context<Self>) {
        self.active_view = view;
        cx.notify();
    }

    pub fn select_symbol(&mut self, symbol: String, cx: &mut Context<Self>) {
        self.selected_symbol = Some(symbol);
        cx.notify();
    }

    pub fn set_time_period(&mut self, period: TimePeriod, cx: &mut Context<Self>) {
        self.selected_period = period;
        cx.notify();
    }
}

impl Render for StanleyApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .size_full()
            .flex()
            .flex_row()
            .bg(theme.background)
            .text_color(theme.text)
            .font_family("Inter")
            .child(self.render_sidebar(cx))
            .child(self.render_main_content(cx))
    }
}

impl StanleyApp {
    fn render_sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .w(px(260.0))
            .h_full()
            .flex()
            .flex_col()
            .bg(theme.sidebar_bg)
            .border_r_1()
            .border_color(theme.border_subtle)
            .child(self.render_logo())
            .child(self.render_nav_items(cx))
            .child(self.render_watchlist(cx))
    }

    fn render_logo(&self) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .px(px(20.0))
            .py(px(24.0))
            .flex()
            .items_center()
            .gap(px(14.0))
            .border_b_1()
            .border_color(theme.border_subtle)
            .mb(px(8.0))
            .child(
                // Logo icon with refined styling
                div()
                    .size(px(40.0))
                    .bg(theme.accent)
                    .rounded(px(10.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .border_1()
                    .border_color(theme.accent_glow)
                    .child(
                        div()
                            .text_size(px(20.0))
                            .font_weight(FontWeight::BLACK)
                            .text_color(hsla(0.0, 0.0, 1.0, 0.95))
                            .child("S"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(2.0))
                    .child(
                        div()
                            .text_size(px(18.0))
                            .font_weight(FontWeight::BOLD)
                            .text_color(theme.text)
                            .child("Stanley"),
                    )
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(theme.text_dimmed)
                            .child("Investment Analysis"),
                    ),
            )
    }

    fn render_nav_items(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .flex()
            .flex_col()
            .gap(px(2.0))
            .px(px(12.0))
            .py(px(12.0))
            // Section label
            .child(
                div()
                    .text_size(px(10.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.text_dimmed)
                    .px(px(12.0))
                    .mb(px(8.0))
                    .child("NAVIGATION"),
            )
            .child(self.nav_item("Dashboard", ActiveView::Dashboard, cx))
            .child(self.nav_item("Money Flow", ActiveView::MoneyFlow, cx))
            .child(self.nav_item("Institutional", ActiveView::Institutional, cx))
            .child(self.nav_item("Dark Pool", ActiveView::DarkPool, cx))
            .child(self.nav_item("Options Flow", ActiveView::Options, cx))
            .child(self.nav_item("Portfolio", ActiveView::Portfolio, cx))
            .child(self.nav_item("Research", ActiveView::Research, cx))
    }

    fn nav_item(
        &self,
        label: &str,
        view: ActiveView,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let is_active = self.active_view == view;
        let theme = &self.theme;

        let bg = if is_active {
            theme.accent_subtle
        } else {
            transparent_black()
        };
        let text_color = if is_active {
            theme.accent
        } else {
            theme.text_muted
        };
        let hover_text = if is_active {
            theme.accent
        } else {
            theme.text_secondary
        };

        div()
            .id(SharedString::from(format!("nav-{:?}", view)))
            .relative()
            .flex()
            .items_center()
            .gap(px(10.0))
            .px(px(12.0))
            .py(px(10.0))
            .rounded(px(8.0))
            .bg(bg)
            .text_color(text_color)
            .text_size(px(13.0))
            .font_weight(if is_active {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::NORMAL
            })
            .cursor_pointer()
            .hover(|s| s.bg(theme.nav_hover).text_color(hover_text))
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.set_active_view(view, cx);
            }))
            // Active indicator bar on the left
            .when(is_active, |s| {
                s.child(
                    div()
                        .absolute()
                        .left(px(-12.0))
                        .top(px(8.0))
                        .bottom(px(8.0))
                        .w(px(3.0))
                        .rounded(px(2.0))
                        .bg(theme.nav_active_indicator),
                )
            })
            .child(label.to_string())
    }

    fn render_watchlist(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .flex_grow()
            .flex()
            .flex_col()
            .px(px(12.0))
            .py(px(16.0))
            .border_t_1()
            .border_color(theme.border_subtle)
            .mt(px(8.0))
            .child(
                div()
                    .text_size(px(10.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.text_dimmed)
                    .px(px(12.0))
                    .mb(px(12.0))
                    .child("WATCHLIST"),
            )
            .children(
                self.watchlist
                    .iter()
                    .map(|symbol| self.watchlist_item(symbol, cx))
                    .collect::<Vec<_>>(),
            )
    }

    fn watchlist_item(&self, symbol: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;
        let is_selected = self.selected_symbol.as_deref() == Some(symbol);
        let symbol_owned = symbol.to_string();

        let bg = if is_selected {
            theme.accent_subtle
        } else {
            transparent_black()
        };
        let text_color = if is_selected {
            theme.text
        } else {
            theme.text_secondary
        };

        div()
            .id(SharedString::from(format!("watchlist-{}", symbol)))
            .relative()
            .px(px(12.0))
            .py(px(10.0))
            .rounded(px(8.0))
            .bg(bg)
            .cursor_pointer()
            .hover(|s| s.bg(theme.nav_hover))
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.select_symbol(symbol_owned.clone(), cx);
            }))
            .flex()
            .justify_between()
            .items_center()
            // Selected indicator
            .when(is_selected, |s| {
                s.border_l_2()
                    .border_color(theme.accent)
                    .pl(px(10.0))
            })
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(2.0))
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(text_color)
                            .child(symbol.to_string()),
                    )
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(theme.text_dimmed)
                            .child("$192.42"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .items_end()
                    .gap(px(2.0))
                    .child(
                        div()
                            .text_size(px(12.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.positive)
                            .child("+2.4%"),
                    )
                    .child(
                        div()
                            .text_size(px(10.0))
                            .text_color(theme.positive_muted)
                            .child("+$4.52"),
                    ),
            )
    }

    fn render_main_content(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex_grow()
            .h_full()
            .flex()
            .flex_col()
            .child(self.render_header(cx))
            .child(self.render_content_area())
    }

    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;
        let symbol = self.selected_symbol.as_deref().unwrap_or("Select Symbol");

        div()
            .h(px(72.0))
            .px(px(28.0))
            .flex()
            .items_center()
            .justify_between()
            .border_b_1()
            .border_color(theme.border_subtle)
            .bg(theme.background)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(20.0))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(12.0))
                            .child(
                                div()
                                    .text_size(px(26.0))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(theme.text)
                                    .child(symbol.to_string()),
                            )
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .text_color(theme.text_dimmed)
                                    .child("Apple Inc."),
                            ),
                    )
                    .child(
                        // Price change badge with improved styling
                        div()
                            .px(px(12.0))
                            .py(px(6.0))
                            .rounded(px(6.0))
                            .bg(theme.positive_subtle)
                            .border_1()
                            .border_color(theme.positive_muted)
                            .text_color(theme.positive)
                            .text_size(px(13.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("+$4.52 (2.41%)"),
                    ),
            )
            .child(
                // Time period selector with improved styling
                div()
                    .flex()
                    .gap(px(4.0))
                    .p(px(4.0))
                    .rounded(px(8.0))
                    .bg(theme.card_bg)
                    .border_1()
                    .border_color(theme.border_subtle)
                    .children(
                        TimePeriod::all()
                            .iter()
                            .map(|&period| self.time_period_button(period, cx))
                            .collect::<Vec<_>>(),
                    ),
            )
    }

    fn time_period_button(&self, period: TimePeriod, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;
        let is_selected = self.selected_period == period;

        let bg = if is_selected {
            theme.accent_subtle
        } else {
            transparent_black()
        };
        let text_color = if is_selected {
            theme.accent
        } else {
            theme.text_muted
        };

        div()
            .id(SharedString::from(format!("period-{}", period.label())))
            .px(px(14.0))
            .py(px(6.0))
            .rounded(px(6.0))
            .bg(bg)
            .text_size(px(12.0))
            .font_weight(if is_selected {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::MEDIUM
            })
            .text_color(text_color)
            .cursor_pointer()
            .hover(|s| s.bg(theme.hover_bg).text_color(theme.text_secondary))
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.set_time_period(period, cx);
            }))
            .child(period.label())
    }

    fn render_content_area(&self) -> impl IntoElement {
        div()
            .flex_grow()
            .p(px(28.0))
            .flex()
            .flex_col()
            .gap(px(24.0))
            .overflow_hidden()
            .child(self.render_metrics_row())
            .child(self.render_analysis_cards())
    }

    fn render_metrics_row(&self) -> impl IntoElement {
        div()
            .flex()
            .gap(px(20.0))
            .child(self.metric_card("Money Flow Score", "0.72", "Bullish", true))
            .child(self.metric_card("Institutional %", "75.4%", "+2.1% QoQ", true))
            .child(self.metric_card("Dark Pool %", "34.2%", "Above avg", false))
            .child(self.metric_card("Short Interest", "3.8%", "-0.4%", true))
    }

    fn metric_card(
        &self,
        title: &str,
        value: &str,
        subtitle: &str,
        positive: bool,
    ) -> impl IntoElement {
        let theme = &self.theme;
        let accent = if positive {
            theme.positive
        } else {
            theme.negative
        };
        let accent_subtle = if positive {
            theme.positive_subtle
        } else {
            theme.negative_subtle
        };
        let accent_muted = if positive {
            theme.positive_muted
        } else {
            theme.negative_muted
        };

        div()
            .flex_1()
            .p(px(20.0))
            .rounded(px(12.0))
            .bg(theme.card_bg)
            .border_1()
            .border_color(theme.border)
            // Hover effect for cards
            .cursor_pointer()
            .hover(|s| s.bg(theme.card_bg_elevated).border_color(theme.border_strong))
            .flex()
            .flex_col()
            .gap(px(12.0))
            // Top section with title and indicator
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(
                        div()
                            .text_size(px(12.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text_muted)
                            .child(title.to_string()),
                    )
                    // Colored indicator dot
                    .child(div().size(px(8.0)).rounded_full().bg(accent)),
            )
            // Value with improved typography
            .child(
                div()
                    .text_size(px(32.0))
                    .font_weight(FontWeight::BOLD)
                    .text_color(theme.text)
                    .child(value.to_string()),
            )
            // Subtitle badge
            .child(div().flex().child(
                div()
                    .px(px(10.0))
                    .py(px(4.0))
                    .rounded(px(6.0))
                    .bg(accent_subtle)
                    .border_1()
                    .border_color(accent_muted)
                    .text_color(accent)
                    .text_size(px(11.0))
                    .font_weight(FontWeight::MEDIUM)
                    .child(subtitle.to_string()),
            ))
    }

    fn render_analysis_cards(&self) -> impl IntoElement {
        div()
            .flex()
            .gap(px(20.0))
            .child(
                div()
                    .flex_1()
                    .child(self.analysis_card("Sector Money Flow", self.render_sector_flow())),
            )
            .child(
                div()
                    .flex_1()
                    .child(self.analysis_card("Top Institutional Holders", self.render_holders())),
            )
    }

    fn analysis_card(&self, title: &str, content: impl IntoElement) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .p(px(24.0))
            .rounded(px(12.0))
            .bg(theme.card_bg)
            .border_1()
            .border_color(theme.border)
            .flex()
            .flex_col()
            .gap(px(20.0))
            // Card header
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(
                        div()
                            .text_size(px(15.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.text)
                            .child(title.to_string()),
                    )
                    // View all link
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(theme.accent)
                            .cursor_pointer()
                            .hover(|s| s.text_color(theme.accent_hover))
                            .child("View All"),
                    ),
            )
            .child(content)
    }

    fn render_sector_flow(&self) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap(px(14.0))
            .child(self.sector_row("XLK (Technology)", 0.82, true))
            .child(self.sector_row("XLF (Financials)", 0.45, true))
            .child(self.sector_row("XLE (Energy)", -0.23, false))
            .child(self.sector_row("XLV (Healthcare)", 0.31, true))
            .child(self.sector_row("XLI (Industrials)", 0.12, true))
    }

    fn sector_row(&self, name: &str, score: f32, positive: bool) -> impl IntoElement {
        let theme = &self.theme;
        let bar_color = if positive {
            theme.positive
        } else {
            theme.negative
        };
        let bar_bg = if positive {
            theme.positive_subtle
        } else {
            theme.negative_subtle
        };
        // Calculate bar width as percentage (max 100%)
        let bar_width_percent = (score.abs() * 100.0).min(100.0);
        let bar_width_px = bar_width_percent * 1.8; // Approximate conversion for visual width

        div()
            .flex()
            .items_center()
            .gap(px(16.0))
            .py(px(4.0))
            .px(px(8.0))
            .cursor_pointer()
            .rounded(px(6.0))
            .hover(|s| s.bg(theme.hover_bg))
            .child(
                div()
                    .w(px(130.0))
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(theme.text_secondary)
                    .child(name.to_string()),
            )
            .child(
                // Progress bar container with improved styling
                div()
                    .flex_grow()
                    .h(px(10.0))
                    .rounded(px(5.0))
                    .bg(bar_bg)
                    .overflow_hidden()
                    .child(
                        // Progress bar fill
                        div()
                            .h_full()
                            .w(px(bar_width_px))
                            .rounded(px(5.0))
                            .bg(bar_color),
                    ),
            )
            .child(
                // Score display with improved styling
                div()
                    .w(px(60.0))
                    .text_size(px(13.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(bar_color)
                    .child(format!("{:+.2}", score)),
            )
    }

    fn render_holders(&self) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap(px(0.0))
            .child(self.holder_row("Vanguard Group", "8.2%", "+0.3%", true))
            .child(self.holder_row("BlackRock", "6.8%", "+0.1%", true))
            .child(self.holder_row("State Street", "4.1%", "-0.2%", false))
            .child(self.holder_row("Fidelity", "2.9%", "+0.4%", true))
            .child(self.holder_row("T. Rowe Price", "1.8%", "+0.1%", true))
    }

    fn holder_row(
        &self,
        name: &str,
        ownership: &str,
        change: &str,
        is_positive: bool,
    ) -> impl IntoElement {
        let theme = &self.theme;
        let change_color = if is_positive {
            theme.positive
        } else {
            theme.negative
        };

        div()
            .flex()
            .items_center()
            .justify_between()
            .py(px(12.0))
            .px(px(8.0))
            .border_b_1()
            .border_color(theme.border_subtle)
            .cursor_pointer()
            .rounded(px(4.0))
            .hover(|s| s.bg(theme.hover_bg).border_color(transparent_black()))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.0))
                    // Holder avatar/icon
                    .child(
                        div()
                            .size(px(32.0))
                            .rounded(px(6.0))
                            .bg(theme.accent_subtle)
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_size(px(12.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.accent)
                            .child(name.chars().next().unwrap_or('?').to_string()),
                    )
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.text_secondary)
                            .child(name.to_string()),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap(px(20.0))
                    .items_center()
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.text)
                            .child(ownership.to_string()),
                    )
                    .child(
                        // Change badge
                        div()
                            .px(px(8.0))
                            .py(px(4.0))
                            .rounded(px(4.0))
                            .bg(if is_positive {
                                theme.positive_subtle
                            } else {
                                theme.negative_subtle
                            })
                            .text_size(px(11.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(change_color)
                            .child(change.to_string()),
                    ),
            )
    }
}
