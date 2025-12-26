//! Main application state and rendering for Stanley GUI

use crate::api::{NoteResponse, StanleyClient};
use crate::theme::Theme;
use gpui::prelude::FluentBuilder;
use gpui::*;
use std::sync::Arc;

/// Loading state for async data
#[derive(Debug, Clone, Default)]
pub enum LoadingState<T> {
    #[default]
    NotStarted,
    Loading,
    Loaded(T),
    Error(String),
}

impl<T> LoadingState<T> {
    pub fn is_loading(&self) -> bool {
        matches!(self, LoadingState::Loading)
    }

    #[allow(dead_code)]
    pub fn is_loaded(&self) -> bool {
        matches!(self, LoadingState::Loaded(_))
    }

    #[allow(dead_code)]
    pub fn is_error(&self) -> bool {
        matches!(self, LoadingState::Error(_))
    }
}

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
    /// Notes-related state
    #[allow(dead_code)]
    notes_search_query: String,
    notes_active_tab: NotesTab,
    /// Cached notes data with loading states
    theses: Vec<ThesisNote>,
    theses_loading: LoadingState<()>,
    trades: Vec<TradeNote>,
    trades_loading: LoadingState<()>,
    /// API client for backend communication
    api_client: Arc<StanleyClient>,
    /// API connection status
    api_connected: LoadingState<bool>,
}

/// Notes panel tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NotesTab {
    #[default]
    Theses,
    Trades,
    Search,
}

/// Investment thesis note data
#[derive(Debug, Clone)]
pub struct ThesisNote {
    pub name: String,
    pub symbol: String,
    pub status: ThesisStatus,
    pub conviction: String,
    pub entry_price: Option<f64>,
    pub target_price: Option<f64>,
    pub modified: String,
}

impl ThesisNote {
    /// Parse a ThesisNote from API NoteResponse
    pub fn from_note_response(note: &NoteResponse) -> Option<Self> {
        let frontmatter = &note.frontmatter;

        // Extract fields from frontmatter JSON
        let symbol = frontmatter.get("symbol")?.as_str()?.to_string();
        let status_str = frontmatter
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("research");
        let conviction = frontmatter
            .get("conviction")
            .and_then(|v| v.as_str())
            .unwrap_or("Medium")
            .to_string();
        let entry_price = frontmatter
            .get("entry_price")
            .and_then(|v| v.as_f64());
        let target_price = frontmatter
            .get("target_price")
            .and_then(|v| v.as_f64());
        let modified = frontmatter
            .get("modified")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let status = match status_str.to_lowercase().as_str() {
            "active" => ThesisStatus::Active,
            "watchlist" => ThesisStatus::Watchlist,
            "closed" => ThesisStatus::Closed,
            "invalidated" => ThesisStatus::Invalidated,
            _ => ThesisStatus::Research,
        };

        Some(ThesisNote {
            name: note.name.clone(),
            symbol,
            status,
            conviction,
            entry_price,
            target_price,
            modified,
        })
    }
}

/// Thesis status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ThesisStatus {
    Research,
    Watchlist,
    Active,
    Closed,
    Invalidated,
}

impl ThesisStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ThesisStatus::Research => "Research",
            ThesisStatus::Watchlist => "Watchlist",
            ThesisStatus::Active => "Active",
            ThesisStatus::Closed => "Closed",
            ThesisStatus::Invalidated => "Invalidated",
        }
    }
}

/// Trade journal note data
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TradeNote {
    pub name: String,
    pub symbol: String,
    pub direction: TradeDirection,
    pub status: TradeStatus,
    pub entry_price: f64,
    pub exit_price: Option<f64>,
    pub shares: f64,
    pub pnl: Option<f64>,
    pub pnl_percent: Option<f64>,
    pub entry_date: String,
}

impl TradeNote {
    /// Parse a TradeNote from API NoteResponse
    pub fn from_note_response(note: &NoteResponse) -> Option<Self> {
        let frontmatter = &note.frontmatter;

        // Extract fields from frontmatter JSON
        let symbol = frontmatter.get("symbol")?.as_str()?.to_string();
        let direction_str = frontmatter
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("long");
        let status_str = frontmatter
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("open");
        let entry_price = frontmatter
            .get("entry_price")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let exit_price = frontmatter
            .get("exit_price")
            .and_then(|v| v.as_f64());
        let shares = frontmatter
            .get("shares")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let pnl = frontmatter.get("pnl").and_then(|v| v.as_f64());
        let pnl_percent = frontmatter.get("pnl_percent").and_then(|v| v.as_f64());
        let entry_date = frontmatter
            .get("entry_date")
            .or_else(|| frontmatter.get("created"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let direction = match direction_str.to_lowercase().as_str() {
            "short" => TradeDirection::Short,
            _ => TradeDirection::Long,
        };

        let status = match status_str.to_lowercase().as_str() {
            "closed" => TradeStatus::Closed,
            "partial" => TradeStatus::Partial,
            _ => TradeStatus::Open,
        };

        Some(TradeNote {
            name: note.name.clone(),
            symbol,
            direction,
            status,
            entry_price,
            exit_price,
            shares,
            pnl,
            pnl_percent,
            entry_date,
        })
    }
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeDirection {
    Long,
    Short,
}

impl TradeDirection {
    pub fn label(&self) -> &'static str {
        match self {
            TradeDirection::Long => "Long",
            TradeDirection::Short => "Short",
        }
    }
}

/// Trade status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TradeStatus {
    Open,
    Closed,
    Partial,
}

impl TradeStatus {
    pub fn label(&self) -> &'static str {
        match self {
            TradeStatus::Open => "Open",
            TradeStatus::Closed => "Closed",
            TradeStatus::Partial => "Partial",
        }
    }
}

/// Available views in the application
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveView {
    #[default]
    Dashboard,
    MoneyFlow,
    Institutional,
    DarkPool,
    Options,
    Portfolio,
    Research,
    Notes,
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
    pub fn new(cx: &mut Context<Self>) -> Self {
        let api_client = Arc::new(StanleyClient::new());

        let mut app = Self {
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
            notes_search_query: String::new(),
            notes_active_tab: NotesTab::default(),
            theses: Vec::new(),
            theses_loading: LoadingState::NotStarted,
            trades: Vec::new(),
            trades_loading: LoadingState::NotStarted,
            api_client,
            api_connected: LoadingState::NotStarted,
        };

        // Start loading data from API
        app.check_api_health(cx);

        app
    }

    /// Check API health and connection status
    pub fn check_api_health(&mut self, cx: &mut Context<Self>) {
        self.api_connected = LoadingState::Loading;
        let client = self.api_client.clone();

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let result = client.health_check().await;

            let _ = cx.update(|cx| {
                if let Some(entity) = this.upgrade() {
                    entity.update(cx, |app: &mut Self, cx: &mut Context<Self>| {
                        match result {
                            Ok(health) => {
                                app.api_connected = LoadingState::Loaded(health.core);
                                // If connected, load notes data
                                if health.core {
                                    app.load_theses(cx);
                                    app.load_trades(cx);
                                }
                            }
                            Err(e) => {
                                app.api_connected = LoadingState::Error(format!("{:?}", e));
                                // Load fallback demo data
                                app.load_demo_data();
                            }
                        }
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    /// Load theses from API
    pub fn load_theses(&mut self, cx: &mut Context<Self>) {
        self.theses_loading = LoadingState::Loading;
        let client = self.api_client.clone();

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let result = client.get_theses(None, None).await;

            let _ = cx.update(|cx| {
                if let Some(entity) = this.upgrade() {
                    entity.update(cx, |app: &mut Self, cx: &mut Context<Self>| {
                        match result {
                            Ok(notes) => {
                                app.theses = notes
                                    .into_iter()
                                    .filter_map(|n| ThesisNote::from_note_response(&n))
                                    .collect();
                                app.theses_loading = LoadingState::Loaded(());
                            }
                            Err(e) => {
                                app.theses_loading = LoadingState::Error(format!("{:?}", e));
                            }
                        }
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    /// Load trades from API
    pub fn load_trades(&mut self, cx: &mut Context<Self>) {
        self.trades_loading = LoadingState::Loading;
        let client = self.api_client.clone();

        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let result = client.get_trades(None, None).await;

            let _ = cx.update(|cx| {
                if let Some(entity) = this.upgrade() {
                    entity.update(cx, |app: &mut Self, cx: &mut Context<Self>| {
                        match result {
                            Ok(notes) => {
                                app.trades = notes
                                    .into_iter()
                                    .filter_map(|n| TradeNote::from_note_response(&n))
                                    .collect();
                                app.trades_loading = LoadingState::Loaded(());
                            }
                            Err(e) => {
                                app.trades_loading = LoadingState::Error(format!("{:?}", e));
                            }
                        }
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    /// Load demo/fallback data when API is unavailable
    fn load_demo_data(&mut self) {
        self.theses = vec![
            ThesisNote {
                name: "AAPL Investment Thesis".to_string(),
                symbol: "AAPL".to_string(),
                status: ThesisStatus::Active,
                conviction: "High".to_string(),
                entry_price: Some(175.0),
                target_price: Some(220.0),
                modified: "2024-12-20".to_string(),
            },
            ThesisNote {
                name: "NVDA Investment Thesis".to_string(),
                symbol: "NVDA".to_string(),
                status: ThesisStatus::Active,
                conviction: "Very High".to_string(),
                entry_price: Some(450.0),
                target_price: Some(600.0),
                modified: "2024-12-18".to_string(),
            },
            ThesisNote {
                name: "GOOGL Investment Thesis".to_string(),
                symbol: "GOOGL".to_string(),
                status: ThesisStatus::Research,
                conviction: "Medium".to_string(),
                entry_price: None,
                target_price: Some(180.0),
                modified: "2024-12-15".to_string(),
            },
            ThesisNote {
                name: "META Investment Thesis".to_string(),
                symbol: "META".to_string(),
                status: ThesisStatus::Watchlist,
                conviction: "Medium".to_string(),
                entry_price: None,
                target_price: Some(650.0),
                modified: "2024-12-10".to_string(),
            },
        ];
        self.theses_loading = LoadingState::Loaded(());

        self.trades = vec![
            TradeNote {
                name: "AAPL Long - 2024-12-15".to_string(),
                symbol: "AAPL".to_string(),
                direction: TradeDirection::Long,
                status: TradeStatus::Open,
                entry_price: 175.50,
                exit_price: None,
                shares: 100.0,
                pnl: None,
                pnl_percent: None,
                entry_date: "2024-12-15".to_string(),
            },
            TradeNote {
                name: "NVDA Long - 2024-11-20".to_string(),
                symbol: "NVDA".to_string(),
                direction: TradeDirection::Long,
                status: TradeStatus::Closed,
                entry_price: 450.0,
                exit_price: Some(520.0),
                shares: 50.0,
                pnl: Some(3500.0),
                pnl_percent: Some(15.56),
                entry_date: "2024-11-20".to_string(),
            },
            TradeNote {
                name: "TSLA Short - 2024-12-01".to_string(),
                symbol: "TSLA".to_string(),
                direction: TradeDirection::Short,
                status: TradeStatus::Closed,
                entry_price: 380.0,
                exit_price: Some(350.0),
                shares: 25.0,
                pnl: Some(750.0),
                pnl_percent: Some(7.89),
                entry_date: "2024-12-01".to_string(),
            },
        ];
        self.trades_loading = LoadingState::Loaded(());
    }

    /// Refresh all data from API
    #[allow(dead_code)]
    pub fn refresh_data(&mut self, cx: &mut Context<Self>) {
        self.check_api_health(cx);
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

    pub fn set_notes_tab(&mut self, tab: NotesTab, cx: &mut Context<Self>) {
        self.notes_active_tab = tab;
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
            .child(self.nav_item("Notes", ActiveView::Notes, cx))
    }

    fn nav_item(&self, label: &str, view: ActiveView, cx: &mut Context<Self>) -> impl IntoElement {
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
                s.border_l_2().border_color(theme.accent).pl(px(10.0))
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
            .child(self.render_content_area(cx))
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

    fn render_content_area(&self, cx: &mut Context<Self>) -> impl IntoElement {
        match self.active_view {
            ActiveView::Notes => self.render_notes_panel(cx),
            _ => self.render_dashboard_content(),
        }
    }

    fn render_dashboard_content(&self) -> Div {
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
            .hover(|s| {
                s.bg(theme.card_bg_elevated)
                    .border_color(theme.border_strong)
            })
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
            .child(
                div().flex().child(
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
                ),
            )
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

    // Notes Panel Rendering

    fn render_notes_panel(&self, cx: &mut Context<Self>) -> Div {
        let show_trade_stats = self.notes_active_tab == NotesTab::Trades;

        let mut panel = div()
            .flex_grow()
            .p(px(28.0))
            .flex()
            .flex_col()
            .gap(px(24.0))
            .overflow_hidden()
            // Notes header with tabs
            .child(self.render_notes_header(cx))
            // Notes content based on active tab
            .child(match self.notes_active_tab {
                NotesTab::Theses => self.render_theses_list(cx),
                NotesTab::Trades => self.render_trades_list(cx),
                NotesTab::Search => self.render_notes_search(cx),
            });

        // Trade statistics card (shown for trades tab)
        if show_trade_stats {
            panel = panel.child(self.render_trade_stats());
        }

        panel
    }

    fn render_notes_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .flex()
            .justify_between()
            .items_center()
            .pb(px(16.0))
            .border_b_1()
            .border_color(theme.border_subtle)
            // Left: Title and tabs
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(32.0))
                    .child(
                        div()
                            .text_size(px(24.0))
                            .font_weight(FontWeight::BOLD)
                            .text_color(theme.text)
                            .child("Notes Vault"),
                    )
                    .child(self.render_notes_tabs(cx)),
            )
            // Right: Action buttons
            .child(
                div()
                    .flex()
                    .gap(px(12.0))
                    .child(self.render_action_button("New Thesis", theme.accent))
                    .child(self.render_action_button("New Trade", theme.positive)),
            )
    }

    fn render_notes_tabs(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .flex()
            .gap(px(4.0))
            .p(px(4.0))
            .rounded(px(8.0))
            .bg(theme.card_bg)
            .border_1()
            .border_color(theme.border_subtle)
            .child(self.notes_tab_button("Theses", NotesTab::Theses, cx))
            .child(self.notes_tab_button("Trades", NotesTab::Trades, cx))
            .child(self.notes_tab_button("Search", NotesTab::Search, cx))
    }

    fn notes_tab_button(
        &self,
        label: &str,
        tab: NotesTab,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = &self.theme;
        let is_selected = self.notes_active_tab == tab;

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
            .id(SharedString::from(format!("notes-tab-{:?}", tab)))
            .px(px(16.0))
            .py(px(8.0))
            .rounded(px(6.0))
            .bg(bg)
            .text_size(px(13.0))
            .font_weight(if is_selected {
                FontWeight::SEMIBOLD
            } else {
                FontWeight::MEDIUM
            })
            .text_color(text_color)
            .cursor_pointer()
            .hover(|s| s.bg(theme.hover_bg).text_color(theme.text_secondary))
            .on_click(cx.listener(move |this, _event, _window, cx| {
                this.set_notes_tab(tab, cx);
            }))
            .child(label.to_string())
    }

    fn render_action_button(&self, label: &str, color: Hsla) -> impl IntoElement {
        div()
            .px(px(16.0))
            .py(px(10.0))
            .rounded(px(8.0))
            .bg(color.opacity(0.15))
            .border_1()
            .border_color(color.opacity(0.3))
            .text_size(px(13.0))
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(color)
            .cursor_pointer()
            .hover(|s| s.bg(color.opacity(0.25)))
            .child(label.to_string())
    }

    fn render_theses_list(&self, cx: &mut Context<Self>) -> Div {
        let theme = &self.theme;

        let mut container = div().flex().flex_col().gap(px(16.0));

        // Show loading state
        if self.theses_loading.is_loading() {
            return container.child(self.render_loading_state("Loading theses..."));
        }

        // Show error state
        if let LoadingState::Error(ref err) = self.theses_loading {
            return container
                .child(self.render_error_state(err, "theses"))
                .child(self.render_retry_button("Retry", cx));
        }

        // Header with count and API status
        container = container.child(
            div()
                .flex()
                .items_center()
                .gap(px(12.0))
                .child(
                    div()
                        .text_size(px(14.0))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(theme.text_muted)
                        .child(format!("{} Investment Theses", self.theses.len())),
                )
                .child(self.render_api_status_badge()),
        );

        // Theses list
        if self.theses.is_empty() {
            container = container.child(self.render_empty_state("No theses found"));
        } else {
            container = container.child(
                div().flex().flex_col().gap(px(12.0)).children(
                    self.theses
                        .iter()
                        .map(|thesis| self.render_thesis_card(thesis))
                        .collect::<Vec<_>>(),
                ),
            );
        }

        container
    }

    fn render_loading_state(&self, message: &str) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .p(px(40.0))
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap(px(12.0))
            .child(
                div()
                    .text_size(px(14.0))
                    .text_color(theme.text_muted)
                    .child(message.to_string()),
            )
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(theme.text_dimmed)
                    .child("Connecting to API..."),
            )
    }

    fn render_error_state(&self, error: &str, context: &str) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .p(px(20.0))
            .rounded(px(8.0))
            .bg(theme.negative_subtle)
            .border_1()
            .border_color(theme.negative_muted)
            .flex()
            .flex_col()
            .gap(px(8.0))
            .child(
                div()
                    .text_size(px(14.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.negative)
                    .child(format!("Failed to load {}", context)),
            )
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(theme.text_muted)
                    .child(error.to_string()),
            )
    }

    fn render_empty_state(&self, message: &str) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .p(px(40.0))
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap(px(8.0))
            .child(
                div()
                    .text_size(px(14.0))
                    .text_color(theme.text_muted)
                    .child(message.to_string()),
            )
    }

    fn render_retry_button(&self, label: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .id("retry-button")
            .px(px(16.0))
            .py(px(8.0))
            .rounded(px(6.0))
            .bg(theme.accent_subtle)
            .border_1()
            .border_color(theme.accent_muted)
            .text_size(px(13.0))
            .font_weight(FontWeight::MEDIUM)
            .text_color(theme.accent)
            .cursor_pointer()
            .hover(|s| s.bg(theme.accent.opacity(0.2)))
            .on_click(cx.listener(|this, _event, _window, cx| {
                this.check_api_health(cx);
            }))
            .child(label.to_string())
    }

    fn render_api_status_badge(&self) -> impl IntoElement {
        let theme = &self.theme;

        let (status_text, bg, border, text_color): (&str, Hsla, Hsla, Hsla) = match &self.api_connected {
            LoadingState::NotStarted => ("Connecting...", theme.accent_subtle, theme.accent_muted, theme.accent),
            LoadingState::Loading => ("Connecting...", theme.accent_subtle, theme.accent_muted, theme.accent),
            LoadingState::Loaded(true) => ("Live", theme.positive_subtle, theme.positive_muted, theme.positive),
            LoadingState::Loaded(false) => ("Offline", theme.negative_subtle, theme.negative_muted, theme.negative),
            LoadingState::Error(_) => ("Demo Mode", hsla(0.12, 0.85, 0.55, 0.15), hsla(0.12, 0.85, 0.55, 0.3), hsla(0.12, 0.85, 0.55, 1.0)),
        };

        div()
            .px(px(8.0))
            .py(px(3.0))
            .rounded(px(4.0))
            .bg(bg)
            .border_1()
            .border_color(border)
            .text_size(px(10.0))
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(text_color)
            .child(status_text.to_string())
    }

    fn render_thesis_card(&self, thesis: &ThesisNote) -> impl IntoElement {
        let theme = &self.theme;
        let status_color = match thesis.status {
            ThesisStatus::Active => theme.positive,
            ThesisStatus::Research => theme.accent,
            ThesisStatus::Watchlist => hsla(0.12, 0.85, 0.55, 1.0), // Orange
            ThesisStatus::Closed => theme.text_muted,
            ThesisStatus::Invalidated => theme.negative,
        };

        div()
            .p(px(20.0))
            .rounded(px(12.0))
            .bg(theme.card_bg)
            .border_1()
            .border_color(theme.border)
            .cursor_pointer()
            .hover(|s| {
                s.bg(theme.card_bg_elevated)
                    .border_color(theme.border_strong)
            })
            .flex()
            .flex_col()
            .gap(px(12.0))
            // Header row: Symbol, status badge, conviction
            .child(
                div()
                    .flex()
                    .justify_between()
                    .items_center()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(12.0))
                            .child(
                                div()
                                    .text_size(px(18.0))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(theme.text)
                                    .child(thesis.symbol.clone()),
                            )
                            .child(
                                // Status badge
                                div()
                                    .px(px(10.0))
                                    .py(px(4.0))
                                    .rounded(px(6.0))
                                    .bg(status_color.opacity(0.15))
                                    .border_1()
                                    .border_color(status_color.opacity(0.3))
                                    .text_size(px(11.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(status_color)
                                    .child(thesis.status.label().to_string()),
                            ),
                    )
                    .child(
                        // Conviction badge
                        div()
                            .px(px(10.0))
                            .py(px(4.0))
                            .rounded(px(6.0))
                            .bg(theme.accent_subtle)
                            .text_size(px(11.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(theme.accent)
                            .child(format!("{} Conviction", thesis.conviction)),
                    ),
            )
            // Title
            .child(
                div()
                    .text_size(px(14.0))
                    .text_color(theme.text_secondary)
                    .child(thesis.name.clone()),
            )
            // Price info row
            .child(
                div()
                    .flex()
                    .gap(px(24.0))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(2.0))
                            .child(
                                div()
                                    .text_size(px(11.0))
                                    .text_color(theme.text_dimmed)
                                    .child("Entry Price"),
                            )
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(theme.text)
                                    .child(
                                        thesis
                                            .entry_price
                                            .map(|p| format!("${:.2}", p))
                                            .unwrap_or_else(|| "—".to_string()),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(2.0))
                            .child(
                                div()
                                    .text_size(px(11.0))
                                    .text_color(theme.text_dimmed)
                                    .child("Target Price"),
                            )
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(theme.positive)
                                    .child(
                                        thesis
                                            .target_price
                                            .map(|p| format!("${:.2}", p))
                                            .unwrap_or_else(|| "—".to_string()),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(2.0))
                            .child(
                                div()
                                    .text_size(px(11.0))
                                    .text_color(theme.text_dimmed)
                                    .child("Upside"),
                            )
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(theme.positive)
                                    .child(match (thesis.entry_price, thesis.target_price) {
                                        (Some(entry), Some(target)) => {
                                            let upside = ((target / entry) - 1.0) * 100.0;
                                            format!("{:+.1}%", upside)
                                        }
                                        _ => "—".to_string(),
                                    }),
                            ),
                    )
                    .child(
                        div().flex_grow().flex().justify_end().child(
                            div()
                                .text_size(px(11.0))
                                .text_color(theme.text_dimmed)
                                .child(format!("Updated {}", thesis.modified)),
                        ),
                    ),
            )
    }

    fn render_trades_list(&self, cx: &mut Context<Self>) -> Div {
        let theme = &self.theme;

        let mut container = div().flex().flex_col().gap(px(16.0));

        // Show loading state
        if self.trades_loading.is_loading() {
            return container.child(self.render_loading_state("Loading trades..."));
        }

        // Show error state
        if let LoadingState::Error(ref err) = self.trades_loading {
            return container
                .child(self.render_error_state(err, "trades"))
                .child(self.render_retry_button("Retry", cx));
        }

        // Header with count and API status
        container = container.child(
            div()
                .flex()
                .items_center()
                .gap(px(12.0))
                .child(
                    div()
                        .text_size(px(14.0))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(theme.text_muted)
                        .child(format!("{} Trade Journal Entries", self.trades.len())),
                )
                .child(self.render_api_status_badge()),
        );

        // Trades list
        if self.trades.is_empty() {
            container = container.child(self.render_empty_state("No trades found"));
        } else {
            container = container.child(
                div().flex().flex_col().gap(px(12.0)).children(
                    self.trades
                        .iter()
                        .map(|trade| self.render_trade_card(trade))
                        .collect::<Vec<_>>(),
                ),
            );
        }

        container
    }

    fn render_trade_card(&self, trade: &TradeNote) -> impl IntoElement {
        let theme = &self.theme;
        let is_profitable = trade.pnl.map(|p| p > 0.0).unwrap_or(false);
        let pnl_color = if is_profitable {
            theme.positive
        } else {
            theme.negative
        };
        let direction_color = match trade.direction {
            TradeDirection::Long => theme.positive,
            TradeDirection::Short => theme.negative,
        };
        let status_color = match trade.status {
            TradeStatus::Open => theme.accent,
            TradeStatus::Closed => theme.text_muted,
            TradeStatus::Partial => hsla(0.12, 0.85, 0.55, 1.0),
        };

        div()
            .p(px(20.0))
            .rounded(px(12.0))
            .bg(theme.card_bg)
            .border_1()
            .border_color(theme.border)
            .cursor_pointer()
            .hover(|s| {
                s.bg(theme.card_bg_elevated)
                    .border_color(theme.border_strong)
            })
            .flex()
            .justify_between()
            .items_center()
            // Left: Trade info
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(20.0))
                    // Symbol and direction
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(12.0))
                            .child(
                                div()
                                    .text_size(px(18.0))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(theme.text)
                                    .child(trade.symbol.clone()),
                            )
                            .child(
                                // Direction badge
                                div()
                                    .px(px(8.0))
                                    .py(px(4.0))
                                    .rounded(px(4.0))
                                    .bg(direction_color.opacity(0.15))
                                    .text_size(px(10.0))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(direction_color)
                                    .child(trade.direction.label().to_uppercase()),
                            )
                            .child(
                                // Status badge
                                div()
                                    .px(px(8.0))
                                    .py(px(4.0))
                                    .rounded(px(4.0))
                                    .bg(status_color.opacity(0.15))
                                    .text_size(px(10.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(status_color)
                                    .child(trade.status.label()),
                            ),
                    )
                    // Entry/Exit prices
                    .child(
                        div()
                            .flex()
                            .gap(px(16.0))
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .child(
                                        div()
                                            .text_size(px(10.0))
                                            .text_color(theme.text_dimmed)
                                            .child("Entry"),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .font_weight(FontWeight::MEDIUM)
                                            .text_color(theme.text)
                                            .child(format!("${:.2}", trade.entry_price)),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .child(
                                        div()
                                            .text_size(px(10.0))
                                            .text_color(theme.text_dimmed)
                                            .child("Exit"),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .font_weight(FontWeight::MEDIUM)
                                            .text_color(theme.text)
                                            .child(
                                                trade
                                                    .exit_price
                                                    .map(|p| format!("${:.2}", p))
                                                    .unwrap_or_else(|| "—".to_string()),
                                            ),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .child(
                                        div()
                                            .text_size(px(10.0))
                                            .text_color(theme.text_dimmed)
                                            .child("Shares"),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .font_weight(FontWeight::MEDIUM)
                                            .text_color(theme.text)
                                            .child(format!("{:.0}", trade.shares)),
                                    ),
                            ),
                    )
                    // Date
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(theme.text_dimmed)
                            .child(trade.entry_date.clone()),
                    ),
            )
            // Right: P&L
            .child(
                div()
                    .flex()
                    .flex_col()
                    .items_end()
                    .gap(px(2.0))
                    .when(trade.pnl.is_some(), |s| {
                        s.child(
                            div()
                                .text_size(px(16.0))
                                .font_weight(FontWeight::BOLD)
                                .text_color(pnl_color)
                                .child(format!("{:+.2}", trade.pnl.unwrap_or(0.0))),
                        )
                        .child(
                            div()
                                .text_size(px(12.0))
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(pnl_color.opacity(0.8))
                                .child(format!("{:+.2}%", trade.pnl_percent.unwrap_or(0.0))),
                        )
                    })
                    .when(trade.pnl.is_none(), |s| {
                        s.child(
                            div()
                                .text_size(px(14.0))
                                .text_color(theme.text_dimmed)
                                .child("Open"),
                        )
                    }),
            )
    }

    fn render_trade_stats(&self) -> impl IntoElement {
        let theme = &self.theme;

        // Calculate stats from trades
        let closed_trades: Vec<_> = self
            .trades
            .iter()
            .filter(|t| t.status == TradeStatus::Closed)
            .collect();
        let total_pnl: f64 = closed_trades.iter().filter_map(|t| t.pnl).sum();
        let winners = closed_trades
            .iter()
            .filter(|t| t.pnl.map(|p| p > 0.0).unwrap_or(false))
            .count();
        let win_rate = if !closed_trades.is_empty() {
            (winners as f64 / closed_trades.len() as f64) * 100.0
        } else {
            0.0
        };

        div()
            .p(px(24.0))
            .rounded(px(12.0))
            .bg(theme.card_bg)
            .border_1()
            .border_color(theme.border)
            .flex()
            .flex_col()
            .gap(px(20.0))
            .child(
                div()
                    .text_size(px(15.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.text)
                    .child("Trade Statistics"),
            )
            .child(
                div()
                    .flex()
                    .gap(px(32.0))
                    .child(self.stat_item(
                        "Total Trades",
                        &closed_trades.len().to_string(),
                        theme.text,
                    ))
                    .child(self.stat_item("Winners", &winners.to_string(), theme.positive))
                    .child(self.stat_item(
                        "Losers",
                        &(closed_trades.len() - winners).to_string(),
                        theme.negative,
                    ))
                    .child(self.stat_item("Win Rate", &format!("{:.1}%", win_rate), theme.accent))
                    .child(self.stat_item(
                        "Total P&L",
                        &format!("${:+.2}", total_pnl),
                        if total_pnl >= 0.0 {
                            theme.positive
                        } else {
                            theme.negative
                        },
                    )),
            )
    }

    fn stat_item(&self, label: &str, value: &str, color: Hsla) -> impl IntoElement {
        let theme = &self.theme;

        div()
            .flex()
            .flex_col()
            .gap(px(4.0))
            .child(
                div()
                    .text_size(px(11.0))
                    .text_color(theme.text_dimmed)
                    .child(label.to_string()),
            )
            .child(
                div()
                    .text_size(px(20.0))
                    .font_weight(FontWeight::BOLD)
                    .text_color(color)
                    .child(value.to_string()),
            )
    }

    fn render_notes_search(&self, _cx: &mut Context<Self>) -> Div {
        let theme = &self.theme;

        div()
            .flex()
            .flex_col()
            .gap(px(20.0))
            // Search input placeholder
            .child(
                div()
                    .w_full()
                    .px(px(16.0))
                    .py(px(14.0))
                    .rounded(px(10.0))
                    .bg(theme.card_bg)
                    .border_1()
                    .border_color(theme.border)
                    .text_size(px(14.0))
                    .text_color(theme.text_dimmed)
                    .child("Search notes... (FTS5 powered)"),
            )
            // Search results placeholder
            .child(
                div()
                    .p(px(40.0))
                    .flex()
                    .flex_col()
                    .items_center()
                    .justify_center()
                    .gap(px(12.0))
                    .child(
                        div()
                            .text_size(px(48.0))
                            .text_color(theme.text_dimmed)
                            .child("..."),
                    )
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(theme.text_muted)
                            .child("Enter a search term to find notes"),
                    )
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(theme.text_dimmed)
                            .child("Supports FTS5 syntax: AND, OR, \"exact phrase\""),
                    ),
            )
    }
}
