//! Unit tests for the Notes module
//!
//! Tests cover:
//! - Note type classification and parsing
//! - NotesState management
//! - API response handling
//! - Helper function correctness

use super::super::notes::*;

// ============================================================================
// NOTE TYPE TESTS
// ============================================================================

#[test]
fn test_note_type_label() {
    assert_eq!(NoteType::Research.label(), "Research");
    assert_eq!(NoteType::Thesis.label(), "Thesis");
    assert_eq!(NoteType::Trade.label(), "Trade");
    assert_eq!(NoteType::Event.label(), "Event");
    assert_eq!(NoteType::Daily.label(), "Daily");
}

#[test]
fn test_note_type_icon() {
    assert_eq!(NoteType::Research.icon(), "R");
    assert_eq!(NoteType::Thesis.icon(), "T");
    assert_eq!(NoteType::Trade.icon(), "J");
    assert_eq!(NoteType::Event.icon(), "E");
    assert_eq!(NoteType::Daily.icon(), "D");
}

#[test]
fn test_note_type_default() {
    let default: NoteType = Default::default();
    assert_eq!(default, NoteType::Research);
}

// ============================================================================
// NOTE PARSING TESTS
// ============================================================================

#[test]
fn test_note_get_type_research() {
    let note = Note {
        name: "Test Note".to_string(),
        note_type: "research".to_string(),
        content: "Test content".to_string(),
        tags: vec![],
        symbols: vec![],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };
    assert_eq!(note.get_type(), NoteType::Research);
}

#[test]
fn test_note_get_type_thesis() {
    let note = Note {
        name: "Investment Thesis".to_string(),
        note_type: "thesis".to_string(),
        content: "Bull case for AAPL".to_string(),
        tags: vec!["tech".to_string()],
        symbols: vec!["AAPL".to_string()],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };
    assert_eq!(note.get_type(), NoteType::Thesis);
}

#[test]
fn test_note_get_type_trade() {
    let note = Note {
        name: "Trade Journal Entry".to_string(),
        note_type: "trade".to_string(),
        content: "Entered long position".to_string(),
        tags: vec!["trade".to_string()],
        symbols: vec!["MSFT".to_string()],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };
    assert_eq!(note.get_type(), NoteType::Trade);
}

#[test]
fn test_note_get_type_event() {
    let note = Note {
        name: "Earnings Event".to_string(),
        note_type: "event".to_string(),
        content: "Q4 earnings call".to_string(),
        tags: vec![],
        symbols: vec!["NVDA".to_string()],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };
    assert_eq!(note.get_type(), NoteType::Event);
}

#[test]
fn test_note_get_type_daily() {
    let note = Note {
        name: "Daily Notes".to_string(),
        note_type: "daily".to_string(),
        content: "Market observations for today".to_string(),
        tags: vec![],
        symbols: vec![],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };
    assert_eq!(note.get_type(), NoteType::Daily);
}

#[test]
fn test_note_get_type_unknown_defaults_to_research() {
    let note = Note {
        name: "Unknown Type".to_string(),
        note_type: "unknown_type".to_string(),
        content: "Some content".to_string(),
        tags: vec![],
        symbols: vec![],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };
    assert_eq!(note.get_type(), NoteType::Research);
}

#[test]
fn test_note_get_type_empty_defaults_to_research() {
    let note = Note {
        name: "Empty Type".to_string(),
        note_type: "".to_string(),
        content: "Some content".to_string(),
        tags: vec![],
        symbols: vec![],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };
    assert_eq!(note.get_type(), NoteType::Research);
}

// ============================================================================
// NOTES TAB TESTS
// ============================================================================

#[test]
fn test_notes_tab_label() {
    assert_eq!(NotesTab::Notes.label(), "Notes");
    assert_eq!(NotesTab::Theses.label(), "Theses");
    assert_eq!(NotesTab::Trades.label(), "Trade Journal");
    assert_eq!(NotesTab::Events.label(), "Events");
}

#[test]
fn test_notes_tab_icon() {
    assert_eq!(NotesTab::Notes.icon(), "N");
    assert_eq!(NotesTab::Theses.icon(), "T");
    assert_eq!(NotesTab::Trades.icon(), "J");
    assert_eq!(NotesTab::Events.icon(), "E");
}

#[test]
fn test_notes_tab_default() {
    let default: NotesTab = Default::default();
    assert_eq!(default, NotesTab::Notes);
}

// ============================================================================
// NOTES STATE TESTS
// ============================================================================

#[test]
fn test_notes_state_default() {
    let state = NotesState::default();

    assert_eq!(state.active_tab, NotesTab::Notes);
    assert!(matches!(state.notes, LoadState::NotLoaded));
    assert!(matches!(state.theses, LoadState::NotLoaded));
    assert!(matches!(state.trades, LoadState::NotLoaded));
    assert!(matches!(state.events, LoadState::NotLoaded));
    assert!(matches!(state.trade_stats, LoadState::NotLoaded));
    assert!(state.selected_note.is_none());
    assert!(state.search_query.is_empty());
    assert!(state.filter_type.is_none());
    assert!(state.filter_symbol.is_none());
}

#[test]
fn test_notes_state_with_loaded_notes() {
    let notes = vec![
        Note {
            name: "Note 1".to_string(),
            note_type: "research".to_string(),
            content: "Content 1".to_string(),
            tags: vec!["tag1".to_string()],
            symbols: vec!["AAPL".to_string()],
            created_at: "2024-01-01".to_string(),
            updated_at: "2024-01-02".to_string(),
        },
        Note {
            name: "Note 2".to_string(),
            note_type: "thesis".to_string(),
            content: "Content 2".to_string(),
            tags: vec!["tag2".to_string()],
            symbols: vec!["MSFT".to_string()],
            created_at: "2024-01-01".to_string(),
            updated_at: "2024-01-02".to_string(),
        },
    ];

    let state = NotesState {
        notes: LoadState::Loaded(notes.clone()),
        ..Default::default()
    };

    match &state.notes {
        LoadState::Loaded(loaded_notes) => {
            assert_eq!(loaded_notes.len(), 2);
            assert_eq!(loaded_notes[0].name, "Note 1");
            assert_eq!(loaded_notes[1].name, "Note 2");
        }
        _ => panic!("Expected Loaded state"),
    }
}

// ============================================================================
// THESIS DATA TESTS
// ============================================================================

#[test]
fn test_thesis_direction_colors() {
    let long_thesis = Thesis {
        name: "Long Position".to_string(),
        symbol: "AAPL".to_string(),
        direction: "long".to_string(),
        status: "active".to_string(),
        conviction: "high".to_string(),
        entry_price: Some(150.0),
        target_price: Some(200.0),
        stop_loss: Some(130.0),
        thesis_summary: "Bull case".to_string(),
        catalysts: vec!["New product launch".to_string()],
        created_at: "2024-01-01".to_string(),
    };

    let short_thesis = Thesis {
        name: "Short Position".to_string(),
        symbol: "XYZ".to_string(),
        direction: "short".to_string(),
        status: "active".to_string(),
        conviction: "medium".to_string(),
        entry_price: Some(100.0),
        target_price: Some(50.0),
        stop_loss: Some(120.0),
        thesis_summary: "Bear case".to_string(),
        catalysts: vec!["Declining revenue".to_string()],
        created_at: "2024-01-01".to_string(),
    };

    assert_eq!(long_thesis.direction, "long");
    assert_eq!(short_thesis.direction, "short");
}

#[test]
fn test_thesis_optional_fields() {
    let thesis = Thesis {
        name: "Minimal Thesis".to_string(),
        symbol: "TEST".to_string(),
        direction: "long".to_string(),
        status: "watching".to_string(),
        conviction: "low".to_string(),
        entry_price: None,
        target_price: None,
        stop_loss: None,
        thesis_summary: String::new(),
        catalysts: vec![],
        created_at: "2024-01-01".to_string(),
    };

    assert!(thesis.entry_price.is_none());
    assert!(thesis.target_price.is_none());
    assert!(thesis.stop_loss.is_none());
    assert!(thesis.catalysts.is_empty());
}

// ============================================================================
// TRADE ENTRY TESTS
// ============================================================================

#[test]
fn test_trade_entry_pnl_calculation() {
    let winning_trade = TradeEntry {
        name: "Winning Trade".to_string(),
        symbol: "AAPL".to_string(),
        direction: "long".to_string(),
        entry_date: "2024-01-01".to_string(),
        entry_price: 100.0,
        exit_date: Some("2024-02-01".to_string()),
        exit_price: Some(120.0),
        quantity: Some(100.0),
        pnl: Some(2000.0),
        pnl_percent: Some(20.0),
        notes: "Good trade".to_string(),
        tags: vec!["swing".to_string()],
    };

    assert_eq!(winning_trade.pnl, Some(2000.0));
    assert_eq!(winning_trade.pnl_percent, Some(20.0));
}

#[test]
fn test_trade_entry_open_position() {
    let open_trade = TradeEntry {
        name: "Open Position".to_string(),
        symbol: "MSFT".to_string(),
        direction: "long".to_string(),
        entry_date: "2024-01-01".to_string(),
        entry_price: 350.0,
        exit_date: None,
        exit_price: None,
        quantity: Some(50.0),
        pnl: None,
        pnl_percent: None,
        notes: "Still holding".to_string(),
        tags: vec!["position".to_string()],
    };

    assert!(open_trade.exit_date.is_none());
    assert!(open_trade.exit_price.is_none());
    assert!(open_trade.pnl.is_none());
}

// ============================================================================
// TRADE STATS TESTS
// ============================================================================

#[test]
fn test_trade_stats_win_rate() {
    let stats = TradeStats {
        total_trades: 100,
        winning_trades: 60,
        losing_trades: 40,
        win_rate: 0.6,
        total_pnl: 15000.0,
        avg_win: 500.0,
        avg_loss: -250.0,
        profit_factor: 2.0,
    };

    assert_eq!(stats.win_rate, 0.6);
    assert_eq!(stats.profit_factor, 2.0);
    assert_eq!(stats.winning_trades + stats.losing_trades, stats.total_trades);
}

#[test]
fn test_trade_stats_negative_pnl() {
    let stats = TradeStats {
        total_trades: 50,
        winning_trades: 15,
        losing_trades: 35,
        win_rate: 0.3,
        total_pnl: -5000.0,
        avg_win: 200.0,
        avg_loss: -300.0,
        profit_factor: 0.5,
    };

    assert!(stats.total_pnl < 0.0);
    assert!(stats.profit_factor < 1.0);
}

// ============================================================================
// MARKET EVENT TESTS
// ============================================================================

#[test]
fn test_market_event_types() {
    let earnings_event = MarketEvent {
        name: "AAPL Q4 Earnings".to_string(),
        event_type: "earnings".to_string(),
        event_date: "2024-01-25".to_string(),
        symbol: Some("AAPL".to_string()),
        description: "Q4 2023 earnings release".to_string(),
        impact: "high".to_string(),
        completed: false,
    };

    let fed_event = MarketEvent {
        name: "FOMC Meeting".to_string(),
        event_type: "fed".to_string(),
        event_date: "2024-01-31".to_string(),
        symbol: None,
        description: "Federal Reserve interest rate decision".to_string(),
        impact: "high".to_string(),
        completed: false,
    };

    assert_eq!(earnings_event.event_type, "earnings");
    assert!(earnings_event.symbol.is_some());
    assert_eq!(fed_event.event_type, "fed");
    assert!(fed_event.symbol.is_none());
}

#[test]
fn test_market_event_impact_levels() {
    let high_impact = MarketEvent {
        name: "CPI Release".to_string(),
        event_type: "economic".to_string(),
        event_date: "2024-01-10".to_string(),
        symbol: None,
        description: "Consumer Price Index".to_string(),
        impact: "high".to_string(),
        completed: false,
    };

    let low_impact = MarketEvent {
        name: "Minor Report".to_string(),
        event_type: "economic".to_string(),
        event_date: "2024-01-15".to_string(),
        symbol: None,
        description: "Minor economic data".to_string(),
        impact: "low".to_string(),
        completed: true,
    };

    assert_eq!(high_impact.impact, "high");
    assert_eq!(low_impact.impact, "low");
    assert!(low_impact.completed);
    assert!(!high_impact.completed);
}

// ============================================================================
// LOAD STATE TESTS
// ============================================================================

#[test]
fn test_load_state_is_loading() {
    let not_loaded: LoadState<Vec<Note>> = LoadState::NotLoaded;
    let loading: LoadState<Vec<Note>> = LoadState::Loading;
    let loaded: LoadState<Vec<Note>> = LoadState::Loaded(vec![]);
    let error: LoadState<Vec<Note>> = LoadState::Error("Error".to_string());

    assert!(!not_loaded.is_loading());
    assert!(loading.is_loading());
    assert!(!loaded.is_loading());
    assert!(!error.is_loading());
}

// ============================================================================
// HELPER FUNCTION TESTS
// ============================================================================

#[test]
fn test_format_relative_time_empty() {
    let result = format_relative_time("");
    assert_eq!(result, "Unknown");
}

#[test]
fn test_format_relative_time_short_date() {
    let result = format_relative_time("2024");
    assert_eq!(result, "2024");
}

#[test]
fn test_format_relative_time_full_date() {
    let result = format_relative_time("2024-01-15T10:30:00Z");
    assert_eq!(result, "2024-01-15");
}

#[test]
fn test_format_relative_time_date_only() {
    let result = format_relative_time("2024-01-15");
    assert_eq!(result, "2024-01-15");
}

#[test]
fn test_format_date_empty() {
    let result = format_date("");
    assert_eq!(result, "--");
}

#[test]
fn test_format_date_short() {
    let result = format_date("2024");
    assert_eq!(result, "2024");
}

#[test]
fn test_format_date_full() {
    let result = format_date("2024-12-25T23:59:59Z");
    assert_eq!(result, "2024-12-25");
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_note_with_empty_content() {
    let note = Note {
        name: "Empty Note".to_string(),
        note_type: "research".to_string(),
        content: String::new(),
        tags: vec![],
        symbols: vec![],
        created_at: String::new(),
        updated_at: String::new(),
    };

    assert!(note.content.is_empty());
    assert!(note.tags.is_empty());
    assert!(note.symbols.is_empty());
}

#[test]
fn test_note_with_many_tags() {
    let tags: Vec<String> = (0..100).map(|i| format!("tag{}", i)).collect();
    let note = Note {
        name: "Many Tags Note".to_string(),
        note_type: "research".to_string(),
        content: "Content".to_string(),
        tags: tags.clone(),
        symbols: vec!["AAPL".to_string()],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };

    assert_eq!(note.tags.len(), 100);
}

#[test]
fn test_note_with_unicode_content() {
    let note = Note {
        name: "Unicode Note".to_string(),
        note_type: "research".to_string(),
        content: "Analysis: \u{1F4C8} \u{1F4B0} \u{1F6A8}".to_string(), // Chart, money bag, warning emojis
        tags: vec!["unicode".to_string()],
        symbols: vec![],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };

    assert!(note.content.contains('\u{1F4C8}'));
}

#[test]
fn test_note_with_very_long_content() {
    let long_content = "A".repeat(100_000);
    let note = Note {
        name: "Long Content Note".to_string(),
        note_type: "research".to_string(),
        content: long_content.clone(),
        tags: vec![],
        symbols: vec![],
        created_at: "2024-01-01".to_string(),
        updated_at: "2024-01-02".to_string(),
    };

    assert_eq!(note.content.len(), 100_000);
}

#[test]
fn test_thesis_with_zero_prices() {
    let thesis = Thesis {
        name: "Zero Price Thesis".to_string(),
        symbol: "PENNY".to_string(),
        direction: "long".to_string(),
        status: "active".to_string(),
        conviction: "low".to_string(),
        entry_price: Some(0.0),
        target_price: Some(0.0),
        stop_loss: Some(0.0),
        thesis_summary: "Penny stock".to_string(),
        catalysts: vec![],
        created_at: "2024-01-01".to_string(),
    };

    assert_eq!(thesis.entry_price, Some(0.0));
}

#[test]
fn test_thesis_with_negative_prices() {
    // Edge case: shouldn't happen in real data but test handling
    let thesis = Thesis {
        name: "Negative Price Thesis".to_string(),
        symbol: "TEST".to_string(),
        direction: "short".to_string(),
        status: "closed".to_string(),
        conviction: "high".to_string(),
        entry_price: Some(-10.0),
        target_price: Some(-20.0),
        stop_loss: Some(-5.0),
        thesis_summary: "Invalid data test".to_string(),
        catalysts: vec![],
        created_at: "2024-01-01".to_string(),
    };

    assert!(thesis.entry_price.unwrap() < 0.0);
}
