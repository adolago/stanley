//! Stanley GUI - GPUI-based interface for institutional investment analysis
//!
//! This application provides a graphical interface for the Stanley investment
//! analysis platform, focusing on money flow analysis, institutional holdings,
//! and market data visualization.

mod app;
mod components;
mod theme;
mod api;

use gpui::*;
use app::StanleyApp;

fn main() {
    Application::new().run(|cx: &mut App| {
        // Set up window options
        let window_options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds {
                origin: Point::default(),
                size: Size {
                    width: px(1400.0),
                    height: px(900.0),
                },
            })),
            titlebar: Some(TitlebarOptions {
                title: Some("Stanley - Institutional Investment Analysis".into()),
                appears_transparent: false,
                ..Default::default()
            }),
            ..Default::default()
        };

        cx.open_window(window_options, |_window, cx| {
            cx.new(|cx| StanleyApp::new(cx))
        })
        .unwrap();
    });
}
