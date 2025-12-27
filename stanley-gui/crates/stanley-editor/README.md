# Stanley Editor

A standalone text editor component extracted and adapted from [Zed's editor crate](https://github.com/zed-industries/zed).

## License

This crate is dual-licensed under Apache-2.0 OR GPL-3.0-or-later, following Zed's licensing model.

- Original work Copyright 2024 Zed Industries Inc.
- Modifications Copyright 2024 Stanley Contributors

## Overview

Stanley Editor provides a minimal, standalone text editor built on GPUI. It focuses on core text editing functionality without the LSP, Git, or collaboration features found in Zed.

## Features

- Text buffer management with efficient storage
- Display mapping (soft wrapping, tabs)
- Selection and cursor management
- Scrolling and viewport handling
- GPUI-based rendering

## What Was Extracted

The following components were extracted from Zed's editor crate:

| Component | Source | Description |
|-----------|--------|-------------|
| `buffer.rs` | `text/` crate | Text buffer with undo/redo |
| `display_map/` | `editor/display_map/` | Coordinate transformations |
| `selection.rs` | `editor/selections_collection.rs` | Multi-cursor support |
| `scroll/` | `editor/scroll/` | Scroll management |
| `movement.rs` | `editor/movement.rs` | Cursor movement logic |
| `element.rs` | `editor/element.rs` | GPUI rendering |
| `actions.rs` | `editor/actions.rs` | Keybinding actions |

## What Was Removed

The following Zed features were intentionally excluded:

- **LSP Integration** (`lsp.rs`, `lsp_ext.rs`, `lsp_colors.rs`)
- **Git Integration** (`git.rs`, `git/blame.rs`)
- **Collaboration** (all collab-related code)
- **Language Extensions** (`clangd_ext.rs`, `rust_analyzer_ext.rs`)
- **Project/Workspace Integration** (project, workspace dependencies)
- **AI/Edit Predictions** (edit_prediction_*)
- **Tasks/Runnables** (`tasks.rs`)
- **Signature Help** (`signature_help.rs`)
- **Hover Popovers** (`hover_popover.rs`, `hover_links.rs`)
- **Code Actions** (`code_context_menus.rs`)
- **Inlay Hints** (`inlays/inlay_hints.rs`)

## Architecture

```
stanley-editor/
├── Cargo.toml          # Minimal dependencies
├── src/
│   ├── lib.rs          # Crate root with core types
│   ├── actions.rs      # Editor actions/commands
│   ├── buffer.rs       # Text buffer implementation
│   ├── display_map/    # Display coordinate mapping
│   │   └── mod.rs
│   ├── editor.rs       # Main Editor struct
│   ├── element.rs      # GPUI Element for rendering
│   ├── movement.rs     # Cursor movement logic
│   ├── scroll/         # Scroll management
│   │   ├── mod.rs
│   │   └── autoscroll.rs
│   ├── selection.rs    # Selection management
│   ├── settings.rs     # Editor configuration
│   └── shims.rs        # Compatibility shims
└── README.md
```

## Usage

```rust
use stanley_editor::{Editor, EditorMode, Buffer};
use gpui::*;

// Create an editor with some text
let editor = cx.new(|cx| {
    Editor::for_buffer(
        cx.new(|_| Buffer::from_text("Hello, world!")),
        cx
    )
});

// Render using EditorElement
let element = EditorElement::new(editor);
```

## Dependencies

- `gpui` - Zed's GPU-accelerated UI framework
- `serde` - Serialization for settings
- `parking_lot` - Efficient synchronization
- `anyhow` - Error handling

## Future Work

- [ ] Syntax highlighting (without LSP)
- [ ] Basic search/replace
- [ ] Multiple cursors UI
- [ ] Clipboard integration
- [ ] Input method support
