"""
Notes Module

Obsidian-style note-taking system for investment research.
Features markdown files with YAML frontmatter, wiki-style linking,
and full-text search.

Example usage:
    from stanley.notes import NoteManager

    # Initialize with vault path
    manager = NoteManager("~/Documents/StanleyVault")

    # Create an investment thesis
    thesis = manager.create_thesis(
        symbol="AAPL",
        company_name="Apple Inc.",
        sector="Technology"
    )

    # Create a trade journal entry
    trade = manager.create_trade(
        symbol="AAPL",
        direction="long",
        entry_price=175.50,
        shares=100
    )

    # Search notes
    results = manager.search("Apple valuation")

    # Get all active theses
    active = manager.get_theses(status="active")
"""

from .models import (
    ConvictionLevel,
    Note,
    NoteFrontmatter,
    NoteType,
    ThesisFrontmatter,
    ThesisStatus,
    TradeFrontmatter,
    TradeDirection,
    TradeStatus,
)
from .templates import Templates
from .vault import Vault

__all__ = [
    # Main classes
    "NoteManager",
    "Vault",
    "Templates",
    # Models
    "Note",
    "NoteFrontmatter",
    "ThesisFrontmatter",
    "TradeFrontmatter",
    # Enums
    "NoteType",
    "ThesisStatus",
    "TradeStatus",
    "TradeDirection",
    "ConvictionLevel",
]


class NoteManager:
    """
    High-level interface for the Stanley notes system.

    Provides simplified methods for common operations like
    creating theses, logging trades, and searching notes.
    """

    def __init__(self, vault_path: str = None):
        """
        Initialize the note manager.

        Args:
            vault_path: Path to the vault directory.
                       Defaults to ~/.stanley/vault
        """
        import os
        from pathlib import Path

        if vault_path is None:
            vault_path = Path.home() / ".stanley" / "vault"

        self.vault = Vault(vault_path)
        self.templates = Templates()
        self.vault.load()

    def create_thesis(
        self,
        symbol: str,
        company_name: str = "",
        sector: str = "",
        conviction: str = "medium",
        content: str = None,
    ) -> Note:
        """
        Create a new investment thesis.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            company_name: Full company name
            sector: Sector/industry
            conviction: low, medium, high, or very_high
            content: Optional custom content (uses template if None)

        Returns:
            Created Note
        """
        conviction_level = ConvictionLevel(conviction)
        frontmatter, template_content = self.templates.investment_thesis(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            conviction=conviction_level,
        )

        return self.vault.create_note(
            name=f"{symbol.upper()} Investment Thesis",
            content=content or template_content,
            frontmatter=frontmatter,
        )

    def create_trade(
        self,
        symbol: str,
        direction: str = "long",
        entry_price: float = 0.0,
        shares: float = 0.0,
        entry_date: str = None,
        content: str = None,
    ) -> Note:
        """
        Create a new trade journal entry.

        Args:
            symbol: Stock symbol
            direction: "long" or "short"
            entry_price: Entry price
            shares: Number of shares
            entry_date: Entry date (ISO format, defaults to today)
            content: Optional custom content

        Returns:
            Created Note
        """
        from datetime import datetime

        trade_direction = TradeDirection(direction)
        trade_date = datetime.fromisoformat(entry_date) if entry_date else datetime.now()

        frontmatter, template_content = self.templates.trade_journal(
            symbol=symbol,
            direction=trade_direction,
            entry_price=entry_price,
            shares=shares,
            entry_date=trade_date,
        )

        date_str = trade_date.strftime("%Y-%m-%d")
        return self.vault.create_note(
            name=f"{symbol.upper()} {direction.title()} - {date_str}",
            content=content or template_content,
            frontmatter=frontmatter,
        )

    def close_trade(
        self,
        trade_name: str,
        exit_price: float,
        exit_date: str = None,
        exit_reason: str = "",
        lessons: str = "",
        grade: str = "",
    ) -> Note:
        """
        Close an open trade.

        Args:
            trade_name: Name of the trade note
            exit_price: Exit price
            exit_date: Exit date (ISO format, defaults to today)
            exit_reason: Reason for exit
            lessons: Lessons learned
            grade: Self-assessment grade

        Returns:
            Updated Note
        """
        from datetime import datetime

        note = self.vault.get(trade_name)
        if not note:
            raise ValueError(f"Trade not found: {trade_name}")

        if not isinstance(note.frontmatter, TradeFrontmatter):
            raise ValueError(f"Note is not a trade: {trade_name}")

        fm = note.frontmatter
        fm.exit_price = exit_price
        fm.exit_date = datetime.fromisoformat(exit_date) if exit_date else datetime.now()
        fm.status = TradeStatus.CLOSED
        fm.exit_reason = exit_reason
        fm.lessons_learned = lessons
        fm.grade = grade

        return self.vault.update_note(trade_name, note.content, fm)

    def create_company(
        self,
        symbol: str,
        company_name: str = "",
        sector: str = "",
        content: str = None,
    ) -> Note:
        """
        Create a company research note.

        Args:
            symbol: Stock symbol
            company_name: Full company name
            sector: Sector/industry
            content: Optional custom content

        Returns:
            Created Note
        """
        frontmatter, template_content = self.templates.company_research(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
        )

        return self.vault.create_note(
            name=f"{company_name or symbol.upper()}",
            content=content or template_content,
            frontmatter=frontmatter,
            folder="companies",
        )

    def create_daily_note(self, date: str = None, content: str = None) -> Note:
        """
        Create a daily note.

        Args:
            date: Date (ISO format, defaults to today)
            content: Optional custom content

        Returns:
            Created Note
        """
        from datetime import datetime

        note_date = datetime.fromisoformat(date) if date else datetime.now()
        frontmatter, template_content = self.templates.daily_note(date=note_date)

        return self.vault.create_note(
            name=note_date.strftime("%Y-%m-%d"),
            content=content or template_content,
            frontmatter=frontmatter,
        )

    def search(self, query: str, limit: int = 50) -> list:
        """
        Full-text search across notes.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        return self.vault.search(query, limit)

    def get_theses(self, status: str = None, symbol: str = None) -> list:
        """
        Get investment theses.

        Args:
            status: Filter by status (research, watchlist, active, closed, invalidated)
            symbol: Filter by symbol

        Returns:
            List of thesis notes
        """
        thesis_status = ThesisStatus(status) if status else None
        return self.vault.get_theses(status=thesis_status, symbol=symbol)

    def get_trades(self, status: str = None, symbol: str = None) -> list:
        """
        Get trade journal entries.

        Args:
            status: Filter by status (open, closed, partial)
            symbol: Filter by symbol

        Returns:
            List of trade notes
        """
        trade_status = TradeStatus(status) if status else None
        return self.vault.get_trades(status=trade_status, symbol=symbol)

    def get_trade_stats(self) -> dict:
        """
        Get aggregate trade statistics.

        Returns:
            Dict with win rate, P&L, etc.
        """
        return self.vault.get_trade_stats()

    def get_backlinks(self, note_name: str) -> list:
        """
        Get all notes that link to the given note.

        Args:
            note_name: Name of the note

        Returns:
            List of notes with backlinks
        """
        return self.vault.get_backlinks(note_name)

    def get_graph(self) -> dict:
        """
        Get the note graph for visualization.

        Returns:
            Dict with nodes and edges
        """
        return self.vault.get_graph()

    def get_note(self, name: str) -> Note:
        """Get a note by name."""
        return self.vault.get(name)

    def update_note(self, name: str, content: str) -> Note:
        """Update a note's content."""
        return self.vault.update_note(name, content)

    def delete_note(self, name: str) -> bool:
        """Delete a note."""
        return self.vault.delete_note(name)

    def list_notes(self, note_type: str = None, tags: list = None, limit: int = 100) -> list:
        """List notes with optional filters."""
        ntype = NoteType(note_type) if note_type else None
        return self.vault.list_notes(note_type=ntype, tags=tags, limit=limit)

    def health_check(self) -> bool:
        """Check if notes system is operational."""
        return self.vault.health_check()
