"""
Note Models Module

Data models for the Stanley notes system, following Obsidian-style
markdown notes with YAML frontmatter and wiki-style linking.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class NoteType(Enum):
    """Types of notes in the vault."""

    GENERAL = "general"
    THESIS = "thesis"  # Investment thesis
    TRADE = "trade"  # Trade journal entry
    COMPANY = "company"  # Company research
    SECTOR = "sector"  # Sector analysis
    MACRO = "macro"  # Macro research
    MEETING = "meeting"  # Meeting notes
    DAILY = "daily"  # Daily notes


class ConvictionLevel(Enum):
    """Conviction level for investment theses."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ThesisStatus(Enum):
    """Status of an investment thesis."""

    RESEARCH = "research"  # Still researching
    WATCHLIST = "watchlist"  # On watchlist
    ACTIVE = "active"  # Active position
    CLOSED = "closed"  # Position closed
    INVALIDATED = "invalidated"  # Thesis was wrong


class TradeDirection(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


class TradeStatus(Enum):
    """Trade status."""

    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class NoteFrontmatter:
    """YAML frontmatter for a note."""

    title: str
    note_type: NoteType = NoteType.GENERAL
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        data = {
            "title": self.title,
            "type": self.note_type.value,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "tags": self.tags,
            "aliases": self.aliases,
            **self.extra,
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "NoteFrontmatter":
        """Parse from YAML string."""
        data = yaml.safe_load(yaml_str) or {}

        note_type = NoteType.GENERAL
        if "type" in data:
            try:
                note_type = NoteType(data.pop("type"))
            except ValueError:
                pass

        created = datetime.now()
        if "created" in data:
            try:
                created = datetime.fromisoformat(data.pop("created"))
            except (ValueError, TypeError):
                pass

        modified = datetime.now()
        if "modified" in data:
            try:
                modified = datetime.fromisoformat(data.pop("modified"))
            except (ValueError, TypeError):
                pass

        return cls(
            title=data.pop("title", "Untitled"),
            note_type=note_type,
            created=created,
            modified=modified,
            tags=data.pop("tags", []),
            aliases=data.pop("aliases", []),
            extra=data,
        )


@dataclass
class ThesisFrontmatter(NoteFrontmatter):
    """Frontmatter for investment thesis notes."""

    symbol: str = ""
    company_name: str = ""
    sector: str = ""
    status: ThesisStatus = ThesisStatus.RESEARCH
    conviction: ConvictionLevel = ConvictionLevel.MEDIUM
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size_percent: Optional[float] = None
    time_horizon: str = ""  # e.g., "6-12 months"
    catalyst: str = ""
    key_risks: List[str] = field(default_factory=list)
    bull_case: str = ""
    bear_case: str = ""
    base_case: str = ""

    def __post_init__(self):
        self.note_type = NoteType.THESIS

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        data = {
            "title": self.title,
            "type": self.note_type.value,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "sector": self.sector,
            "status": self.status.value,
            "conviction": self.conviction.value,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "position_size_percent": self.position_size_percent,
            "time_horizon": self.time_horizon,
            "catalyst": self.catalyst,
            "key_risks": self.key_risks,
            "bull_case": self.bull_case,
            "bear_case": self.bear_case,
            "base_case": self.base_case,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "tags": self.tags,
            "aliases": self.aliases,
            **self.extra,
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ThesisFrontmatter":
        """Parse from YAML string."""
        data = yaml.safe_load(yaml_str) or {}

        status = ThesisStatus.RESEARCH
        if "status" in data:
            try:
                status = ThesisStatus(data.pop("status"))
            except ValueError:
                pass

        conviction = ConvictionLevel.MEDIUM
        if "conviction" in data:
            try:
                conviction = ConvictionLevel(data.pop("conviction"))
            except ValueError:
                pass

        created = datetime.now()
        if "created" in data:
            try:
                created = datetime.fromisoformat(data.pop("created"))
            except (ValueError, TypeError):
                pass

        modified = datetime.now()
        if "modified" in data:
            try:
                modified = datetime.fromisoformat(data.pop("modified"))
            except (ValueError, TypeError):
                pass

        return cls(
            title=data.pop("title", "Untitled"),
            symbol=data.pop("symbol", ""),
            company_name=data.pop("company_name", ""),
            sector=data.pop("sector", ""),
            status=status,
            conviction=conviction,
            entry_price=data.pop("entry_price", None),
            target_price=data.pop("target_price", None),
            stop_loss=data.pop("stop_loss", None),
            position_size_percent=data.pop("position_size_percent", None),
            time_horizon=data.pop("time_horizon", ""),
            catalyst=data.pop("catalyst", ""),
            key_risks=data.pop("key_risks", []),
            bull_case=data.pop("bull_case", ""),
            bear_case=data.pop("bear_case", ""),
            base_case=data.pop("base_case", ""),
            created=created,
            modified=modified,
            tags=data.pop("tags", []),
            aliases=data.pop("aliases", []),
            extra=data,
        )


@dataclass
class TradeFrontmatter(NoteFrontmatter):
    """Frontmatter for trade journal entries."""

    symbol: str = ""
    direction: TradeDirection = TradeDirection.LONG
    status: TradeStatus = TradeStatus.OPEN
    entry_date: Optional[datetime] = None
    exit_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    shares: float = 0.0
    commission: float = 0.0
    thesis_link: str = ""  # [[Link to thesis]]
    entry_reason: str = ""
    exit_reason: str = ""
    lessons_learned: str = ""
    emotional_state: str = ""  # e.g., "confident", "fearful", "fomo"
    grade: str = ""  # A-F self-assessment

    def __post_init__(self):
        self.note_type = NoteType.TRADE

    @property
    def pnl(self) -> Optional[float]:
        """Calculate P&L if trade is closed."""
        if self.exit_price is None:
            return None
        if self.direction == TradeDirection.LONG:
            return (self.exit_price - self.entry_price) * self.shares - self.commission
        else:
            return (self.entry_price - self.exit_price) * self.shares - self.commission

    @property
    def pnl_percent(self) -> Optional[float]:
        """Calculate P&L percentage."""
        if self.exit_price is None or self.entry_price == 0:
            return None
        if self.direction == TradeDirection.LONG:
            return ((self.exit_price / self.entry_price) - 1) * 100
        else:
            return ((self.entry_price / self.exit_price) - 1) * 100

    @property
    def holding_period_days(self) -> Optional[int]:
        """Calculate holding period in days."""
        if self.entry_date is None:
            return None
        end = self.exit_date or datetime.now()
        return (end - self.entry_date).days

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        data = {
            "title": self.title,
            "type": self.note_type.value,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "status": self.status.value,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "shares": self.shares,
            "commission": self.commission,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "holding_period_days": self.holding_period_days,
            "thesis_link": self.thesis_link,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
            "lessons_learned": self.lessons_learned,
            "emotional_state": self.emotional_state,
            "grade": self.grade,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "tags": self.tags,
            "aliases": self.aliases,
            **self.extra,
        }
        data = {k: v for k, v in data.items() if v is not None}
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TradeFrontmatter":
        """Parse from YAML string."""
        data = yaml.safe_load(yaml_str) or {}

        direction = TradeDirection.LONG
        if "direction" in data:
            try:
                direction = TradeDirection(data.pop("direction"))
            except ValueError:
                pass

        status = TradeStatus.OPEN
        if "status" in data:
            try:
                status = TradeStatus(data.pop("status"))
            except ValueError:
                pass

        entry_date = None
        if "entry_date" in data and data["entry_date"]:
            try:
                entry_date = datetime.fromisoformat(data.pop("entry_date"))
            except (ValueError, TypeError):
                data.pop("entry_date", None)

        exit_date = None
        if "exit_date" in data and data["exit_date"]:
            try:
                exit_date = datetime.fromisoformat(data.pop("exit_date"))
            except (ValueError, TypeError):
                data.pop("exit_date", None)

        created = datetime.now()
        if "created" in data:
            try:
                created = datetime.fromisoformat(data.pop("created"))
            except (ValueError, TypeError):
                pass

        modified = datetime.now()
        if "modified" in data:
            try:
                modified = datetime.fromisoformat(data.pop("modified"))
            except (ValueError, TypeError):
                pass

        # Remove computed fields
        data.pop("pnl", None)
        data.pop("pnl_percent", None)
        data.pop("holding_period_days", None)

        return cls(
            title=data.pop("title", "Untitled"),
            symbol=data.pop("symbol", ""),
            direction=direction,
            status=status,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=data.pop("entry_price", 0.0),
            exit_price=data.pop("exit_price", None),
            shares=data.pop("shares", 0.0),
            commission=data.pop("commission", 0.0),
            thesis_link=data.pop("thesis_link", ""),
            entry_reason=data.pop("entry_reason", ""),
            exit_reason=data.pop("exit_reason", ""),
            lessons_learned=data.pop("lessons_learned", ""),
            emotional_state=data.pop("emotional_state", ""),
            grade=data.pop("grade", ""),
            created=created,
            modified=modified,
            tags=data.pop("tags", []),
            aliases=data.pop("aliases", []),
            extra=data,
        )


@dataclass
class Note:
    """A note in the vault."""

    path: Path
    frontmatter: NoteFrontmatter
    content: str
    _outgoing_links: Set[str] = field(default_factory=set)
    _incoming_links: Set[str] = field(default_factory=set)

    # Regex for wiki-style links: [[Link]] or [[Link|Display Text]]
    LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

    @property
    def name(self) -> str:
        """Get note name without extension."""
        return self.path.stem

    @property
    def outgoing_links(self) -> Set[str]:
        """Get all outgoing wiki links from this note."""
        if not self._outgoing_links:
            self._outgoing_links = set(self.LINK_PATTERN.findall(self.content))
        return self._outgoing_links

    @property
    def incoming_links(self) -> Set[str]:
        """Get all incoming links (backlinks) to this note."""
        return self._incoming_links

    def add_backlink(self, source_note: str) -> None:
        """Add a backlink from another note."""
        self._incoming_links.add(source_note)

    def to_markdown(self) -> str:
        """Convert note to markdown with YAML frontmatter."""
        yaml_content = self.frontmatter.to_yaml()
        return f"---\n{yaml_content}---\n\n{self.content}"

    @classmethod
    def from_markdown(cls, path: Path, markdown: str) -> "Note":
        """Parse a note from markdown with YAML frontmatter."""
        frontmatter_match = re.match(
            r"^---\n(.*?)\n---\n\n?(.*)", markdown, re.DOTALL
        )

        if frontmatter_match:
            yaml_str = frontmatter_match.group(1)
            content = frontmatter_match.group(2)

            # Determine frontmatter type from YAML
            data = yaml.safe_load(yaml_str) or {}
            note_type = data.get("type", "general")

            if note_type == "thesis":
                frontmatter = ThesisFrontmatter.from_yaml(yaml_str)
            elif note_type == "trade":
                frontmatter = TradeFrontmatter.from_yaml(yaml_str)
            else:
                frontmatter = NoteFrontmatter.from_yaml(yaml_str)
        else:
            frontmatter = NoteFrontmatter(title=path.stem)
            content = markdown

        return cls(path=path, frontmatter=frontmatter, content=content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "path": str(self.path),
            "name": self.name,
            "title": self.frontmatter.title,
            "type": self.frontmatter.note_type.value,
            "created": self.frontmatter.created.isoformat(),
            "modified": self.frontmatter.modified.isoformat(),
            "tags": self.frontmatter.tags,
            "outgoing_links": list(self.outgoing_links),
            "incoming_links": list(self.incoming_links),
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
        }
