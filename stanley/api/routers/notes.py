"""
Stanley Notes Router

FastAPI router for research vault operations including notes, theses, and trade journal.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..dependencies import get_note_manager
from stanley.notes import NoteManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Notes"])


# =============================================================================
# Pydantic Models - Request Types
# =============================================================================


class CreateThesisRequest(BaseModel):
    """Request to create an investment thesis."""

    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(default="", description="Company name")
    sector: str = Field(default="", description="Sector/industry")
    conviction: str = Field(default="medium", description="Conviction level")
    content: Optional[str] = Field(default=None, description="Custom content")


class CreateTradeRequest(BaseModel):
    """Request to create a trade journal entry."""

    symbol: str = Field(..., description="Stock symbol")
    direction: str = Field(default="long", description="Trade direction")
    entry_price: float = Field(default=0.0, ge=0, description="Entry price")
    shares: float = Field(default=0.0, ge=0, description="Number of shares")
    entry_date: Optional[str] = Field(
        default=None, description="Entry date (ISO format)"
    )
    content: Optional[str] = Field(default=None, description="Custom content")


class CloseTradeRequest(BaseModel):
    """Request to close a trade."""

    exit_price: float = Field(..., ge=0, description="Exit price")
    exit_date: Optional[str] = Field(default=None, description="Exit date (ISO format)")
    exit_reason: str = Field(default="", description="Reason for exit")
    lessons: str = Field(default="", description="Lessons learned")
    grade: str = Field(default="", description="Self-assessment grade")


class CreateEventRequest(BaseModel):
    """Request to create an event note."""

    symbol: str = Field(..., description="Stock symbol")
    company_name: str = Field(default="", description="Company name")
    event_type: str = Field(
        default="conference",
        description="Event type (earnings_call, investor_day, conference, etc.)",
    )
    event_date: Optional[str] = Field(
        default=None, description="Event date (ISO format)"
    )
    host: str = Field(default="", description="Bank/broker hosting the event")
    participants: List[str] = Field(default=[], description="List of participant names")
    content: Optional[str] = Field(default=None, description="Custom content")


class CreatePersonRequest(BaseModel):
    """Request to create a person/executive profile."""

    full_name: str = Field(..., description="Person's full name")
    current_role: str = Field(default="", description="Current role (CEO, CFO, etc.)")
    current_company: str = Field(default="", description="Current company name")
    linkedin_url: str = Field(default="", description="LinkedIn profile URL")
    content: Optional[str] = Field(default=None, description="Custom content")


class CreateSectorRequest(BaseModel):
    """Request to create a sector overview."""

    sector_name: str = Field(..., description="Sector name")
    sub_sectors: List[str] = Field(default=[], description="List of sub-sectors")
    companies: List[str] = Field(default=[], description="List of companies covered")
    content: Optional[str] = Field(default=None, description="Custom content")


class UpdateNoteRequest(BaseModel):
    """Request to update a note."""

    content: str = Field(..., description="New content")


class CreateNoteRequest(BaseModel):
    """Request to create a generic note."""

    name: str = Field(..., description="Note name")
    content: str = Field(default="", description="Note content")
    note_type: str = Field(default="note", description="Note type")
    tags: List[str] = Field(default=[], description="Tags for the note")


# =============================================================================
# Response Models
# =============================================================================


class NoteResponse(BaseModel):
    """Note response model."""

    name: str
    note_type: str
    created: str
    modified: str
    tags: List[str] = []
    links: List[str] = []


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool
    data: Optional[dict | list] = None
    error: Optional[str] = None
    timestamp: str


# =============================================================================
# Helper Functions
# =============================================================================


def get_timestamp() -> str:
    """Get current ISO timestamp."""
    from datetime import datetime

    return datetime.utcnow().isoformat() + "Z"


def create_response(
    data=None, error: Optional[str] = None, success: bool = True
) -> dict:
    """Create a standardized API response."""
    import numpy as np

    def _convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    converted_data = _convert_numpy_types(data) if data is not None else None

    return {
        "success": success and error is None,
        "data": converted_data,
        "error": error,
        "timestamp": get_timestamp(),
    }


# =============================================================================
# Notes Endpoints
# =============================================================================


@router.get("/notes")
async def list_notes(
    note_type: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 100,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    List notes with optional filters.

    Args:
        note_type: Filter by type (thesis, trade, company, etc.)
        tags: Comma-separated list of tags
        limit: Maximum results
    """
    try:
        tag_list = tags.split(",") if tags else None
        notes = note_manager.list_notes(note_type=note_type, tags=tag_list, limit=limit)

        return create_response(data=[n.to_dict() for n in notes])

    except Exception as e:
        logger.error(f"Error listing notes: {e}")
        return create_response(error=str(e), success=False)


@router.get("/notes/search")
async def search_notes(
    query: str,
    limit: int = 50,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    Full-text search across notes.

    Args:
        query: Search query
        limit: Maximum results
    """
    try:
        results = note_manager.search(query, limit)
        return create_response(data=results)

    except Exception as e:
        logger.error(f"Error searching notes: {e}")
        return create_response(error=str(e), success=False)


@router.get("/notes/graph")
async def get_notes_graph(
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Get the note graph for visualization."""
    try:
        graph = note_manager.get_graph()
        return create_response(data=graph)

    except Exception as e:
        logger.error(f"Error getting notes graph: {e}")
        return create_response(error=str(e), success=False)


@router.get("/notes/{name}")
async def get_note(
    name: str,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Get a specific note by name."""
    try:
        note = note_manager.get_note(name)
        if not note:
            raise HTTPException(status_code=404, detail=f"Note not found: {name}")

        return create_response(
            data={
                **note.to_dict(),
                "content": note.content,
                "frontmatter": note.frontmatter.to_yaml(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting note {name}: {e}")
        return create_response(error=str(e), success=False)


@router.get("/notes/{name}/backlinks")
async def get_note_backlinks(
    name: str,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Get all notes that link to the given note."""
    try:
        backlinks = note_manager.get_backlinks(name)
        return create_response(data=[n.to_dict() for n in backlinks])

    except Exception as e:
        logger.error(f"Error getting backlinks for {name}: {e}")
        return create_response(error=str(e), success=False)


@router.put("/notes/{name}")
async def update_note(
    name: str,
    request: UpdateNoteRequest,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Update a note's content."""
    try:
        note = note_manager.update_note(name, request.content)
        return create_response(data=note.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating note {name}: {e}")
        return create_response(error=str(e), success=False)


@router.delete("/notes/{name}")
async def delete_note(
    name: str,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Delete a note."""
    try:
        deleted = note_manager.delete_note(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Note not found: {name}")

        return create_response(data={"deleted": name})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note {name}: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Thesis Endpoints
# =============================================================================


@router.get("/theses")
async def list_theses(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    List investment theses.

    Args:
        status: Filter by status (research, watchlist, active, closed, invalidated)
        symbol: Filter by symbol
    """
    try:
        theses = note_manager.get_theses(status=status, symbol=symbol)
        return create_response(data=[t.to_dict() for t in theses])

    except Exception as e:
        logger.error(f"Error listing theses: {e}")
        return create_response(error=str(e), success=False)


@router.post("/theses")
async def create_thesis(
    request: CreateThesisRequest,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Create a new investment thesis."""
    try:
        thesis = note_manager.create_thesis(
            symbol=request.symbol,
            company_name=request.company_name,
            sector=request.sector,
            conviction=request.conviction,
            content=request.content,
        )

        return create_response(data=thesis.to_dict())

    except Exception as e:
        logger.error(f"Error creating thesis: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Trade Journal Endpoints
# =============================================================================


@router.get("/trades")
async def list_trades(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    List trade journal entries.

    Args:
        status: Filter by status (open, closed, partial)
        symbol: Filter by symbol
    """
    try:
        trades = note_manager.get_trades(status=status, symbol=symbol)
        return create_response(data=[t.to_dict() for t in trades])

    except Exception as e:
        logger.error(f"Error listing trades: {e}")
        return create_response(error=str(e), success=False)


@router.post("/trades")
async def create_trade(
    request: CreateTradeRequest,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Create a new trade journal entry."""
    try:
        trade = note_manager.create_trade(
            symbol=request.symbol,
            direction=request.direction,
            entry_price=request.entry_price,
            shares=request.shares,
            entry_date=request.entry_date,
            content=request.content,
        )

        return create_response(data=trade.to_dict())

    except Exception as e:
        logger.error(f"Error creating trade: {e}")
        return create_response(error=str(e), success=False)


@router.post("/trades/{name}/close")
async def close_trade(
    name: str,
    request: CloseTradeRequest,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Close an open trade."""
    try:
        trade = note_manager.close_trade(
            trade_name=name,
            exit_price=request.exit_price,
            exit_date=request.exit_date,
            exit_reason=request.exit_reason,
            lessons=request.lessons,
            grade=request.grade,
        )

        return create_response(data=trade.to_dict())

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error closing trade {name}: {e}")
        return create_response(error=str(e), success=False)


@router.get("/trades/stats")
async def get_trade_stats(
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Get aggregate trade statistics."""
    try:
        stats = note_manager.get_trade_stats()
        return create_response(data=stats)

    except Exception as e:
        logger.error(f"Error getting trade stats: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Event Endpoints
# =============================================================================


@router.get("/events")
async def list_events(
    event_type: Optional[str] = None,
    symbol: Optional[str] = None,
    company: Optional[str] = None,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    List event notes (conference calls, investor days, etc.).

    Args:
        event_type: Filter by type (earnings_call, conference, investor_day, etc.)
        symbol: Filter by stock symbol
        company: Filter by company name
    """
    try:
        events = note_manager.get_events(
            event_type=event_type, symbol=symbol, company=company
        )
        return create_response(data=[e.to_dict() for e in events])

    except Exception as e:
        logger.error(f"Error listing events: {e}")
        return create_response(error=str(e), success=False)


@router.post("/events")
async def create_event(
    request: CreateEventRequest,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Create a new event note."""
    try:
        event = note_manager.create_event(
            symbol=request.symbol,
            company_name=request.company_name,
            event_type=request.event_type,
            event_date=request.event_date,
            host=request.host,
            participants=request.participants,
            content=request.content,
        )

        return create_response(data=event.to_dict())

    except Exception as e:
        logger.error(f"Error creating event: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# People Endpoints
# =============================================================================


@router.get("/people")
async def list_people(
    company: Optional[str] = None,
    role: Optional[str] = None,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    List person/executive profile notes.

    Args:
        company: Filter by company name
        role: Filter by role (CEO, CFO, etc.)
    """
    try:
        people = note_manager.get_people(company=company, role=role)
        return create_response(data=[p.to_dict() for p in people])

    except Exception as e:
        logger.error(f"Error listing people: {e}")
        return create_response(error=str(e), success=False)


@router.post("/people")
async def create_person(
    request: CreatePersonRequest,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Create a new person/executive profile."""
    try:
        person = note_manager.create_person(
            full_name=request.full_name,
            current_role=request.current_role,
            current_company=request.current_company,
            linkedin_url=request.linkedin_url,
            content=request.content,
        )

        return create_response(data=person.to_dict())

    except Exception as e:
        logger.error(f"Error creating person: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Sector Endpoints
# =============================================================================


@router.get("/sectors")
async def list_sectors(
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Get all sector overview notes."""
    try:
        sectors = note_manager.get_sectors()
        return create_response(data=[s.to_dict() for s in sectors])

    except Exception as e:
        logger.error(f"Error listing sectors: {e}")
        return create_response(error=str(e), success=False)


@router.post("/sectors")
async def create_sector(
    request: CreateSectorRequest,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """Create a new sector overview."""
    try:
        sector = note_manager.create_sector(
            sector_name=request.sector_name,
            sub_sectors=request.sub_sectors,
            companies=request.companies,
            content=request.content,
        )

        return create_response(data=sector.to_dict())

    except Exception as e:
        logger.error(f"Error creating sector: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Daily Notes Endpoints
# =============================================================================


@router.post("/daily")
async def create_daily_note(
    date: Optional[str] = None,
    content: Optional[str] = None,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    Create a daily note.

    Args:
        date: Date (ISO format, defaults to today)
        content: Optional custom content
    """
    try:
        note = note_manager.create_daily_note(date=date, content=content)
        return create_response(data=note.to_dict())

    except Exception as e:
        logger.error(f"Error creating daily note: {e}")
        return create_response(error=str(e), success=False)


# =============================================================================
# Company Research Endpoints
# =============================================================================


@router.post("/companies")
async def create_company(
    symbol: str,
    company_name: str = "",
    sector: str = "",
    content: Optional[str] = None,
    note_manager: NoteManager = Depends(get_note_manager),
):
    """
    Create a company research note.

    Args:
        symbol: Stock symbol
        company_name: Full company name
        sector: Sector/industry
        content: Optional custom content
    """
    try:
        note = note_manager.create_company(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            content=content,
        )
        return create_response(data=note.to_dict())

    except Exception as e:
        logger.error(f"Error creating company note: {e}")
        return create_response(error=str(e), success=False)
