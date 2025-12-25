"""
Templates Module

Pre-built templates for investment thesis, trade journal,
company research, and other note types.
"""

from datetime import datetime
from typing import Optional

from .models import (
    ConvictionLevel,
    NoteFrontmatter,
    NoteType,
    ThesisFrontmatter,
    ThesisStatus,
    TradeFrontmatter,
    TradeDirection,
    TradeStatus,
)


class Templates:
    """Pre-built note templates following best practices."""

    @staticmethod
    def investment_thesis(
        symbol: str,
        company_name: str = "",
        sector: str = "",
        conviction: ConvictionLevel = ConvictionLevel.MEDIUM,
    ) -> tuple[ThesisFrontmatter, str]:
        """
        Create an investment thesis template.

        Args:
            symbol: Stock symbol
            company_name: Company name
            sector: Sector/industry
            conviction: Initial conviction level

        Returns:
            Tuple of (frontmatter, content)
        """
        frontmatter = ThesisFrontmatter(
            title=f"{symbol} Investment Thesis",
            symbol=symbol.upper(),
            company_name=company_name,
            sector=sector,
            status=ThesisStatus.RESEARCH,
            conviction=conviction,
            tags=["thesis", sector.lower().replace(" ", "-")] if sector else ["thesis"],
        )

        content = f"""# {symbol} Investment Thesis

## Executive Summary

_One paragraph summary of the investment thesis._

## Company Overview

**Company:** {company_name or symbol}
**Sector:** {sector}
**Market Cap:**
**Current Price:**

## Investment Thesis

### Core Thesis

_What is the main reason to own this stock?_

### Key Investment Merits

1. **Merit 1:**
2. **Merit 2:**
3. **Merit 3:**

## Valuation

### Current Valuation

| Metric | Value | Sector Avg | Comment |
|--------|-------|------------|---------|
| P/E | | | |
| EV/EBITDA | | | |
| P/FCF | | | |
| P/B | | | |

### Target Price Derivation

_How did you arrive at your target price?_

**Base Case:** $
**Bull Case:** $
**Bear Case:** $

## Catalysts

### Near-term (0-6 months)

-

### Medium-term (6-18 months)

-

## Risks

### Key Risks

1. **Risk 1:**
   - Mitigation:
2. **Risk 2:**
   - Mitigation:
3. **Risk 3:**
   - Mitigation:

### What Would Invalidate the Thesis?

_Under what conditions would you exit the position?_

## Competitive Analysis

### Competitive Position

_Moat, market share, competitive advantages_

### Key Competitors

- [[Competitor 1]]
- [[Competitor 2]]

## Financial Analysis

### Revenue & Growth

_Historical growth, drivers, sustainability_

### Margins & Profitability

_Gross margin, operating margin, trends_

### Balance Sheet

_Debt levels, liquidity, capital allocation_

### Cash Flow

_FCF generation, uses of cash_

## Management

### Leadership Assessment

_Quality of management, track record, alignment_

### Capital Allocation History

_Buybacks, dividends, M&A track record_

## Position Sizing

**Conviction Level:** {conviction.value}
**Suggested Position Size:** % of portfolio
**Entry Strategy:**

## Updates & Developments

### {datetime.now().strftime("%Y-%m-%d")} - Initial Thesis

_Initial thesis creation._

---

## Related Notes

- [[{sector} Sector Overview]] (if exists)
- [[Trades/{symbol}]] (trade journal entries)

## Sources

-
"""
        return frontmatter, content

    @staticmethod
    def trade_journal(
        symbol: str,
        direction: TradeDirection = TradeDirection.LONG,
        entry_price: float = 0.0,
        shares: float = 0.0,
        entry_date: Optional[datetime] = None,
    ) -> tuple[TradeFrontmatter, str]:
        """
        Create a trade journal entry template.

        Args:
            symbol: Stock symbol
            direction: Long or short
            entry_price: Entry price
            shares: Number of shares
            entry_date: Entry date

        Returns:
            Tuple of (frontmatter, content)
        """
        entry_date = entry_date or datetime.now()
        title = f"{symbol} {direction.value.title()} - {entry_date.strftime('%Y-%m-%d')}"

        frontmatter = TradeFrontmatter(
            title=title,
            symbol=symbol.upper(),
            direction=direction,
            status=TradeStatus.OPEN,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            tags=["trade", direction.value],
        )

        content = f"""# {title}

## Trade Summary

**Symbol:** {symbol.upper()}
**Direction:** {direction.value.title()}
**Entry Date:** {entry_date.strftime("%Y-%m-%d")}
**Entry Price:** ${entry_price:.2f}
**Shares:** {shares}
**Position Value:** ${entry_price * shares:.2f}

## Entry Rationale

### Why This Trade?

_What triggered the entry?_

### Link to Thesis

- [[Theses/{symbol} Investment Thesis]]

### Technical Setup (if applicable)

_Chart patterns, levels, indicators_

### Catalyst

_What catalyst are you playing?_

## Risk Management

### Stop Loss

**Stop Price:** $
**Max Loss:** $
**Risk/Reward:**

### Position Size Justification

_Why this size?_

## Trade Management Plan

### Profit Targets

1. Target 1: $ (% of position)
2. Target 2: $ (% of position)
3. Target 3: $ (% of position)

### Scaling Plan

_When/how will you add or reduce?_

## Pre-Trade Checklist

- [ ] Thesis is current and valid
- [ ] Position size appropriate for conviction
- [ ] Stop loss defined
- [ ] Profit targets set
- [ ] No upcoming binary events (unless intentional)
- [ ] Portfolio exposure acceptable

## Emotional Check-In

**Confidence Level:** /10
**Emotional State:**
**FOMO/Fear Factor:**

## Trade Updates

### {entry_date.strftime("%Y-%m-%d")} - Entry

Entered {direction.value} position at ${entry_price:.2f}

---

## Exit (Complete When Closed)

**Exit Date:**
**Exit Price:** $
**P&L:** $
**P&L %:**
**Holding Period:** days

### Exit Rationale

_Why did you exit?_

### What Went Well

-

### What Could Be Improved

-

### Lessons Learned

_Key takeaways for future trades_

### Grade: /10

_Self-assessment of trade execution_
"""
        return frontmatter, content

    @staticmethod
    def company_research(
        symbol: str,
        company_name: str = "",
        sector: str = "",
    ) -> tuple[NoteFrontmatter, str]:
        """
        Create a company research note template.

        Args:
            symbol: Stock symbol
            company_name: Company name
            sector: Sector/industry

        Returns:
            Tuple of (frontmatter, content)
        """
        frontmatter = NoteFrontmatter(
            title=f"{company_name or symbol} Research",
            note_type=NoteType.COMPANY,
            tags=["company", sector.lower().replace(" ", "-")] if sector else ["company"],
            extra={"symbol": symbol.upper(), "sector": sector},
        )

        content = f"""# {company_name or symbol} Research

## Company Profile

**Symbol:** {symbol.upper()}
**Company:** {company_name}
**Sector:** {sector}
**Industry:**
**Founded:**
**Headquarters:**
**Employees:**
**Website:**

## Business Description

_What does the company do?_

## Business Segments

| Segment | Revenue % | Description |
|---------|-----------|-------------|
| | | |

## Key Metrics

| Metric | Value | Trend |
|--------|-------|-------|
| Revenue | | |
| Net Income | | |
| FCF | | |
| Gross Margin | | |
| Operating Margin | | |
| ROE | | |
| Debt/Equity | | |

## Competitive Position

### Market Share

### Competitors

- [[Competitor 1]]
- [[Competitor 2]]
- [[Competitor 3]]

### Competitive Advantages (Moat)

1.
2.
3.

## Management

**CEO:**
**CFO:**
**Key Executives:**

### Insider Ownership

### Compensation Alignment

## Recent Developments

### Earnings History

| Quarter | EPS | Rev | Surprise |
|---------|-----|-----|----------|
| | | | |

### News & Events

-

## SEC Filings

- [[10-K {datetime.now().year}]]
- [[10-Q Latest]]
- [[Proxy Statement]]

## Institutional Ownership

### Top Holders

1.
2.
3.

### Recent Changes

## Analyst Coverage

| Analyst | Rating | Target |
|---------|--------|--------|
| | | |

## Related Notes

- [[Theses/{symbol} Investment Thesis]]
- [[{sector} Sector Overview]]
"""
        return frontmatter, content

    @staticmethod
    def sector_overview(
        sector: str,
        description: str = "",
    ) -> tuple[NoteFrontmatter, str]:
        """
        Create a sector overview template.

        Args:
            sector: Sector name
            description: Brief description

        Returns:
            Tuple of (frontmatter, content)
        """
        frontmatter = NoteFrontmatter(
            title=f"{sector} Sector Overview",
            note_type=NoteType.SECTOR,
            tags=["sector", sector.lower().replace(" ", "-")],
        )

        content = f"""# {sector} Sector Overview

## Sector Description

{description or "_Overview of the sector and its role in the economy._"}

## Key Characteristics

- **Cyclicality:**
- **Capital Intensity:**
- **Regulatory Environment:**
- **Growth Profile:**

## Industry Structure

### Market Size

**TAM:** $
**Growth Rate:** %

### Value Chain

_Description of the industry value chain_

### Key Players

| Company | Market Cap | Market Share |
|---------|------------|--------------|
| [[Company 1]] | | |
| [[Company 2]] | | |
| [[Company 3]] | | |

## Sector Drivers

### Demand Drivers

1.
2.
3.

### Supply Factors

1.
2.

## Sector Risks

1.
2.
3.

## Valuation Benchmarks

| Metric | Sector Avg | Range |
|--------|------------|-------|
| P/E | | |
| EV/EBITDA | | |
| P/B | | |
| Div Yield | | |

## Current Sector View

**Outlook:** Bullish / Neutral / Bearish
**Reasoning:**

## Macro Sensitivity

### Interest Rates

### Economic Cycle

### Commodity Prices

## ETFs & Indices

| Symbol | Name | Expense |
|--------|------|---------|
| | | |

## Companies Covered

```dataview
TABLE symbol, status, conviction
FROM "Theses"
WHERE sector = "{sector}"
SORT modified DESC
```

## Recent Sector News

-

## Related Notes

- [[Macro Environment]]
"""
        return frontmatter, content

    @staticmethod
    def daily_note(
        date: Optional[datetime] = None,
    ) -> tuple[NoteFrontmatter, str]:
        """
        Create a daily note template.

        Args:
            date: Date for the note (defaults to today)

        Returns:
            Tuple of (frontmatter, content)
        """
        date = date or datetime.now()
        title = date.strftime("%Y-%m-%d")

        frontmatter = NoteFrontmatter(
            title=title,
            note_type=NoteType.DAILY,
            tags=["daily"],
        )

        content = f"""# {title}

## Market Overview

### Major Indices

| Index | Close | Change |
|-------|-------|--------|
| S&P 500 | | |
| Nasdaq | | |
| Russell 2000 | | |
| VIX | | |

### Sector Performance

_Best/worst sectors today_

## Portfolio Review

### Open Positions

_Quick check on open positions_

### P&L Today

**Realized:** $
**Unrealized:** $

## Market Notes

### Key Themes

-

### Notable Movers

-

### Economic Data

-

## Trade Ideas

### New Ideas

-

### Watchlist Updates

-

## Research Notes

_Notes from research done today_

## Tomorrow's Focus

- [ ]
- [ ]

## Reflection

_What went well? What could improve?_

---

<< [[{(date.replace(day=date.day-1) if date.day > 1 else date).strftime("%Y-%m-%d")}]] | [[{(date.replace(day=date.day+1)).strftime("%Y-%m-%d")}]] >>
"""
        return frontmatter, content

    @staticmethod
    def meeting_notes(
        title: str,
        company: str = "",
        attendees: str = "",
    ) -> tuple[NoteFrontmatter, str]:
        """
        Create a meeting notes template.

        Args:
            title: Meeting title
            company: Company discussed
            attendees: Meeting attendees

        Returns:
            Tuple of (frontmatter, content)
        """
        frontmatter = NoteFrontmatter(
            title=title,
            note_type=NoteType.MEETING,
            tags=["meeting"],
            extra={"company": company, "attendees": attendees},
        )

        date = datetime.now().strftime("%Y-%m-%d")

        content = f"""# {title}

**Date:** {date}
**Company:** {company}
**Attendees:** {attendees}
**Type:** Earnings Call / Investor Day / 1-on-1 / Conference

## Key Takeaways

1.
2.
3.

## Detailed Notes

### Topic 1

### Topic 2

### Topic 3

## Q&A Highlights

### Question 1

**Q:**
**A:**

## Management Tone

_Assessment of management confidence, body language, etc._

## Implications for Thesis

_How does this affect your investment thesis?_

## Action Items

- [ ]
- [ ]

## Related Notes

- [[Companies/{company}]]
- [[Theses/{company} Investment Thesis]]
"""
        return frontmatter, content
