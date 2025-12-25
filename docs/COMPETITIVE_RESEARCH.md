# Stanley Competitive Research & Feature Roadmap

## Executive Summary

This document analyzes competing investment research platforms to identify features that could enhance Stanley as the ultimate fundamental analyst software companion. Research conducted December 2025.

---

## 1. Professional Investment Research Platforms

### Tier 1: Enterprise Platforms ($20K+/year)

| Platform | Key Differentiator | Features to Consider for Stanley |
|----------|-------------------|----------------------------------|
| **Bloomberg Terminal** | Industry standard, real-time data | Chat/messaging, universal search, Excel integration |
| **FactSet Workstation** | Quantitative research focus | Portfolio analytics, data management, customizable dashboards |
| **S&P Capital IQ Pro** | Private company data | M&A transactions, ownership data, granular financials |
| **LSEG Workspace** | Macro + broker research | Datastream macroeconomic data, ESG metrics, Reuters news |

### Tier 2: Mid-Market Platforms ($500-2K/month)

| Platform | Key Differentiator | Features to Consider for Stanley |
|----------|-------------------|----------------------------------|
| **[AlphaSense](https://www.alpha-sense.com/)** | AI-powered search across documents | NLP search across filings, expert transcripts, custom alerts |
| **[Sentieo](https://sentieo.com/)** (now AlphaSense) | SEC filing search engine | Document annotation, research notebook, Excel plug-in |
| **Morningstar Direct** | Fund flow data, analyst ratings | Proprietary ratings, diversity data, pre-built datasets |

### Tier 3: Individual Investor Platforms ($0-500/month)

| Platform | Key Differentiator | Features to Consider for Stanley |
|----------|-------------------|----------------------------------|
| **[Koyfin](https://www.koyfin.com/)** | Best Bloomberg alternative for individuals | Customizable dashboards, 30+ years data, affordable pricing |
| **[Fiscal.ai](https://fiscal.ai/)** (fka FinChat) | AI-powered financial chat | Conversational queries, verified data, human-reviewed |
| **[Seeking Alpha](https://seekingalpha.com/)** | Crowdsourced analysis | Quant ratings, earnings transcripts, community insights |
| **[Stock Rover](https://www.stockrover.com/)** | 650+ metrics, powerful screening | 10 years historical data, unmatched screening |
| **[Ziggma](https://ziggma.com/)** | Industry-specific KPIs | Banking/REIT/insurance specialized metrics |

---

## 2. AI-Powered Research Assistants

### Key Players

| Tool | Capability | Stanley Integration Opportunity |
|------|------------|--------------------------------|
| **[Fiscal.ai Copilot](https://fiscal.ai/)** | Scores 2-4x higher than GPT-4/Claude on FinanceBench | Train specialized finance model, verified data citations |
| **[FinRobot](https://github.com/AI4Finance-Foundation/FinRobot)** | Open-source AI agent platform | Multi-agent research workflows, extensible architecture |
| **[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)** | Open-source finance LLM fine-tuning | Fine-tune models on SEC filings, earnings calls |
| **AlphaSense Smart Synonyms** | NLP for financial terminology | Semantic search across documents |

### Feature Ideas from AI Research Tools

1. **Conversational Queries**: "What are AAPL's biggest risk factors?" → Dashboard generation
2. **Citation Tracking**: Every AI response linked to source documents
3. **Sentiment Analysis**: Automated sentiment scoring of earnings calls
4. **Anomaly Detection**: Flag unusual patterns in financials

---

## 3. Note-Taking & Knowledge Management

### Platform Comparison

| Tool | Strength | Weakness | Stanley Approach |
|------|----------|----------|------------------|
| **[Obsidian](https://obsidian.md/)** | Knowledge graph, local storage, privacy | No native collaboration | Bi-directional linking between research notes |
| **[Notion](https://notion.so/)** | Collaboration, databases, templates | Cloud-only, less privacy | Structured templates for investment theses |
| **[Anytype](https://anytype.io/)** | Open-source, best of both | Newer, smaller ecosystem | Consider for inspiration |

### Investment-Specific Note-Taking Features

From [Reasonable Deviations blog](https://reasonabledeviations.com/2021/09/18/how-i-use-notion/):

1. **Investment Thesis Template**
   - Entry criteria and price targets
   - Risk factors and position sizing
   - Links to all research sources

2. **Trade Journal**
   - Open/closed positions table
   - Success/failure tagging
   - Performance attribution

3. **Research Database**
   - Subpages per asset (tagged by asset class)
   - Structured checklists for due diligence
   - Links to earnings transcripts, filings

### Stanley Note-Taking Module Design

```
stanley/notes/
├── thesis.py       # Investment thesis templates
├── journal.py      # Trade journal with P&L tracking
├── linking.py      # Bi-directional links between entities
└── search.py       # Full-text search across all notes
```

---

## 4. SEC Filings & 13F Analysis

### Data Sources

| Source | Coverage | Cost | Notes |
|--------|----------|------|-------|
| **SEC EDGAR** | Official source | Free | Raw XML/HTML, requires parsing |
| **[WhaleWisdom](https://whalewisdom.com/)** | 20+ years 13F data | Paid | WhaleScore, WhaleIndex tools |
| **[13F Pro](https://www.13fpro.com/)** | Comprehensive 13F tracking | Paid | Daily processing, portfolio monitoring |
| **[SEC-API](https://sec-api.io/)** | 1994+ filings | Paid | REST API, structured data |
| **[edgartools](https://github.com/dgunning/edgartools)** | Python library | Free | Already integrated in Stanley |

### 13F Analysis Features to Add

1. **Whale Tracking Dashboard**: Monitor top institutional managers
2. **Position Changes Alerts**: New positions, exits, significant changes
3. **Crowded Trade Detection**: Identify when many institutions hold same stock
4. **Hedge Fund Cloning**: Replicate top manager portfolios
5. **Ownership Concentration Risk**: Alert when few holders control float

### Important Limitations (from research)
- 13F filings are delayed 45 days
- No short positions disclosed
- Options positions inconsistently reported
- Not real-time, use as research input not trading signal

---

## 5. Open Source Platforms

### OpenBB Platform

[OpenBB](https://openbb.co/) is the most direct open-source competitor to Stanley.

**Architecture Insights:**
- Open Data Platform (ODP) as "connect once, consume everywhere" layer
- Python environments for quants
- Excel integration for analysts
- MCP servers for AI agents
- REST APIs for applications

**Key Features to Adopt:**
1. **Modular Data Layer**: Database integrations (MySQL, SQLite, Snowflake)
2. **Template Automation**: Papermill notebooks for ticker analysis
3. **AI Copilot Integration**: Open-sourced LLM integration
4. **30+ Years Fundamental Data**: Free data access strategy

### QuantConnect

[QuantConnect](https://www.quantconnect.com/) focuses on algorithmic trading research.

**Features Relevant to Stanley:**
- Point-in-time backtester (avoid look-ahead bias)
- 40+ alternative data vendors
- Parameter optimization on cloud compute
- Jupyter notebook environment
- Multi-asset support (stocks, ETFs, options, crypto)

---

## 6. Portfolio & Thesis Tracking

### Best Practices from Research

1. **Investment Thesis Documentation**
   - Written thesis forces accountability
   - Track predictions vs. outcomes
   - Learn from mistakes systematically

2. **Portfolio Visualization**
   - [Portfolio Visualizer](https://www.portfoliovisualizer.com/) features:
     - Monte Carlo simulation
     - Factor regression analysis
     - Efficient frontier optimization
     - Correlation analysis

3. **Trade Journaling**
   - Entry/exit reasoning
   - Emotional state tracking
   - Position sizing rationale

---

## 7. Recommended Feature Roadmap for Stanley

### Phase 1: Knowledge Management (Note-Taking)

```python
# New module: stanley/notes/
- Investment thesis templates with structured fields
- Trade journal with automatic P&L calculation
- Bi-directional linking (company ↔ thesis ↔ filings)
- Full-text search with semantic understanding
- Tagging system (sector, theme, conviction level)
```

### Phase 2: Enhanced 13F Analytics

```python
# Enhance: stanley/analytics/institutional.py
- Whale tracking dashboard
- Position change alerts
- Crowded trade detection
- Ownership concentration risk scoring
- Historical position reconstruction
```

### Phase 3: AI-Powered Research Assistant

```python
# New module: stanley/ai/
- Conversational queries ("What are AAPL's risk factors?")
- Automatic citation to source documents
- Earnings call sentiment analysis
- Financial statement anomaly detection
- Research report generation
```

### Phase 4: Document Search & Indexing

```python
# New module: stanley/search/
- Full-text search across SEC filings
- Semantic search with embeddings
- Document annotation and highlighting
- Research notebook integration
- Excel export functionality
```

### Phase 5: Portfolio Analytics Enhancement

```python
# Enhance: stanley/portfolio/
- Monte Carlo simulation
- Factor exposure analysis
- Efficient frontier optimization
- Tax-loss harvesting suggestions
- Rebalancing recommendations
```

### Phase 6: Collaboration & Sharing

```python
# New module: stanley/collaboration/
- Shared research workspaces
- Investment thesis sharing
- Peer review workflows
- Audit trail for changes
```

---

## 8. Data Architecture Recommendations

Based on OpenBB's approach, Stanley should adopt:

```
┌─────────────────────────────────────────────────────────┐
│                    Stanley Core                          │
├─────────────────────────────────────────────────────────┤
│  Data Layer (ODP-style)                                 │
│  ├── SEC EDGAR (edgartools)                             │
│  ├── Market Data (OpenBB, yfinance)                     │
│  ├── Macro Data (DBnomics)                              │
│  ├── Alternative Data (news, sentiment)                 │
│  └── User Data (notes, theses, journals)                │
├─────────────────────────────────────────────────────────┤
│  Storage Layer                                          │
│  ├── SQLite (local, default)                            │
│  ├── PostgreSQL (production)                            │
│  ├── Vector DB (embeddings for search)                  │
│  └── File System (documents, attachments)               │
├─────────────────────────────────────────────────────────┤
│  API Layer                                              │
│  ├── REST API (FastAPI)                                 │
│  ├── Python SDK                                         │
│  ├── MCP Server (for AI agents)                         │
│  └── Excel Add-in                                       │
├─────────────────────────────────────────────────────────┤
│  UI Layer                                               │
│  ├── Rust GUI (GPUI) - Desktop                          │
│  ├── Web Dashboard (future)                             │
│  └── CLI                                                │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Competitive Positioning

### Stanley's Unique Value Proposition

| Competitor Gap | Stanley Opportunity |
|----------------|---------------------|
| Bloomberg expensive ($24K/year) | Open-source, free core |
| Koyfin limited to market data | Deep fundamental + macro analysis |
| AlphaSense enterprise-only | Individual investor friendly |
| OpenBB lacks note-taking | Integrated research workflow |
| Notion not finance-specific | Purpose-built for analysts |

### Target User Personas

1. **Independent Fundamental Analyst**: Needs comprehensive data + notes
2. **Small Fund Manager**: Needs portfolio analytics + 13F tracking
3. **Finance Student**: Needs learning tools + paper trading
4. **Retail Investor**: Needs simplified interface + AI assistance

---

## Sources

### Professional Platforms
- [AlphaSense Bloomberg Alternatives](https://www.alpha-sense.com/compare/alternatives-to-bloomberg-terminal/)
- [Koyfin Bloomberg Alternatives](https://www.koyfin.com/blog/best-bloomberg-terminal-alternatives/)
- [Koyfin vs Sentieo](https://www.koyfin.com/compare/sentieo-alternative/)

### AI Research Tools
- [Fiscal.ai (FinChat)](https://fiscal.ai/)
- [FinRobot GitHub](https://github.com/AI4Finance-Foundation/FinRobot)
- [CFI AI Tools Comparison](https://corporatefinanceinstitute.com/resources/career/chatgpt-for-finance/)

### Open Source
- [OpenBB GitHub](https://github.com/OpenBB-finance/OpenBB)
- [OpenBB Blog](https://openbb.co/blog/why-open-source)
- [QuantConnect Platform](https://www.quantconnect.com/)

### Note-Taking & Knowledge Management
- [Obsidian](https://obsidian.md/)
- [Reasonable Deviations - Notion for Investing](https://reasonabledeviations.com/2021/09/18/how-i-use-notion/)
- [Notion Investing Templates](https://www.notion.com/templates/category/investing)

### 13F & SEC Data
- [Dakota 13F Database Comparison](https://www.dakota.com/resources/blog/whalewisdom-opportunity-hunter-sec-api-which-is-right-for-you)
- [SEC 13F Data Sets](https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets)
- [13F Pro](https://www.13fpro.com/)
