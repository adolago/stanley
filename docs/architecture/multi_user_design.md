# Stanley Multi-User Architecture Design

## Executive Summary

This document presents a comprehensive architecture for evolving Stanley from a single-user investment analysis platform to a multi-user, team-capable system with enterprise-grade authentication, authorization, data isolation, and collaboration features.

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Authentication Architecture](#authentication-architecture)
3. [User Profile System](#user-profile-system)
4. [Permission System](#permission-system)
5. [Data Isolation](#data-isolation)
6. [Shared Resources](#shared-resources)
7. [Team Features](#team-features)
8. [Audit Logging](#audit-logging)
9. [Session Management](#session-management)
10. [API Key Management](#api-key-management)
11. [Rate Limiting](#rate-limiting)
12. [Database Schema](#database-schema)
13. [Implementation Phases](#implementation-phases)
14. [Architecture Decision Records](#architecture-decision-records)

---

## Current State Analysis

### Existing Architecture

```
+------------------+     +-------------------+     +------------------+
|   stanley-gui    |---->|    FastAPI API    |---->|    Analyzers     |
|   (Rust/GPUI)    |     |   (main.py)       |     | (20+ analyzers)  |
+------------------+     +-------------------+     +------------------+
                               |                         |
                               v                         v
                         +----------+             +-------------+
                         | AppState |             | DataManager |
                         | (global) |             |  (OpenBB)   |
                         +----------+             +-------------+
                               |
                               v
                         +-------------+
                         |   Vault     |
                         | (Notes/DB)  |
                         +-------------+
```

### Key Observations

1. **Single Global State**: `app_state = AppState()` is a module-level singleton
2. **No Authentication**: API endpoints have no auth middleware
3. **Shared Analyzers**: All analyzers are initialized once and shared
4. **Single Vault**: Notes system uses one vault path
5. **No User Context**: Requests carry no user identity
6. **CORS Only**: Security limited to origin restrictions

### Components Requiring Multi-User Adaptation

| Component | Current State | Multi-User Requirement |
|-----------|--------------|------------------------|
| AppState | Global singleton | User-scoped or request-scoped |
| NoteManager/Vault | Single path | Per-user vault isolation |
| Portfolio holdings | None persisted | Per-user storage |
| API endpoints | No auth | JWT/OAuth middleware |
| Data providers | Shared | Shared with user quotas |
| SQLite databases | Single file | Schema with user_id FK |

---

## Authentication Architecture

### Overview

Implement a layered authentication system supporting:
- Local username/password authentication
- OAuth 2.0 providers (Google, GitHub, Microsoft)
- API key authentication for programmatic access
- JWT tokens for session management

### Component Diagram

```
+------------------+
|     Client       |
+--------+---------+
         |
         v
+--------+---------+
|  Auth Middleware |<--+
+--------+---------+   |
         |             |
         v             |
+--------+---------+   |    +------------------+
|  Auth Provider   +---+--->| Token Validator  |
+------------------+        +------------------+
         |                          |
         v                          v
+--------+---------+        +-------+--------+
|  Local Auth      |        | OAuth Provider |
|  (Argon2 hash)   |        | (Google/GitHub)|
+------------------+        +----------------+
         |                          |
         +------------+-------------+
                      |
                      v
              +-------+--------+
              |  User Service  |
              +-------+--------+
                      |
                      v
              +-------+--------+
              |   User Store   |
              |  (PostgreSQL)  |
              +----------------+
```

### Authentication Flow

#### Local Authentication

```python
# Proposed: stanley/auth/local.py

from argon2 import PasswordHasher
from pydantic import BaseModel, EmailStr

class LocalCredentials(BaseModel):
    email: EmailStr
    password: str

class LocalAuthProvider:
    """Local username/password authentication."""

    def __init__(self, user_repository: UserRepository):
        self.user_repo = user_repository
        self.hasher = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4
        )

    async def authenticate(self, credentials: LocalCredentials) -> AuthResult:
        """Authenticate user with email/password."""
        user = await self.user_repo.get_by_email(credentials.email)
        if not user:
            # Timing-safe comparison to prevent user enumeration
            self.hasher.hash("dummy_password")
            return AuthResult(success=False, error="Invalid credentials")

        try:
            self.hasher.verify(user.password_hash, credentials.password)
            if self.hasher.check_needs_rehash(user.password_hash):
                new_hash = self.hasher.hash(credentials.password)
                await self.user_repo.update_password_hash(user.id, new_hash)
            return AuthResult(success=True, user=user)
        except argon2.exceptions.VerifyMismatchError:
            return AuthResult(success=False, error="Invalid credentials")

    async def register(
        self,
        email: str,
        password: str,
        name: str
    ) -> User:
        """Register new user with local credentials."""
        if await self.user_repo.get_by_email(email):
            raise UserExistsError(f"User with email {email} already exists")

        password_hash = self.hasher.hash(password)
        return await self.user_repo.create(
            email=email,
            password_hash=password_hash,
            name=name,
            auth_provider="local"
        )
```

#### OAuth 2.0 Integration

```python
# Proposed: stanley/auth/oauth.py

from authlib.integrations.starlette_client import OAuth
from starlette.config import Config

class OAuthConfig:
    """OAuth provider configuration."""

    PROVIDERS = {
        "google": {
            "server_metadata_url":
                "https://accounts.google.com/.well-known/openid-configuration",
            "client_kwargs": {"scope": "openid email profile"}
        },
        "github": {
            "authorize_url": "https://github.com/login/oauth/authorize",
            "access_token_url": "https://github.com/login/oauth/access_token",
            "api_base_url": "https://api.github.com/",
            "client_kwargs": {"scope": "read:user user:email"}
        },
        "microsoft": {
            "server_metadata_url":
                "https://login.microsoftonline.com/common/v2.0/"
                ".well-known/openid-configuration",
            "client_kwargs": {"scope": "openid email profile"}
        }
    }

class OAuthProvider:
    """OAuth 2.0 authentication provider."""

    def __init__(self, user_repository: UserRepository):
        self.user_repo = user_repository
        self.oauth = OAuth()
        self._register_providers()

    def _register_providers(self):
        for name, config in OAuthConfig.PROVIDERS.items():
            client_id = os.getenv(f"{name.upper()}_CLIENT_ID")
            client_secret = os.getenv(f"{name.upper()}_CLIENT_SECRET")
            if client_id and client_secret:
                self.oauth.register(
                    name=name,
                    client_id=client_id,
                    client_secret=client_secret,
                    **config
                )

    async def handle_callback(
        self,
        provider: str,
        request: Request
    ) -> AuthResult:
        """Handle OAuth callback and create/update user."""
        client = self.oauth.create_client(provider)
        token = await client.authorize_access_token(request)

        if provider == "google":
            user_info = token.get("userinfo")
        elif provider == "github":
            resp = await client.get("user")
            user_info = resp.json()
        else:
            user_info = await client.userinfo(token=token)

        # Create or update user
        user = await self.user_repo.get_by_oauth_id(
            provider, user_info["id"]
        )
        if not user:
            user = await self.user_repo.create(
                email=user_info.get("email"),
                name=user_info.get("name"),
                oauth_provider=provider,
                oauth_id=str(user_info["id"]),
                avatar_url=user_info.get("avatar_url") or
                           user_info.get("picture")
            )

        return AuthResult(success=True, user=user)
```

#### JWT Token Management

```python
# Proposed: stanley/auth/tokens.py

import jwt
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

class TokenConfig:
    """JWT token configuration."""

    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    ALGORITHM = "HS256"

class TokenService:
    """JWT token generation and validation."""

    def __init__(self, secret_key: str, issuer: str = "stanley"):
        self.secret_key = secret_key
        self.issuer = issuer

    def create_access_token(
        self,
        user_id: UUID,
        permissions: list[str],
        team_id: Optional[UUID] = None
    ) -> str:
        """Create short-lived access token."""
        now = datetime.utcnow()
        payload = {
            "sub": str(user_id),
            "iss": self.issuer,
            "iat": now,
            "exp": now + timedelta(
                minutes=TokenConfig.ACCESS_TOKEN_EXPIRE_MINUTES
            ),
            "jti": str(uuid4()),
            "permissions": permissions,
            "team_id": str(team_id) if team_id else None,
            "type": "access"
        }
        return jwt.encode(payload, self.secret_key,
                         algorithm=TokenConfig.ALGORITHM)

    def create_refresh_token(self, user_id: UUID) -> str:
        """Create long-lived refresh token."""
        now = datetime.utcnow()
        payload = {
            "sub": str(user_id),
            "iss": self.issuer,
            "iat": now,
            "exp": now + timedelta(
                days=TokenConfig.REFRESH_TOKEN_EXPIRE_DAYS
            ),
            "jti": str(uuid4()),
            "type": "refresh"
        }
        return jwt.encode(payload, self.secret_key,
                         algorithm=TokenConfig.ALGORITHM)

    def validate_token(self, token: str) -> TokenPayload:
        """Validate and decode token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[TokenConfig.ALGORITHM],
                issuer=self.issuer
            )
            return TokenPayload(**payload)
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(f"Invalid token: {e}")
```

---

## User Profile System

### User Model

```python
# Proposed: stanley/users/models.py

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, EmailStr, Field

class SubscriptionTier(str, Enum):
    """User subscription tiers with feature access."""
    FREE = "free"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"
    ENTERPRISE = "enterprise"

class UserStatus(str, Enum):
    """User account status."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"

class UserPreferences(BaseModel):
    """User preferences and settings."""

    theme: str = "system"
    default_currency: str = "USD"
    timezone: str = "UTC"
    date_format: str = "YYYY-MM-DD"

    # Analysis defaults
    default_lookback_days: int = 63
    default_benchmark: str = "SPY"
    risk_free_rate_source: str = "13_week_treasury"

    # Notification preferences
    email_notifications: bool = True
    alert_notifications: bool = True
    digest_frequency: str = "daily"  # daily, weekly, none

    # Display preferences
    show_mock_data_warning: bool = True
    compact_tables: bool = False
    chart_animations: bool = True

class User(BaseModel):
    """User entity."""

    id: UUID = Field(default_factory=uuid4)
    email: EmailStr
    name: str

    # Authentication
    password_hash: Optional[str] = None
    oauth_provider: Optional[str] = None
    oauth_id: Optional[str] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None

    # Profile
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None
    location: Optional[str] = None

    # Status
    status: UserStatus = UserStatus.PENDING
    email_verified: bool = False
    email_verified_at: Optional[datetime] = None

    # Subscription
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    subscription_expires_at: Optional[datetime] = None

    # Preferences
    preferences: UserPreferences = Field(
        default_factory=UserPreferences
    )

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None

    # Teams
    team_memberships: list["TeamMembership"] = Field(default_factory=list)

    class Config:
        from_attributes = True
```

### User Repository

```python
# Proposed: stanley/users/repository.py

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import Optional
from uuid import UUID

class UserRepository:
    """User data access layer."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **kwargs) -> User:
        """Create new user."""
        user = UserModel(**kwargs)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return User.model_validate(user)

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        user = result.scalar_one_or_none()
        return User.model_validate(user) if user else None

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        user = result.scalar_one_or_none()
        return User.model_validate(user) if user else None

    async def get_by_oauth_id(
        self,
        provider: str,
        oauth_id: str
    ) -> Optional[User]:
        """Get user by OAuth provider and ID."""
        result = await self.session.execute(
            select(UserModel).where(
                UserModel.oauth_provider == provider,
                UserModel.oauth_id == oauth_id
            )
        )
        user = result.scalar_one_or_none()
        return User.model_validate(user) if user else None

    async def update(self, user_id: UUID, **kwargs) -> User:
        """Update user."""
        kwargs["updated_at"] = datetime.utcnow()
        await self.session.execute(
            update(UserModel)
            .where(UserModel.id == user_id)
            .values(**kwargs)
        )
        await self.session.commit()
        return await self.get_by_id(user_id)
```

---

## Permission System

### Role-Based Access Control (RBAC)

```python
# Proposed: stanley/auth/permissions.py

from enum import Enum
from typing import Set

class Permission(str, Enum):
    """System permissions."""

    # Resource access
    READ_MARKET_DATA = "read:market_data"
    READ_INSTITUTIONAL_DATA = "read:institutional_data"
    READ_OPTIONS_FLOW = "read:options_flow"
    READ_DARK_POOL = "read:dark_pool"
    READ_SEC_FILINGS = "read:sec_filings"
    READ_MACRO_DATA = "read:macro_data"
    READ_COMMODITIES = "read:commodities"

    # Portfolio
    READ_OWN_PORTFOLIO = "read:own_portfolio"
    WRITE_OWN_PORTFOLIO = "write:own_portfolio"
    READ_TEAM_PORTFOLIO = "read:team_portfolio"
    WRITE_TEAM_PORTFOLIO = "write:team_portfolio"

    # Notes/Research
    READ_OWN_NOTES = "read:own_notes"
    WRITE_OWN_NOTES = "write:own_notes"
    READ_SHARED_NOTES = "read:shared_notes"
    WRITE_SHARED_NOTES = "write:shared_notes"

    # Watchlists/Screens
    READ_OWN_WATCHLISTS = "read:own_watchlists"
    WRITE_OWN_WATCHLISTS = "write:own_watchlists"
    READ_SHARED_WATCHLISTS = "read:shared_watchlists"
    WRITE_SHARED_WATCHLISTS = "write:shared_watchlists"

    # Signals
    READ_SIGNALS = "read:signals"
    CREATE_SIGNALS = "create:signals"
    BACKTEST_SIGNALS = "backtest:signals"

    # Admin
    MANAGE_TEAM = "manage:team"
    MANAGE_USERS = "manage:users"
    VIEW_AUDIT_LOG = "view:audit_log"
    MANAGE_API_KEYS = "manage:api_keys"
    ADMIN_SETTINGS = "admin:settings"

class Role(str, Enum):
    """System roles with associated permissions."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    SENIOR_ANALYST = "senior_analyst"
    PORTFOLIO_MANAGER = "portfolio_manager"
    TEAM_ADMIN = "team_admin"
    SYSTEM_ADMIN = "system_admin"

ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_MARKET_DATA,
        Permission.READ_OWN_NOTES,
        Permission.READ_SHARED_NOTES,
        Permission.READ_OWN_WATCHLISTS,
        Permission.READ_SHARED_WATCHLISTS,
    },
    Role.ANALYST: {
        # Includes VIEWER permissions plus:
        Permission.READ_MARKET_DATA,
        Permission.READ_INSTITUTIONAL_DATA,
        Permission.READ_OPTIONS_FLOW,
        Permission.READ_DARK_POOL,
        Permission.READ_SEC_FILINGS,
        Permission.READ_MACRO_DATA,
        Permission.READ_COMMODITIES,
        Permission.READ_OWN_PORTFOLIO,
        Permission.WRITE_OWN_PORTFOLIO,
        Permission.READ_OWN_NOTES,
        Permission.WRITE_OWN_NOTES,
        Permission.READ_SHARED_NOTES,
        Permission.READ_OWN_WATCHLISTS,
        Permission.WRITE_OWN_WATCHLISTS,
        Permission.READ_SHARED_WATCHLISTS,
        Permission.READ_SIGNALS,
    },
    Role.SENIOR_ANALYST: {
        # Includes ANALYST permissions plus:
        Permission.WRITE_SHARED_NOTES,
        Permission.WRITE_SHARED_WATCHLISTS,
        Permission.CREATE_SIGNALS,
        Permission.BACKTEST_SIGNALS,
    },
    Role.PORTFOLIO_MANAGER: {
        # Includes SENIOR_ANALYST permissions plus:
        Permission.READ_TEAM_PORTFOLIO,
        Permission.WRITE_TEAM_PORTFOLIO,
        Permission.VIEW_AUDIT_LOG,
    },
    Role.TEAM_ADMIN: {
        # All permissions except system admin
        Permission.MANAGE_TEAM,
        Permission.MANAGE_API_KEYS,
    },
    Role.SYSTEM_ADMIN: {
        # All permissions
        Permission.MANAGE_USERS,
        Permission.ADMIN_SETTINGS,
    },
}

def get_role_permissions(role: Role) -> Set[Permission]:
    """Get all permissions for a role including inherited permissions."""
    role_hierarchy = {
        Role.VIEWER: [],
        Role.ANALYST: [Role.VIEWER],
        Role.SENIOR_ANALYST: [Role.ANALYST],
        Role.PORTFOLIO_MANAGER: [Role.SENIOR_ANALYST],
        Role.TEAM_ADMIN: [Role.PORTFOLIO_MANAGER],
        Role.SYSTEM_ADMIN: [Role.TEAM_ADMIN],
    }

    permissions = set(ROLE_PERMISSIONS.get(role, set()))
    for parent_role in role_hierarchy.get(role, []):
        permissions |= get_role_permissions(parent_role)

    return permissions
```

### Permission Middleware

```python
# Proposed: stanley/auth/middleware.py

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Extract and validate current user from JWT token."""
    token_service = request.app.state.token_service
    user_repo = request.app.state.user_repository

    try:
        payload = token_service.validate_token(credentials.credentials)
        if payload.type != "access":
            raise HTTPException(
                status_code=401,
                detail="Invalid token type"
            )

        user = await user_repo.get_by_id(UUID(payload.sub))
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        if user.status != UserStatus.ACTIVE:
            raise HTTPException(
                status_code=403,
                detail="Account is not active"
            )

        # Attach permissions to request state
        request.state.user = user
        request.state.permissions = get_role_permissions(user.role)

        return user

    except TokenExpiredError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e)
        )

def require_permission(*permissions: Permission):
    """Dependency that requires specific permissions."""
    async def permission_checker(
        request: Request,
        user: User = Depends(get_current_user)
    ):
        user_permissions = request.state.permissions
        missing = set(permissions) - user_permissions

        if missing:
            raise HTTPException(
                status_code=403,
                detail=f"Missing permissions: {', '.join(p.value for p in missing)}"
            )

        return user

    return permission_checker
```

---

## Data Isolation

### Multi-Tenant Architecture

Stanley will use a **schema-based isolation** approach for PostgreSQL, combined with **row-level security** for shared tables.

```
+-------------------+
|   PostgreSQL DB   |
+-------------------+
        |
        +---------------------------+
        |                           |
+-------v-------+         +---------v---------+
|  public       |         |   tenant_{uuid}   |
|  (shared)     |         |   (per-user)      |
+---------------+         +-------------------+
| - users       |         | - portfolios      |
| - teams       |         | - watchlists      |
| - api_keys    |         | - notes           |
| - audit_logs  |         | - signals         |
| - rate_limits |         | - screens         |
+---------------+         +-------------------+
```

### Row-Level Security

```sql
-- Enable RLS on shared tables
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;

-- Users can only see their own portfolios
CREATE POLICY portfolio_isolation ON portfolios
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Team members can see team portfolios
CREATE POLICY portfolio_team_access ON portfolios
    FOR SELECT
    USING (
        user_id = current_setting('app.current_user_id')::uuid
        OR team_id IN (
            SELECT team_id FROM team_memberships
            WHERE user_id = current_setting('app.current_user_id')::uuid
        )
    );
```

### User-Scoped State

```python
# Proposed: stanley/core/context.py

from contextvars import ContextVar
from typing import Optional
from uuid import UUID

class UserContext:
    """User context for request scope."""

    def __init__(
        self,
        user_id: UUID,
        team_id: Optional[UUID] = None,
        permissions: set[Permission] = None
    ):
        self.user_id = user_id
        self.team_id = team_id
        self.permissions = permissions or set()

# Context variable for current user
_user_context: ContextVar[Optional[UserContext]] = ContextVar(
    "user_context",
    default=None
)

def get_user_context() -> UserContext:
    """Get current user context."""
    ctx = _user_context.get()
    if ctx is None:
        raise RuntimeError("No user context set")
    return ctx

def set_user_context(context: UserContext):
    """Set current user context."""
    return _user_context.set(context)
```

### User-Scoped Services

```python
# Proposed: stanley/core/scoped_state.py

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

@dataclass
class UserScopedState:
    """Per-user analyzer and service instances."""

    user_id: UUID
    vault: Vault
    portfolio_analyzer: PortfolioAnalyzer
    note_manager: NoteManager
    signal_generator: SignalGenerator

    # Shared analyzers (read-only, can be shared)
    data_manager: DataManager
    money_flow_analyzer: MoneyFlowAnalyzer
    institutional_analyzer: InstitutionalAnalyzer
    research_analyzer: ResearchAnalyzer
    commodities_analyzer: CommoditiesAnalyzer

class UserStateManager:
    """Manage user-scoped state with caching."""

    def __init__(
        self,
        shared_state: SharedState,
        vault_base_path: Path
    ):
        self.shared_state = shared_state
        self.vault_base_path = vault_base_path
        self._cache: dict[UUID, UserScopedState] = {}
        self._cache_ttl = 3600  # 1 hour

    async def get_user_state(self, user_id: UUID) -> UserScopedState:
        """Get or create user-scoped state."""
        if user_id in self._cache:
            return self._cache[user_id]

        # Create user-specific vault
        user_vault_path = self.vault_base_path / str(user_id)
        vault = Vault(user_vault_path)

        # Create user-scoped analyzers
        state = UserScopedState(
            user_id=user_id,
            vault=vault,
            portfolio_analyzer=PortfolioAnalyzer(
                self.shared_state.data_manager,
                user_id=user_id
            ),
            note_manager=NoteManager(vault),
            signal_generator=SignalGenerator(
                money_flow_analyzer=self.shared_state.money_flow_analyzer,
                institutional_analyzer=self.shared_state.institutional_analyzer,
                research_analyzer=self.shared_state.research_analyzer,
                portfolio_analyzer=PortfolioAnalyzer(
                    self.shared_state.data_manager,
                    user_id=user_id
                ),
                data_manager=self.shared_state.data_manager,
            ),
            # Shared (read-only) analyzers
            data_manager=self.shared_state.data_manager,
            money_flow_analyzer=self.shared_state.money_flow_analyzer,
            institutional_analyzer=self.shared_state.institutional_analyzer,
            research_analyzer=self.shared_state.research_analyzer,
            commodities_analyzer=self.shared_state.commodities_analyzer,
        )

        self._cache[user_id] = state
        return state

    async def invalidate_user_state(self, user_id: UUID):
        """Remove user state from cache."""
        self._cache.pop(user_id, None)
```

---

## Shared Resources

### Watchlist System

```python
# Proposed: stanley/watchlists/models.py

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class WatchlistVisibility(str, Enum):
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"

class WatchlistItem(BaseModel):
    """Individual item in a watchlist."""

    symbol: str
    added_at: datetime = Field(default_factory=datetime.utcnow)
    added_by: UUID
    notes: Optional[str] = None
    target_price: Optional[float] = None
    alert_enabled: bool = False

class Watchlist(BaseModel):
    """User or team watchlist."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None

    # Ownership
    owner_id: UUID
    team_id: Optional[UUID] = None
    visibility: WatchlistVisibility = WatchlistVisibility.PRIVATE

    # Items
    items: list[WatchlistItem] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Sharing
    shared_with: list[UUID] = Field(default_factory=list)

class WatchlistRepository:
    """Watchlist data access with sharing support."""

    async def get_accessible_watchlists(
        self,
        user_id: UUID,
        team_id: Optional[UUID] = None
    ) -> list[Watchlist]:
        """Get all watchlists accessible to user."""
        query = select(WatchlistModel).where(
            or_(
                # User's own watchlists
                WatchlistModel.owner_id == user_id,
                # Shared with user
                WatchlistModel.shared_with.contains([user_id]),
                # Team watchlists
                and_(
                    WatchlistModel.team_id == team_id,
                    WatchlistModel.visibility.in_([
                        WatchlistVisibility.TEAM,
                        WatchlistVisibility.PUBLIC
                    ])
                ) if team_id else False,
                # Public watchlists
                WatchlistModel.visibility == WatchlistVisibility.PUBLIC
            )
        )
        result = await self.session.execute(query)
        return [Watchlist.model_validate(w) for w in result.scalars()]
```

### Stock Screener Sharing

```python
# Proposed: stanley/screens/models.py

class ScreenerCriteria(BaseModel):
    """Individual screening criterion."""

    field: str  # e.g., "pe_ratio", "market_cap", "institutional_ownership"
    operator: str  # "gt", "lt", "eq", "between", "in"
    value: Any

class Screener(BaseModel):
    """Saved stock screener configuration."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: Optional[str] = None

    # Criteria
    criteria: list[ScreenerCriteria] = Field(default_factory=list)
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    limit: int = 100

    # Ownership & sharing
    owner_id: UUID
    team_id: Optional[UUID] = None
    visibility: WatchlistVisibility = WatchlistVisibility.PRIVATE
    shared_with: list[UUID] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_run_at: Optional[datetime] = None
    run_count: int = 0
```

---

## Team Features

### Team Model

```python
# Proposed: stanley/teams/models.py

class TeamRole(str, Enum):
    MEMBER = "member"
    ANALYST = "analyst"
    MANAGER = "manager"
    ADMIN = "admin"
    OWNER = "owner"

class TeamMembership(BaseModel):
    """User membership in a team."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    team_id: UUID
    role: TeamRole = TeamRole.MEMBER

    joined_at: datetime = Field(default_factory=datetime.utcnow)
    invited_by: Optional[UUID] = None

class Team(BaseModel):
    """Investment team or organization."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    slug: str  # URL-safe identifier
    description: Optional[str] = None

    # Settings
    settings: TeamSettings = Field(default_factory=TeamSettings)

    # Subscription
    subscription_tier: SubscriptionTier = SubscriptionTier.PROFESSIONAL
    max_members: int = 5

    # Members
    members: list[TeamMembership] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: UUID

class TeamSettings(BaseModel):
    """Team-wide settings."""

    # Default permissions for new members
    default_role: TeamRole = TeamRole.MEMBER

    # Sharing defaults
    default_watchlist_visibility: WatchlistVisibility = WatchlistVisibility.TEAM
    default_note_visibility: WatchlistVisibility = WatchlistVisibility.TEAM

    # Portfolio settings
    enable_team_portfolio: bool = True
    team_benchmark: str = "SPY"

    # Data access
    enable_dark_pool_data: bool = True
    enable_options_flow: bool = True
    enable_sec_filings: bool = True
```

### Team Service

```python
# Proposed: stanley/teams/service.py

class TeamService:
    """Team management service."""

    async def create_team(
        self,
        name: str,
        owner_id: UUID,
        description: Optional[str] = None
    ) -> Team:
        """Create a new team."""
        slug = self._generate_slug(name)

        team = await self.team_repo.create(
            name=name,
            slug=slug,
            description=description,
            created_by=owner_id
        )

        # Add owner as first member
        await self.team_repo.add_member(
            team_id=team.id,
            user_id=owner_id,
            role=TeamRole.OWNER
        )

        # Create team vault
        await self.vault_service.create_team_vault(team.id)

        return team

    async def invite_member(
        self,
        team_id: UUID,
        email: str,
        role: TeamRole,
        invited_by: UUID
    ) -> TeamInvitation:
        """Invite user to team."""
        # Check inviter has permission
        inviter_membership = await self.team_repo.get_membership(
            team_id, invited_by
        )
        if inviter_membership.role not in [TeamRole.ADMIN, TeamRole.OWNER]:
            raise PermissionDenied("Only admins can invite members")

        # Check team capacity
        team = await self.team_repo.get_by_id(team_id)
        if len(team.members) >= team.max_members:
            raise TeamCapacityExceeded(
                f"Team is at maximum capacity ({team.max_members})"
            )

        # Create invitation
        invitation = await self.invitation_repo.create(
            team_id=team_id,
            email=email,
            role=role,
            invited_by=invited_by,
            expires_at=datetime.utcnow() + timedelta(days=7)
        )

        # Send invitation email
        await self.email_service.send_team_invitation(
            email=email,
            team=team,
            inviter=await self.user_repo.get_by_id(invited_by),
            invitation=invitation
        )

        return invitation
```

---

## Audit Logging

### Audit Event Model

```python
# Proposed: stanley/audit/models.py

from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime

class AuditEventType(str, Enum):
    """Types of auditable events."""

    # Authentication
    LOGIN = "auth.login"
    LOGOUT = "auth.logout"
    LOGIN_FAILED = "auth.login_failed"
    PASSWORD_CHANGED = "auth.password_changed"
    MFA_ENABLED = "auth.mfa_enabled"
    MFA_DISABLED = "auth.mfa_disabled"

    # User management
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_SUSPENDED = "user.suspended"

    # Team management
    TEAM_CREATED = "team.created"
    TEAM_MEMBER_ADDED = "team.member_added"
    TEAM_MEMBER_REMOVED = "team.member_removed"
    TEAM_ROLE_CHANGED = "team.role_changed"

    # Resource access
    PORTFOLIO_CREATED = "portfolio.created"
    PORTFOLIO_UPDATED = "portfolio.updated"
    PORTFOLIO_DELETED = "portfolio.deleted"
    PORTFOLIO_SHARED = "portfolio.shared"

    WATCHLIST_CREATED = "watchlist.created"
    WATCHLIST_UPDATED = "watchlist.updated"
    WATCHLIST_DELETED = "watchlist.deleted"
    WATCHLIST_SHARED = "watchlist.shared"

    NOTE_CREATED = "note.created"
    NOTE_UPDATED = "note.updated"
    NOTE_DELETED = "note.deleted"
    NOTE_SHARED = "note.shared"

    # API
    API_KEY_CREATED = "api.key_created"
    API_KEY_REVOKED = "api.key_revoked"
    API_REQUEST = "api.request"

    # Data access
    SEC_FILING_ACCESSED = "data.sec_filing"
    INSTITUTIONAL_DATA_ACCESSED = "data.institutional"
    OPTIONS_FLOW_ACCESSED = "data.options_flow"
    DARK_POOL_ACCESSED = "data.dark_pool"

class AuditEvent(BaseModel):
    """Audit log entry."""

    id: UUID = Field(default_factory=uuid4)
    event_type: AuditEventType

    # Actor
    user_id: Optional[UUID] = None
    team_id: Optional[UUID] = None

    # Target
    resource_type: Optional[str] = None
    resource_id: Optional[UUID] = None

    # Context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None

    # Details
    details: dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### Audit Logger

```python
# Proposed: stanley/audit/logger.py

class AuditLogger:
    """Structured audit logging service."""

    def __init__(
        self,
        repository: AuditRepository,
        queue: Optional[MessageQueue] = None
    ):
        self.repository = repository
        self.queue = queue  # For async logging

    async def log(
        self,
        event_type: AuditEventType,
        user_id: Optional[UUID] = None,
        team_id: Optional[UUID] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        details: Optional[dict] = None,
        request: Optional[Request] = None
    ):
        """Log an audit event."""
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            team_id=team_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=self._get_client_ip(request) if request else None,
            user_agent=request.headers.get("user-agent") if request else None,
            request_id=request.state.request_id if request else None
        )

        if self.queue:
            # Async logging for high-throughput
            await self.queue.publish("audit_events", event.model_dump())
        else:
            await self.repository.create(event)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
```

### Audit Middleware

```python
# Proposed: stanley/audit/middleware.py

class AuditMiddleware:
    """FastAPI middleware for automatic request auditing."""

    def __init__(self, app, audit_logger: AuditLogger):
        self.app = app
        self.audit_logger = audit_logger

        # Endpoints that should be audited
        self.audited_patterns = [
            r"/api/institutional/.*",
            r"/api/dark-pool/.*",
            r"/api/options-flow/.*",
            r"/api/sec-filings/.*",
        ]

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path

        # Check if this endpoint should be audited
        should_audit = any(
            re.match(pattern, path)
            for pattern in self.audited_patterns
        )

        if should_audit:
            # Log data access event
            user = getattr(request.state, "user", None)
            event_type = self._get_event_type(path)

            await self.audit_logger.log(
                event_type=event_type,
                user_id=user.id if user else None,
                details={"path": path, "method": request.method},
                request=request
            )

        await self.app(scope, receive, send)
```

---

## Session Management

### Session Model

```python
# Proposed: stanley/auth/sessions.py

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class Session(BaseModel):
    """User session."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID

    # Token tracking
    refresh_token_hash: str

    # Device info
    device_type: Optional[str] = None
    device_name: Optional[str] = None
    browser: Optional[str] = None
    os: Optional[str] = None

    # Location
    ip_address: str
    country: Optional[str] = None
    city: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime

    # Status
    is_active: bool = True
    revoked_at: Optional[datetime] = None
    revoke_reason: Optional[str] = None

class SessionManager:
    """Manage user sessions."""

    def __init__(
        self,
        session_repo: SessionRepository,
        max_sessions_per_user: int = 5
    ):
        self.session_repo = session_repo
        self.max_sessions = max_sessions_per_user

    async def create_session(
        self,
        user_id: UUID,
        refresh_token: str,
        request: Request
    ) -> Session:
        """Create new session, enforcing session limits."""
        # Check existing sessions
        sessions = await self.session_repo.get_active_sessions(user_id)

        if len(sessions) >= self.max_sessions:
            # Revoke oldest session
            oldest = min(sessions, key=lambda s: s.created_at)
            await self.revoke_session(
                oldest.id,
                reason="New session created, maximum sessions exceeded"
            )

        # Parse device info from request
        user_agent = parse_user_agent(
            request.headers.get("user-agent", "")
        )

        session = Session(
            user_id=user_id,
            refresh_token_hash=hash_token(refresh_token),
            device_type=user_agent.device.family,
            browser=user_agent.browser.family,
            os=user_agent.os.family,
            ip_address=get_client_ip(request),
            expires_at=datetime.utcnow() + timedelta(days=30)
        )

        return await self.session_repo.create(session)

    async def revoke_session(
        self,
        session_id: UUID,
        reason: Optional[str] = None
    ):
        """Revoke a session."""
        await self.session_repo.update(
            session_id,
            is_active=False,
            revoked_at=datetime.utcnow(),
            revoke_reason=reason
        )

    async def revoke_all_sessions(
        self,
        user_id: UUID,
        except_session_id: Optional[UUID] = None
    ):
        """Revoke all user sessions (logout everywhere)."""
        sessions = await self.session_repo.get_active_sessions(user_id)

        for session in sessions:
            if session.id != except_session_id:
                await self.revoke_session(
                    session.id,
                    reason="User logged out from all devices"
                )
```

---

## API Key Management

### API Key Model

```python
# Proposed: stanley/auth/api_keys.py

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
import secrets
import hashlib

class APIKeyScope(str, Enum):
    """API key access scopes."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class APIKey(BaseModel):
    """User API key for programmatic access."""

    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    team_id: Optional[UUID] = None

    # Key details
    name: str
    key_prefix: str  # First 8 chars for identification
    key_hash: str    # SHA-256 hash of full key

    # Permissions
    scopes: list[APIKeyScope] = Field(default_factory=list)
    allowed_ips: Optional[list[str]] = None

    # Limits
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    # Status
    is_active: bool = True
    revoked_at: Optional[datetime] = None

class APIKeyService:
    """API key management service."""

    KEY_LENGTH = 32
    PREFIX_LENGTH = 8

    async def create_key(
        self,
        user_id: UUID,
        name: str,
        scopes: list[APIKeyScope],
        expires_in_days: Optional[int] = None,
        allowed_ips: Optional[list[str]] = None
    ) -> tuple[APIKey, str]:
        """Create new API key, returns key object and raw key."""
        # Generate secure random key
        raw_key = secrets.token_urlsafe(self.KEY_LENGTH)
        key_prefix = raw_key[:self.PREFIX_LENGTH]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey(
            user_id=user_id,
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            scopes=scopes,
            allowed_ips=allowed_ips,
            expires_at=(
                datetime.utcnow() + timedelta(days=expires_in_days)
                if expires_in_days else None
            )
        )

        await self.key_repo.create(api_key)

        # Return full key - this is the only time it's available
        return api_key, f"sk_{raw_key}"

    async def validate_key(
        self,
        raw_key: str,
        required_scope: Optional[APIKeyScope] = None
    ) -> APIKey:
        """Validate API key and return key object."""
        if not raw_key.startswith("sk_"):
            raise InvalidAPIKeyError("Invalid key format")

        key_value = raw_key[3:]
        key_hash = hashlib.sha256(key_value.encode()).hexdigest()

        api_key = await self.key_repo.get_by_hash(key_hash)

        if not api_key:
            raise InvalidAPIKeyError("Key not found")

        if not api_key.is_active:
            raise InvalidAPIKeyError("Key has been revoked")

        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            raise InvalidAPIKeyError("Key has expired")

        if required_scope and required_scope not in api_key.scopes:
            raise InsufficientScopeError(
                f"Key does not have {required_scope.value} scope"
            )

        # Update last used
        await self.key_repo.update_last_used(api_key.id)

        return api_key
```

---

## Rate Limiting

### Rate Limiter Implementation

```python
# Proposed: stanley/ratelimit/limiter.py

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import UUID
import redis.asyncio as redis

class RateLimitTier(Enum):
    """Rate limit tiers by subscription."""

    FREE = "free"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"
    ENTERPRISE = "enterprise"

@dataclass
class RateLimitConfig:
    """Rate limit configuration per tier."""

    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    concurrent_requests: int

    # Endpoint-specific limits
    data_requests_per_minute: int  # Market data, institutional data
    compute_requests_per_hour: int  # Backtests, screeners

TIER_LIMITS: dict[RateLimitTier, RateLimitConfig] = {
    RateLimitTier.FREE: RateLimitConfig(
        requests_per_minute=20,
        requests_per_hour=200,
        requests_per_day=1000,
        concurrent_requests=2,
        data_requests_per_minute=10,
        compute_requests_per_hour=5
    ),
    RateLimitTier.PROFESSIONAL: RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
        concurrent_requests=5,
        data_requests_per_minute=30,
        compute_requests_per_hour=50
    ),
    RateLimitTier.INSTITUTIONAL: RateLimitConfig(
        requests_per_minute=200,
        requests_per_hour=5000,
        requests_per_day=50000,
        concurrent_requests=10,
        data_requests_per_minute=100,
        compute_requests_per_hour=200
    ),
    RateLimitTier.ENTERPRISE: RateLimitConfig(
        requests_per_minute=1000,
        requests_per_hour=20000,
        requests_per_day=200000,
        concurrent_requests=50,
        data_requests_per_minute=500,
        compute_requests_per_hour=1000
    ),
}

class SlidingWindowRateLimiter:
    """Redis-based sliding window rate limiter."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Returns:
            (allowed, remaining, reset_in_seconds)
        """
        now = time.time()
        window_start = now - window_seconds
        key = f"ratelimit:{identifier}"

        pipe = self.redis.pipeline()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Count requests in window
        pipe.zcount(key, window_start, now)
        # Set expiry
        pipe.expire(key, window_seconds)

        _, _, count, _ = await pipe.execute()

        allowed = count <= limit
        remaining = max(0, limit - count)
        reset_in = int(window_seconds - (now - window_start))

        return allowed, remaining, reset_in

class UserRateLimiter:
    """Per-user rate limiting."""

    def __init__(
        self,
        limiter: SlidingWindowRateLimiter,
        user_service: UserService
    ):
        self.limiter = limiter
        self.user_service = user_service

    async def check_rate_limit(
        self,
        user_id: UUID,
        endpoint_type: str = "general"
    ) -> RateLimitResult:
        """Check rate limit for user."""
        user = await self.user_service.get_by_id(user_id)
        tier = RateLimitTier(user.subscription_tier.value)
        config = TIER_LIMITS[tier]

        # Check minute limit
        minute_key = f"user:{user_id}:{endpoint_type}:minute"
        allowed, remaining, reset = await self.limiter.check_limit(
            minute_key,
            config.requests_per_minute,
            60
        )

        if not allowed:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_in=reset,
                limit=config.requests_per_minute,
                window="minute"
            )

        # Check hour limit
        hour_key = f"user:{user_id}:{endpoint_type}:hour"
        allowed, remaining, reset = await self.limiter.check_limit(
            hour_key,
            config.requests_per_hour,
            3600
        )

        if not allowed:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_in=reset,
                limit=config.requests_per_hour,
                window="hour"
            )

        return RateLimitResult(
            allowed=True,
            remaining=remaining,
            reset_in=reset,
            limit=config.requests_per_hour,
            window="hour"
        )
```

### Rate Limit Middleware

```python
# Proposed: stanley/ratelimit/middleware.py

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limiting to API requests."""

    ENDPOINT_TYPES = {
        "/api/market/": "data",
        "/api/institutional/": "data",
        "/api/dark-pool/": "data",
        "/api/options-flow/": "data",
        "/api/backtest/": "compute",
        "/api/screen/": "compute",
    }

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path == "/api/health":
            return await call_next(request)

        user = getattr(request.state, "user", None)

        if user:
            # Determine endpoint type
            endpoint_type = "general"
            for prefix, type_ in self.ENDPOINT_TYPES.items():
                if request.url.path.startswith(prefix):
                    endpoint_type = type_
                    break

            # Check rate limit
            result = await request.app.state.rate_limiter.check_rate_limit(
                user.id,
                endpoint_type
            )

            if not result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry in {result.reset_in}s",
                    headers={
                        "X-RateLimit-Limit": str(result.limit),
                        "X-RateLimit-Remaining": str(result.remaining),
                        "X-RateLimit-Reset": str(result.reset_in),
                        "Retry-After": str(result.reset_in)
                    }
                )

            # Add rate limit headers to response
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
            response.headers["X-RateLimit-Reset"] = str(result.reset_in)

            return response

        return await call_next(request)
```

---

## Database Schema

### PostgreSQL Schema

```sql
-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255),

    -- OAuth
    oauth_provider VARCHAR(50),
    oauth_id VARCHAR(255),

    -- MFA
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),

    -- Profile
    avatar_url TEXT,
    bio TEXT,
    company VARCHAR(255),
    job_title VARCHAR(255),

    -- Status
    status VARCHAR(20) DEFAULT 'pending',
    email_verified BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMP,

    -- Subscription
    subscription_tier VARCHAR(20) DEFAULT 'free',
    subscription_expires_at TIMESTAMP,

    -- Preferences (JSONB for flexibility)
    preferences JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login_at TIMESTAMP,

    CONSTRAINT unique_oauth UNIQUE (oauth_provider, oauth_id)
);

-- Teams
CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,

    -- Settings
    settings JSONB DEFAULT '{}',

    -- Subscription
    subscription_tier VARCHAR(20) DEFAULT 'professional',
    max_members INTEGER DEFAULT 5,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(id)
);

-- Team Memberships
CREATE TABLE team_memberships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    role VARCHAR(20) DEFAULT 'member',

    joined_at TIMESTAMP DEFAULT NOW(),
    invited_by UUID REFERENCES users(id),

    CONSTRAINT unique_membership UNIQUE (user_id, team_id)
);

-- Sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    refresh_token_hash VARCHAR(255) NOT NULL,

    -- Device info
    device_type VARCHAR(50),
    device_name VARCHAR(100),
    browser VARCHAR(100),
    os VARCHAR(100),

    -- Location
    ip_address INET,
    country VARCHAR(2),
    city VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    last_active_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMP,
    revoke_reason VARCHAR(255)
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    name VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(8) NOT NULL,
    key_hash VARCHAR(64) NOT NULL UNIQUE,

    scopes TEXT[] DEFAULT '{}',
    allowed_ips INET[],

    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_day INTEGER DEFAULT 10000,

    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,

    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMP
);

-- Portfolios
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    name VARCHAR(255) NOT NULL,
    description TEXT,
    currency VARCHAR(3) DEFAULT 'USD',
    benchmark VARCHAR(10) DEFAULT 'SPY',

    -- Visibility
    visibility VARCHAR(20) DEFAULT 'private',
    shared_with UUID[] DEFAULT '{}',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;

-- Portfolio Holdings
CREATE TABLE portfolio_holdings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,

    symbol VARCHAR(20) NOT NULL,
    shares DECIMAL(18, 8) NOT NULL,
    average_cost DECIMAL(18, 4),

    added_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_holding UNIQUE (portfolio_id, symbol)
);

-- Watchlists
CREATE TABLE watchlists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    name VARCHAR(255) NOT NULL,
    description TEXT,

    visibility VARCHAR(20) DEFAULT 'private',
    shared_with UUID[] DEFAULT '{}',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE watchlist_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    watchlist_id UUID REFERENCES watchlists(id) ON DELETE CASCADE,

    symbol VARCHAR(20) NOT NULL,
    notes TEXT,
    target_price DECIMAL(18, 4),
    alert_enabled BOOLEAN DEFAULT FALSE,

    added_at TIMESTAMP DEFAULT NOW(),
    added_by UUID REFERENCES users(id),

    CONSTRAINT unique_watchlist_item UNIQUE (watchlist_id, symbol)
);

-- Audit Log
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,

    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    team_id UUID REFERENCES teams(id) ON DELETE SET NULL,

    resource_type VARCHAR(50),
    resource_id UUID,

    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(36),

    details JSONB DEFAULT '{}',

    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_oauth ON users(oauth_provider, oauth_id);
CREATE INDEX idx_team_memberships_user ON team_memberships(user_id);
CREATE INDEX idx_team_memberships_team ON team_memberships(team_id);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_active ON sessions(user_id, is_active);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user ON api_keys(user_id);
CREATE INDEX idx_portfolios_user ON portfolios(user_id);
CREATE INDEX idx_watchlists_user ON watchlists(user_id);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_event ON audit_logs(event_type, timestamp);
```

---

## Implementation Phases

### Phase 1: Authentication Foundation (Weeks 1-3)

**Goals:**
- Implement local authentication with Argon2
- Add JWT token service
- Create auth middleware
- Set up user repository

**Deliverables:**
- `/auth/register` endpoint
- `/auth/login` endpoint
- `/auth/refresh` endpoint
- `/auth/logout` endpoint
- Protected endpoint pattern

**Risk Mitigation:**
- Start with local auth before OAuth complexity
- Comprehensive test coverage for auth flows

### Phase 2: User Profiles & Permissions (Weeks 4-5)

**Goals:**
- Implement user profile management
- Add role-based permissions
- Create permission middleware

**Deliverables:**
- User profile CRUD
- Role assignment
- Permission checking decorators
- User preferences system

### Phase 3: Data Isolation (Weeks 6-8)

**Goals:**
- Migrate from global to scoped state
- Implement user-specific vaults
- Add row-level security

**Deliverables:**
- UserScopedState implementation
- Per-user vault directories
- Database RLS policies
- Migration scripts

### Phase 4: OAuth Integration (Weeks 9-10)

**Goals:**
- Add Google OAuth
- Add GitHub OAuth
- Add Microsoft OAuth

**Deliverables:**
- OAuth provider registration
- OAuth callback handlers
- Account linking flow

### Phase 5: Team Features (Weeks 11-13)

**Goals:**
- Implement team creation
- Add team membership
- Enable resource sharing

**Deliverables:**
- Team CRUD operations
- Member invitation flow
- Shared watchlists
- Team portfolios

### Phase 6: API Keys & Rate Limiting (Weeks 14-15)

**Goals:**
- Implement API key management
- Add rate limiting
- Create usage dashboard

**Deliverables:**
- API key generation
- Key validation middleware
- Redis rate limiter
- Usage metrics

### Phase 7: Audit & Monitoring (Weeks 16-17)

**Goals:**
- Implement audit logging
- Add session management
- Create admin dashboard

**Deliverables:**
- Audit event logging
- Session listing/revocation
- Admin audit viewer
- Security alerts

### Phase 8: Testing & Hardening (Weeks 18-20)

**Goals:**
- Security audit
- Performance testing
- Documentation

**Deliverables:**
- Penetration test results
- Load test results
- API documentation
- Migration guide

---

## Architecture Decision Records

### ADR-001: JWT for Session Tokens

**Status:** Proposed

**Context:**
Need to choose between server-side sessions (Redis) and stateless JWT tokens.

**Decision:**
Use JWT tokens with short-lived access tokens (30 min) and long-lived refresh tokens (7 days).

**Rationale:**
- Stateless scaling across API instances
- Reduced Redis dependency for session lookups
- Standard approach for modern APIs
- Refresh token rotation for security

**Consequences:**
- Cannot instantly revoke access tokens (30 min max exposure)
- Larger request headers
- Need refresh token infrastructure

---

### ADR-002: Argon2 for Password Hashing

**Status:** Proposed

**Context:**
Need to select password hashing algorithm.

**Decision:**
Use Argon2id with recommended parameters (t=3, m=65536, p=4).

**Rationale:**
- Winner of Password Hashing Competition
- Memory-hard (resistant to GPU attacks)
- Better than bcrypt/scrypt for modern hardware
- Python argon2-cffi library is well-maintained

**Consequences:**
- Higher memory usage per hash operation
- Slightly slower than bcrypt (by design)

---

### ADR-003: PostgreSQL Row-Level Security

**Status:** Proposed

**Context:**
Need to ensure data isolation between users in shared tables.

**Decision:**
Use PostgreSQL RLS policies for multi-tenant data isolation.

**Rationale:**
- Defense in depth (app + database layer)
- Cannot bypass via SQL injection
- Transparent to application code
- Supports complex team sharing rules

**Consequences:**
- Must set user context on each connection
- More complex query planning
- PostgreSQL-specific (reduces portability)

---

### ADR-004: Redis for Rate Limiting

**Status:** Proposed

**Context:**
Need distributed rate limiting across API instances.

**Decision:**
Use Redis with sliding window algorithm.

**Rationale:**
- Already in requirements.txt
- Atomic operations for accuracy
- Sliding window smooths traffic spikes
- Low latency lookups

**Consequences:**
- Redis becomes critical dependency
- Need Redis cluster for HA

---

## Component Interaction Diagram

```
                                  +------------------+
                                  |   Load Balancer  |
                                  +--------+---------+
                                           |
              +----------------------------+----------------------------+
              |                            |                            |
    +---------v---------+        +---------v---------+        +---------v---------+
    |   API Instance 1  |        |   API Instance 2  |        |   API Instance N  |
    +---------+---------+        +---------+---------+        +---------+---------+
              |                            |                            |
              +----------------------------+----------------------------+
                                           |
              +----------------------------+----------------------------+
              |                            |                            |
    +---------v---------+        +---------v---------+        +---------v---------+
    |      Redis        |        |    PostgreSQL     |        |   Object Store    |
    | (Rate Limiting,   |        | (Users, Teams,    |        | (User Vaults,     |
    |  Session Cache)   |        |  Portfolios)      |        |  Attachments)     |
    +-------------------+        +-------------------+        +-------------------+
```

---

## Security Considerations

### Authentication Security

1. **Password Requirements**
   - Minimum 12 characters
   - Check against HaveIBeenPwned API
   - Block common passwords

2. **MFA Support**
   - TOTP (Google Authenticator)
   - WebAuthn/FIDO2 (future)

3. **Brute Force Protection**
   - Account lockout after 5 failed attempts
   - Progressive delays
   - IP-based rate limiting on login

### Token Security

1. **Access Tokens**
   - Short-lived (30 minutes)
   - Include minimal claims
   - Sign with HS256 (symmetric) or RS256 (asymmetric)

2. **Refresh Tokens**
   - Rotate on each use
   - Store hash in database
   - Detect reuse (token theft indicator)

### API Security

1. **API Keys**
   - SHA-256 hash storage
   - Prefix for identification without revealing key
   - Optional IP whitelisting
   - Scope-based permissions

2. **Rate Limiting**
   - Per-user limits
   - Per-endpoint limits
   - Sliding window algorithm

### Data Security

1. **Encryption**
   - TLS 1.3 for transit
   - AES-256 for sensitive data at rest
   - Encrypted backups

2. **Data Isolation**
   - Row-level security in PostgreSQL
   - Separate vault directories per user
   - Team boundary enforcement

---

## Conclusion

This architecture provides a comprehensive framework for evolving Stanley into a multi-user platform while maintaining security, scalability, and the existing single-user functionality. The phased implementation approach allows for incremental delivery and risk mitigation.

Key architectural principles:
1. **Security First**: Defense in depth with authentication, authorization, and data isolation
2. **Scalability**: Stateless design with horizontal scaling capability
3. **Flexibility**: Support for individual users, teams, and enterprise deployments
4. **Observability**: Comprehensive audit logging and metrics
5. **Backward Compatibility**: Existing functionality preserved during migration
