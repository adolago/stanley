# Stanley Mobile Companion App - Development Roadmap

## Executive Summary

This document outlines the complete mobile development strategy for Stanley, an institutional investment analysis platform. The mobile companion app will provide on-the-go access to alerts, watchlist monitoring, quick market views, and deep integration with the existing Python backend.

---

## 1. Mobile App Scope

### 1.1 Core Purpose
The mobile app serves as a **companion** to the desktop application, not a replacement. Key use cases:

- **Real-time Alerts**: Push notifications for money flow changes, dark pool activity, institutional movements
- **Watchlist Monitoring**: Quick access to tracked symbols with key metrics
- **Quick Views**: Snapshot dashboards for market conditions
- **Research on-the-go**: Access to valuation and DCF analysis
- **Portfolio Monitoring**: VaR, beta, and performance tracking

### 1.2 Feature Priority Matrix

| Feature | Priority | Phase |
|---------|----------|-------|
| Push Notifications | Critical | 1 |
| Watchlist | Critical | 1 |
| Market Data Quick View | Critical | 1 |
| Money Flow Dashboard | High | 1 |
| Institutional Holdings | High | 2 |
| Dark Pool Activity | High | 2 |
| Options Flow | Medium | 2 |
| Research/Valuation | Medium | 3 |
| Portfolio Analytics | Medium | 3 |
| Widget Support | Medium | 3 |
| Apple Watch/Wear OS | Low | 4 |

---

## 2. Technology Stack Decision: React Native vs Flutter

### 2.1 Analysis

| Criteria | React Native | Flutter | Winner |
|----------|-------------|---------|--------|
| **Developer Experience** | Familiar JS/TS | Dart learning curve | React Native |
| **Code Sharing with Web** | High (same patterns) | Low | React Native |
| **Performance** | Good (native modules) | Excellent (compiled) | Flutter |
| **Hot Reload** | Fast | Very fast | Flutter |
| **Ecosystem** | Mature, large | Growing rapidly | React Native |
| **Financial Charting** | Victory Native, react-native-svg | fl_chart, charts_flutter | Tie |
| **Push Notifications** | React Native Firebase | Firebase Flutter | Tie |
| **Native Look & Feel** | Good with effort | Consistent but custom | Tie |

### 2.2 Recommendation: **React Native**

**Rationale:**
1. **JavaScript/TypeScript ecosystem** aligns with future web expansion
2. **React patterns** match mobile development paradigms
3. **Mature library ecosystem** for financial data visualization
4. **Easier native module integration** when needed
5. **Strong community support** for enterprise features
6. **Existing team can ramp up faster** with JS/TS skills

### 2.3 Key Dependencies

```json
{
  "dependencies": {
    "react-native": "^0.73.x",
    "@react-navigation/native": "^6.x",
    "@react-navigation/bottom-tabs": "^6.x",
    "@tanstack/react-query": "^5.x",
    "zustand": "^4.x",
    "react-native-reanimated": "^3.x",
    "react-native-gesture-handler": "^2.x",
    "victory-native": "^40.x",
    "react-native-svg": "^15.x",
    "@react-native-firebase/app": "^18.x",
    "@react-native-firebase/messaging": "^18.x",
    "@react-native-async-storage/async-storage": "^1.x",
    "react-native-mmkv": "^2.x",
    "react-native-keychain": "^8.x"
  }
}
```

---

## 3. Core Mobile Features

### 3.1 Phase 1: Foundation

#### 3.1.1 Authentication & Session Management
```
- QR code pairing with desktop app
- Secure token storage via Keychain/Keystore
- Biometric authentication (Face ID/Touch ID/Fingerprint)
- Session synchronization with desktop
```

#### 3.1.2 Watchlist Management
```
- Add/remove symbols
- Reorder watchlist
- Symbol search with autocomplete
- Quick price tiles with change indicators
- Pull-to-refresh data
```

#### 3.1.3 Market Data Quick View
```
- Symbol header with price, change, volume
- Mini charts (1D, 1W, 1M)
- Key statistics panel
- Swipe between watchlist symbols
```

#### 3.1.4 Money Flow Dashboard
```
- Sector money flow heatmap
- Equity flow score for selected symbol
- Smart money sentiment indicator
- Flow acceleration metrics
```

### 3.2 Phase 2: Analytics

#### 3.2.1 Institutional Holdings
```
- Top holders list with ownership %
- Position change indicators
- New/exited positions badges
- Whale alert integration
```

#### 3.2.2 Dark Pool Activity
```
- Dark pool volume vs total volume
- Large block trade indicators
- Signal direction (bullish/bearish/neutral)
- Historical dark pool chart
```

#### 3.2.3 Options Flow
```
- Call/Put volume summary
- Put/Call ratio gauge
- Net premium flow
- Unusual activity list
- Smart money trades
```

### 3.3 Phase 3: Research & Portfolio

#### 3.3.1 Research Access
```
- Valuation multiples card
- DCF summary with upside %
- Fair value range visualization
- Peer comparison table
```

#### 3.3.2 Portfolio Analytics
```
- Portfolio summary card
- VaR (95%, 99%) display
- Beta vs benchmark
- Sector exposure pie chart
- Top holdings list
```

---

## 4. Push Notifications Architecture

### 4.1 Alert Types

Based on `stanley/analytics/alerts.py`:

| Alert Type | Push Priority | Sound |
|------------|--------------|-------|
| `DARK_POOL_SURGE` | High | Alert |
| `BLOCK_TRADE` (Mega) | Critical | Urgent |
| `UNUSUAL_VOLUME` | High | Alert |
| `SECTOR_ROTATION` | Medium | Default |
| `SMART_MONEY_INFLOW` | High | Alert |
| `SMART_MONEY_OUTFLOW` | High | Alert |
| `FLOW_MOMENTUM_SHIFT` | Medium | Default |
| `INSTITUTIONAL_ACCUMULATION` | High | Alert |
| `INSTITUTIONAL_DISTRIBUTION` | High | Alert |

### 4.2 Backend Integration

New API endpoint required:

```python
# stanley/api/main.py - New endpoint

@app.post("/api/mobile/register-device", tags=["Mobile"])
async def register_device(request: DeviceRegistrationRequest):
    """
    Register mobile device for push notifications.

    Args:
        device_token: FCM/APNs token
        device_type: 'ios' or 'android'
        watchlist: List of symbols to track
        alert_preferences: Alert type preferences
    """
    pass

@app.get("/api/mobile/alerts/{symbol}", tags=["Mobile"])
async def get_mobile_alerts(symbol: str, limit: int = 20):
    """
    Get recent alerts for a symbol (mobile-optimized).
    """
    pass

@app.post("/api/mobile/alerts/subscribe", tags=["Mobile"])
async def subscribe_alerts(request: AlertSubscriptionRequest):
    """
    Subscribe to specific alert types for symbols.
    """
    pass
```

### 4.3 Push Notification Service

```python
# stanley/services/push_notification.py

from firebase_admin import messaging

class PushNotificationService:
    def send_money_flow_alert(
        self,
        device_tokens: List[str],
        alert: MoneyFlowAlert
    ):
        """Send money flow alert to registered devices."""
        pass

    def send_whale_alert(
        self,
        device_tokens: List[str],
        whale_movement: WhaleMovement
    ):
        """Send whale movement alert."""
        pass
```

### 4.4 Notification Categories (iOS)

```swift
// iOS notification categories for actions
UNNotificationCategory(
    identifier: "MONEY_FLOW_ALERT",
    actions: [
        UNNotificationAction(identifier: "VIEW_DETAILS", title: "View Details"),
        UNNotificationAction(identifier: "ADD_TO_WATCHLIST", title: "Add to Watchlist"),
        UNNotificationAction(identifier: "DISMISS", title: "Dismiss")
    ]
)
```

---

## 5. Offline Capability

### 5.1 Offline Data Strategy

| Data Type | Cache Duration | Storage |
|-----------|---------------|---------|
| Watchlist symbols | Indefinite | MMKV |
| Last known prices | 24 hours | MMKV |
| User preferences | Indefinite | MMKV |
| Research reports | 7 days | SQLite |
| Historical charts | 1 day | SQLite |
| Alert history | 7 days | SQLite |

### 5.2 Sync Strategy

```typescript
// Offline-first architecture
const syncManager = {
  // Queue actions when offline
  queueAction: (action: OfflineAction) => void,

  // Sync when connection restored
  syncPendingActions: async () => Promise<void>,

  // Conflict resolution
  resolveConflicts: (local: Data, remote: Data) => Data,
};

// React Query with offline support
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      cacheTime: 1000 * 60 * 60 * 24, // 24 hours
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
});
```

### 5.3 Background Sync

```typescript
// Background fetch for iOS/Android
import BackgroundFetch from 'react-native-background-fetch';

BackgroundFetch.configure({
  minimumFetchInterval: 15, // minutes
  stopOnTerminate: false,
  startOnBoot: true,
  enableHeadless: true,
}, async (taskId) => {
  // Fetch latest alerts
  await fetchLatestAlerts();
  // Update cached prices
  await refreshWatchlistPrices();

  BackgroundFetch.finish(taskId);
});
```

---

## 6. Desktop-Mobile Authentication

### 6.1 QR Code Pairing Flow

```
1. Desktop generates time-limited pairing code
2. Mobile scans QR code containing:
   - Backend URL
   - Pairing token (expires in 5 minutes)
   - Device fingerprint challenge
3. Mobile sends pairing request with device info
4. Backend validates and issues:
   - Access token (short-lived)
   - Refresh token (long-lived)
5. Mobile stores tokens in secure storage
6. Desktop shows "Device paired" confirmation
```

### 6.2 API Endpoints

```python
@app.post("/api/auth/generate-pairing-code", tags=["Auth"])
async def generate_pairing_code():
    """Generate QR code data for mobile pairing."""
    return {
        "pairing_token": generate_secure_token(),
        "expires_at": datetime.now() + timedelta(minutes=5),
        "backend_url": settings.API_URL,
    }

@app.post("/api/auth/pair-device", tags=["Auth"])
async def pair_device(request: DevicePairingRequest):
    """Complete mobile device pairing."""
    pass

@app.post("/api/auth/refresh-token", tags=["Auth"])
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token."""
    pass
```

### 6.3 Session Synchronization

```typescript
// Shared state between desktop and mobile
interface SharedSession {
  watchlist: string[];
  alertPreferences: AlertPreferences;
  lastSyncTimestamp: number;
}

// WebSocket for real-time sync
const syncSocket = new WebSocket(`${API_URL}/ws/sync`);
syncSocket.onmessage = (event) => {
  const update = JSON.parse(event.data);
  if (update.type === 'WATCHLIST_UPDATED') {
    updateLocalWatchlist(update.watchlist);
  }
};
```

---

## 7. Widget Support

### 7.1 iOS Widgets (WidgetKit)

#### Small Widget - Price Ticker
```
+-------------------+
|  AAPL    +1.25%  |
| $245.32           |
+-------------------+
```

#### Medium Widget - Watchlist
```
+-------------------------------+
| WATCHLIST                     |
| AAPL  $245.32  +1.25%        |
| MSFT  $412.15  -0.45%        |
| NVDA  $875.00  +2.34%        |
+-------------------------------+
```

#### Large Widget - Money Flow
```
+-------------------------------+
| MONEY FLOW DASHBOARD          |
|                               |
| XLK  ████████  +0.45         |
| XLF  ███       -0.12         |
| XLE  ██████    +0.28         |
| XLV  ████      +0.15         |
|                               |
| ALERTS: 3 new                 |
+-------------------------------+
```

### 7.2 Android Widgets

Similar widget configurations using Glance API or traditional RemoteViews.

### 7.3 Widget Data Refresh

```swift
// iOS WidgetKit timeline
struct Provider: TimelineProvider {
    func getTimeline(in context: Context, completion: @escaping (Timeline<Entry>) -> ()) {
        // Refresh every 15 minutes during market hours
        let refreshDate = Calendar.current.date(byAdding: .minute, value: 15, to: Date())!
        let timeline = Timeline(entries: [entry], policy: .after(refreshDate))
        completion(timeline)
    }
}
```

---

## 8. Wearable Support

### 8.1 Apple Watch Features

| Feature | Complication | Notification | App |
|---------|-------------|--------------|-----|
| Price ticker | Yes | - | - |
| Price alert | - | Yes | - |
| Watchlist | - | - | Yes |
| Money flow score | Yes | Yes | - |

#### Watch App Screens
1. **Watchlist** - List of symbols with prices
2. **Symbol Detail** - Price, change, mini chart
3. **Alerts** - Recent alert list

### 8.2 Wear OS Features

Similar feature set using Compose for Wear OS.

### 8.3 Complication Data

```swift
// Apple Watch complication
struct StanleyComplicationProvider: ComplicationProvider {
    func getCurrentTimelineEntry(for complication: CLKComplication) async -> CLKComplicationTimelineEntry? {
        // Show current price of primary watchlist symbol
        let price = await fetchPrimarySymbolPrice()
        return createEntry(price: price)
    }
}
```

---

## 9. Mobile-Specific UI Patterns

### 9.1 Navigation Structure

```
Tab Bar Navigation
├── Home (Dashboard)
│   ├── Market Overview
│   ├── Today's Alerts
│   └── Quick Actions
├── Watchlist
│   ├── Symbol List
│   ├── Symbol Detail (push)
│   │   ├── Overview
│   │   ├── Money Flow
│   │   ├── Institutional
│   │   ├── Options
│   │   └── Research
│   └── Add Symbol (modal)
├── Alerts
│   ├── Active Alerts
│   ├── Alert History
│   └── Alert Settings
├── Portfolio
│   ├── Summary
│   ├── Risk Metrics
│   └── Holdings
└── Settings
    ├── Notifications
    ├── Account
    └── Appearance
```

### 9.2 Gesture Support

| Gesture | Action |
|---------|--------|
| Pull-to-refresh | Reload current view data |
| Swipe left on watchlist item | Quick actions (delete, alert) |
| Swipe between symbols | Navigate watchlist |
| Long press on price | Show chart popup |
| Pinch on chart | Zoom chart |

### 9.3 Dark/Light Mode

```typescript
// Theme system matching system preference
const theme = {
  dark: {
    background: '#0a0a0a',
    cardBg: '#141414',
    text: '#ffffff',
    textSecondary: '#a1a1aa',
    accent: '#3b82f6',
    positive: '#22c55e',
    negative: '#ef4444',
  },
  light: {
    background: '#ffffff',
    cardBg: '#f4f4f5',
    text: '#18181b',
    textSecondary: '#71717a',
    accent: '#2563eb',
    positive: '#16a34a',
    negative: '#dc2626',
  },
};
```

### 9.4 Loading States

```typescript
// Skeleton loading for lists
const SymbolListSkeleton = () => (
  <View>
    {[1, 2, 3, 4, 5].map((i) => (
      <SkeletonPlaceholder key={i}>
        <View style={{ flexDirection: 'row', alignItems: 'center' }}>
          <View style={{ width: 48, height: 48, borderRadius: 8 }} />
          <View style={{ marginLeft: 12 }}>
            <View style={{ width: 80, height: 16, borderRadius: 4 }} />
            <View style={{ width: 120, height: 12, borderRadius: 4, marginTop: 6 }} />
          </View>
        </View>
      </SkeletonPlaceholder>
    ))}
  </View>
);
```

---

## 10. API Optimizations for Mobile

### 10.1 Mobile-Specific Endpoints

```python
# stanley/api/mobile.py

from fastapi import APIRouter

router = APIRouter(prefix="/api/mobile", tags=["Mobile"])

@router.get("/dashboard")
async def get_mobile_dashboard(watchlist: str = Query(...)):
    """
    Single endpoint for mobile dashboard data.
    Aggregates multiple data sources into one response.

    Returns:
        - Watchlist prices and changes
        - Top 3 alerts
        - Sector money flow summary
        - Portfolio summary (if configured)
    """
    pass

@router.get("/symbol/{symbol}/summary")
async def get_symbol_summary(symbol: str):
    """
    Optimized symbol summary for mobile.
    Single request replaces multiple desktop API calls.
    """
    return {
        "market_data": await get_market_data_optimized(symbol),
        "money_flow": await get_money_flow_summary(symbol),
        "institutional": await get_top_holders(symbol, limit=5),
        "signals": await get_active_signals(symbol),
    }
```

### 10.2 Response Compression

```python
from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### 10.3 Field Selection

```python
@router.get("/watchlist")
async def get_watchlist(
    symbols: str = Query(...),
    fields: str = Query(default="price,change,volume")
):
    """
    Fetch only requested fields to reduce payload.
    Mobile clients request minimal data.
    """
    pass
```

### 10.4 Pagination & Incremental Sync

```python
@router.get("/alerts")
async def get_alerts(
    since: datetime = Query(None),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0)
):
    """
    Incremental alert fetching.
    Mobile fetches only new alerts since last sync.
    """
    pass
```

### 10.5 WebSocket for Real-time Updates

```python
# stanley/api/websocket.py

@app.websocket("/ws/mobile/{device_id}")
async def mobile_websocket(websocket: WebSocket, device_id: str):
    """
    Real-time updates for mobile clients.

    Events:
        - price_update: Price changes for watchlist
        - alert: New alert triggered
        - sync: Desktop/mobile sync events
    """
    await websocket.accept()

    try:
        while True:
            # Send price updates every 5 seconds during market hours
            if is_market_open():
                prices = await get_watchlist_prices(device_id)
                await websocket.send_json({
                    "type": "price_update",
                    "data": prices
                })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
```

---

## 11. Development Phases

### Phase 1: Foundation (8 weeks)

**Week 1-2: Project Setup**
- Initialize React Native project
- Configure TypeScript, ESLint, Prettier
- Set up CI/CD with GitHub Actions
- Configure Firebase for push notifications

**Week 3-4: Core Infrastructure**
- Implement API client matching desktop patterns
- Set up state management (Zustand + React Query)
- Build navigation structure
- Implement secure storage

**Week 5-6: Authentication**
- QR code scanning
- Device pairing flow
- Token management
- Biometric authentication

**Week 7-8: Watchlist & Market Data**
- Watchlist management UI
- Symbol search
- Price ticker components
- Basic charting

### Phase 2: Analytics (6 weeks)

**Week 9-10: Money Flow**
- Money flow dashboard
- Sector flow visualization
- Equity flow detail

**Week 11-12: Institutional & Dark Pool**
- Institutional holdings list
- Dark pool activity view
- Whale alerts

**Week 13-14: Options Flow**
- Call/Put volume display
- Unusual activity list
- Options sentiment

### Phase 3: Advanced Features (6 weeks)

**Week 15-16: Push Notifications**
- FCM/APNs integration
- Alert subscription management
- Notification handling

**Week 17-18: Research & Portfolio**
- Valuation display
- DCF summary
- Portfolio analytics

**Week 19-20: Widgets**
- iOS WidgetKit implementation
- Android widget implementation

### Phase 4: Wearables & Polish (4 weeks)

**Week 21-22: Apple Watch**
- Watch app development
- Complications

**Week 23-24: Polish & Launch**
- Performance optimization
- Accessibility
- Beta testing
- App store submission

---

## 12. Directory Structure

```
stanley-mobile/
├── android/
├── ios/
├── src/
│   ├── api/
│   │   ├── client.ts
│   │   ├── hooks/
│   │   │   ├── useMarketData.ts
│   │   │   ├── useMoneyFlow.ts
│   │   │   ├── useInstitutional.ts
│   │   │   └── useAlerts.ts
│   │   └── types.ts
│   ├── components/
│   │   ├── charts/
│   │   │   ├── MiniChart.tsx
│   │   │   ├── SectorHeatmap.tsx
│   │   │   └── FlowBar.tsx
│   │   ├── common/
│   │   │   ├── PriceTicker.tsx
│   │   │   ├── ChangeIndicator.tsx
│   │   │   └── LoadingSkeleton.tsx
│   │   └── cards/
│   │       ├── MetricCard.tsx
│   │       ├── AlertCard.tsx
│   │       └── SymbolCard.tsx
│   ├── navigation/
│   │   ├── RootNavigator.tsx
│   │   ├── TabNavigator.tsx
│   │   └── types.ts
│   ├── screens/
│   │   ├── Dashboard/
│   │   ├── Watchlist/
│   │   ├── SymbolDetail/
│   │   ├── Alerts/
│   │   ├── Portfolio/
│   │   └── Settings/
│   ├── store/
│   │   ├── auth.ts
│   │   ├── watchlist.ts
│   │   └── preferences.ts
│   ├── services/
│   │   ├── notifications.ts
│   │   ├── storage.ts
│   │   └── sync.ts
│   ├── theme/
│   │   ├── colors.ts
│   │   ├── typography.ts
│   │   └── spacing.ts
│   └── utils/
│       ├── formatting.ts
│       └── validation.ts
├── App.tsx
├── package.json
├── tsconfig.json
└── README.md
```

---

## 13. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| App Store Rating | >= 4.5 | App Store/Play Store |
| Crash-free Rate | >= 99.5% | Firebase Crashlytics |
| API Response Time | < 200ms | Backend monitoring |
| Push Delivery Rate | >= 98% | FCM/APNs analytics |
| Daily Active Users | 60% of desktop users | Analytics |
| Widget Usage | 30% of mobile users | App analytics |

---

## 14. Security Considerations

1. **Data at Rest**: All sensitive data encrypted using platform keychain
2. **Data in Transit**: TLS 1.3 minimum, certificate pinning
3. **Authentication**: JWT with short expiry, refresh token rotation
4. **Biometrics**: LocalAuthentication framework (iOS), BiometricPrompt (Android)
5. **Code Obfuscation**: ProGuard (Android), App Thinning (iOS)
6. **Jailbreak/Root Detection**: Optional, configurable

---

## 15. Conclusion

The Stanley Mobile Companion app will extend the platform's reach to users on-the-go, focusing on real-time alerts, quick market views, and seamless integration with the desktop experience. React Native provides the optimal balance of development speed, performance, and ecosystem support for this financial application.

The phased approach ensures a solid foundation before adding advanced features, with clear milestones for each development phase. API optimizations specific to mobile ensure efficient data transfer and battery-conscious operations.

---

*Document Version: 1.0*
*Created: December 2024*
*Author: Stanley Development Team*
