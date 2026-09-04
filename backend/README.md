# TAJ QX OTC Market Data Bridge

This FastAPI service is the server-side bridge between the Android app and the
authenticated `A11ksa/API-Quotex` WebSocket client. It does not generate or
simulate market data.

## Required environment

Set `QUOTEX_SSID` in the deployment provider's environment settings. Keep it
server-side; do not put it in source code, logs, Android configuration, or Git.
`QUOTEX_IS_DEMO` defaults to `true` and may also be configured as an environment
variable.

## Candle endpoint

```text
GET /api/market-candles?asset=EURUSD_otc&timeframe=60&count=60
```

The endpoint returns validated real candles with these fields:
`timestamp`, `open`, `high`, `low`, and `close`.

When the SSID is missing or the authenticated WebSocket cannot provide valid
candles, the endpoint returns HTTP 503 with `success: false`, an explicit error,
`status: "NO LIVE DATA"`, and an empty `candles` array. No mock candles or
forced signals are produced.