from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
import math
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image

logger = logging.getLogger("taj_qx.market_data")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    from api_quotex import AsyncQuotexClient
except ImportError:
    AsyncQuotexClient = None


app = FastAPI(
    title="TAJ QX Overlay Bot Backend",
    version="2.0.0",
)


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class AnalyzeChartRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded chart image")
    timeframe: str = "1m"
    timing: str = "5s"
    pair: str = "UNKNOWN"


class SmartMoney(BaseModel):
    orderBlock: str = "Not detected"
    fairValueGap: str = "Not detected"
    marketStructureBreak: str = "Not confirmed"
    liquiditySweep: str = "Not confirmed"


class TechnicalFactors(BaseModel):
    trendDirection: str = "Unknown"
    supportZone: str = "Not detected"
    resistanceZone: str = "Not detected"
    currentZoneReaction: str = "Not detected"
    emaFastSlow: str = "Not available"
    candlestickPattern: str = "Not confirmed"
    volatility: str = "Unknown"
    smartMoney: SmartMoney = Field(default_factory=SmartMoney)
    buyConfluenceScore: int = 0
    sellConfluenceScore: int = 0
    confluenceChecks: list[dict[str, Any]] = Field(default_factory=list)


class AnalyzeChartResponse(BaseModel):
    id: str
    timestamp: int
    pair: str
    timeframe: str
    timing: str
    signal: str
    confidence: int
    summaryReason: str
    technicalFactors: TechnicalFactors


class MarketCandle(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float


class MarketCandlesResponse(BaseModel):
    success: bool
    asset: str
    timeframe: int
    status: str
    candles: list[MarketCandle] = Field(default_factory=list)
    message: str = ""
    error: str | None = None


# =========================================================
# CONFIG
# =========================================================

def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_model() -> str:
    return os.getenv(
        "OPENAI_VISION_MODEL",
        "gpt-4o"
    ).strip()


def get_max_image_mb() -> int:
    try:
        return max(
            1,
            int(os.getenv("MAX_IMAGE_MB", "8"))
        )
    except ValueError:
        return 8


def get_quotex_ssid() -> str:
    return os.getenv("QUOTEX_SSID", "").strip()


def get_quotex_is_demo() -> bool:
    return os.getenv("QUOTEX_IS_DEMO", "true").strip().lower() == "true"


def valid_candle(candle: Any) -> MarketCandle | None:
    try:
        timestamp = int(candle.timestamp.timestamp())
        values = [candle.open, candle.high, candle.low, candle.close]
        if timestamp <= 0 or not all(math.isfinite(float(value)) for value in values):
            return None
        open_price, high_price, low_price, close_price = map(float, values)
        if min(open_price, high_price, low_price, close_price) <= 0:
            return None
        if high_price < max(open_price, close_price) or low_price > min(open_price, close_price):
            return None
        return MarketCandle(
            timestamp=timestamp,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


async def fetch_otc_candles(asset: str, timeframe: int, count: int) -> list[MarketCandle]:
    ssid = get_quotex_ssid()
    if not ssid:
        logger.warning("OTC candle fetch skipped: QUOTEX_SSID configured=false")
        return []
    if AsyncQuotexClient is None:
        logger.error("OTC candle fetch skipped: api-quotex is unavailable")
        return []

    logger.info(
        "Quotex/WebSocket connection attempt asset=%s timeframe=%s",
        asset,
        timeframe,
    )
    client = AsyncQuotexClient(ssid=ssid, is_demo=get_quotex_is_demo())
    try:
        connected = await client.connect()
        if not connected:
            logger.warning("Quotex/WebSocket connection failed asset=%s timeframe=%s", asset, timeframe)
            return []
        logger.info("Quotex/WebSocket connection succeeded asset=%s timeframe=%s", asset, timeframe)
        raw_candles = await client.get_candles(asset, timeframe, count=count)
        logger.info("Quotex candle response received asset=%s timeframe=%s count=%s", asset, timeframe, len(raw_candles or []))
        validated = [
            candle
            for raw_candle in raw_candles
            if (candle := valid_candle(raw_candle)) is not None
        ]
        invalid_count = len(raw_candles or []) - len(validated)
        if invalid_count:
            logger.warning("Invalid candle response asset=%s invalid_count=%s", asset, invalid_count)
        if not validated:
            logger.warning("Empty/invalid candle response asset=%s timeframe=%s", asset, timeframe)
        return sorted(validated, key=lambda candle: candle.timestamp)[-count:]
    except Exception as exc:
        logger.error(
            "Quotex candle fetch failed asset=%s timeframe=%s error_type=%s",
            asset,
            timeframe,
            type(exc).__name__,
        )
        return []
    finally:
        await client.disconnect()


# =========================================================
# IMAGE DECODER
# =========================================================

def decode_chart_image(value: str) -> tuple[Image.Image, bytes]:
    try:
        if "," in value and value.startswith("data:"):
            value = value.split(",", 1)[1]

        raw = base64.b64decode(value, validate=True)

        max_bytes = get_max_image_mb() * 1024 * 1024

        if len(raw) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="Chart image is too large."
            )

        image = Image.open(io.BytesIO(raw))
        image.load()

        width, height = image.size

        if width < 200 or height < 100:
            raise HTTPException(
                status_code=400,
                detail="Chart image resolution is too small."
            )

        return image.convert("RGB"), raw

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chart image: {exc}"
        )


# =========================================================
# SAFE WAIT
# =========================================================

def wait_response(
    pair: str,
    timeframe: str,
    timing: str,
    reason: str,
) -> AnalyzeChartResponse:

    return AnalyzeChartResponse(
        id=str(uuid.uuid4()),
        timestamp=int(time.time() * 1000),
        pair=pair,
        timeframe=timeframe,
        timing=timing,
        signal="WAIT",
        confidence=0,
        summaryReason=reason,
        technicalFactors=TechnicalFactors(),
    )


# =========================================================
# STRICT VISION PROMPT
# =========================================================

VISION_SYSTEM_PROMPT = """
You are the TAJ QX Chart Vision Analysis Engine.

Your job is to analyze a trading chart screenshot and determine
whether there is enough visible evidence for the NEXT candle.

IMPORTANT RULES:

1. Never invent prices, indicators, candles, zones or patterns.
2. Only use information visibly supported by the screenshot.
3. If the chart is unclear, cropped, obstructed, too small, or lacks
   enough evidence, return WAIT.
4. Do not claim certainty.
5. Do not guarantee profit or win rate.
6. BUY means the visible evidence favors bullish continuation/reversal
   for the NEXT candle.
7. SELL means the visible evidence favors bearish continuation/reversal
   for the NEXT candle.
8. WAIT means evidence is insufficient or conflicting.
9. Confidence must represent evidence strength, not guaranteed accuracy.
10. Prefer WAIT when evidence is ambiguous.
11. Analyze the requested timeframe.
12. Do not assume an EMA, RSI, S/R zone, FVG, order block or liquidity
    sweep exists unless it is actually visible or strongly inferable
    from the chart.

Return ONLY valid JSON.

Required JSON:

{
  "signal": "BUY|SELL|WAIT",
  "confidence": 0,
  "summaryReason": "...",
  "technicalFactors": {
    "trendDirection": "...",
    "supportZone": "...",
    "resistanceZone": "...",
    "currentZoneReaction": "...",
    "emaFastSlow": "...",
    "candlestickPattern": "...",
    "volatility": "...",
    "smartMoney": {
      "orderBlock": "...",
      "fairValueGap": "...",
      "marketStructureBreak": "...",
      "liquiditySweep": "..."
    },
    "buyConfluenceScore": 0,
    "sellConfluenceScore": 0,
    "confluenceChecks": [
      {
        "name": "...",
        "passed": true,
        "direction": "BUY|SELL|WAIT",
        "impact": "...",
        "scoreContribution": 0
      }
    ]
  }
}

Confidence must be an integer from 0 to 100.

Confluence scores must be integers from 0 to 100.

Do not output markdown fences.
"""


# =========================================================
# JSON NORMALIZATION
# =========================================================

def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        return max(
            minimum,
            min(maximum, int(value))
        )
    except Exception:
        return minimum


def normalize_signal(value: Any) -> str:
    signal = str(value or "WAIT").upper().strip()

    if signal not in {"BUY", "SELL", "WAIT"}:
        return "WAIT"

    return signal


def normalize_factors(value: Any) -> TechnicalFactors:
    if not isinstance(value, dict):
        return TechnicalFactors()

    smart = value.get("smartMoney", {})

    if not isinstance(smart, dict):
        smart = {}

    checks = value.get("confluenceChecks", [])

    if not isinstance(checks, list):
        checks = []

    normalized_checks = []

    for item in checks[:20]:
        if not isinstance(item, dict):
            continue

        normalized_checks.append(
            {
                "name": str(item.get("name", "Unknown")),
                "passed": bool(item.get("passed", False)),
                "direction": normalize_signal(
                    item.get("direction", "WAIT")
                ),
                "impact": str(
                    item.get("impact", "")
                ),
                "scoreContribution": clamp_int(
                    item.get("scoreContribution", 0),
                    0,
                    100,
                ),
            }
        )

    return TechnicalFactors(
        trendDirection=str(
            value.get("trendDirection", "Unknown")
        ),
        supportZone=str(
            value.get("supportZone", "Not detected")
        ),
        resistanceZone=str(
            value.get("resistanceZone", "Not detected")
        ),
        currentZoneReaction=str(
            value.get(
                "currentZoneReaction",
                "Not detected"
            )
        ),
        emaFastSlow=str(
            value.get(
                "emaFastSlow",
                "Not available"
            )
        ),
        candlestickPattern=str(
            value.get(
                "candlestickPattern",
                "Not confirmed"
            )
        ),
        volatility=str(
            value.get("volatility", "Unknown")
        ),
        smartMoney=SmartMoney(
            orderBlock=str(
                smart.get(
                    "orderBlock",
                    "Not detected"
                )
            ),
            fairValueGap=str(
                smart.get(
                    "fairValueGap",
                    "Not detected"
                )
            ),
            marketStructureBreak=str(
                smart.get(
                    "marketStructureBreak",
                    "Not confirmed"
                )
            ),
            liquiditySweep=str(
                smart.get(
                    "liquiditySweep",
                    "Not confirmed"
                )
            ),
        ),
        buyConfluenceScore=clamp_int(
            value.get(
                "buyConfluenceScore",
                0
            ),
            0,
            100,
        ),
        sellConfluenceScore=clamp_int(
            value.get(
                "sellConfluenceScore",
                0
            ),
            0,
            100,
        ),
        confluenceChecks=normalized_checks,
    )


# =========================================================
# VISION ANALYSIS
# =========================================================

def analyze_with_vision(
    image_bytes: bytes,
    pair: str,
    timeframe: str,
    timing: str,
) -> AnalyzeChartResponse:

    api_key = get_api_key()

    if not api_key:
        return wait_response(
            pair,
            timeframe,
            timing,
            "WAIT: OPENAI_API_KEY is not configured."
        )

    try:
        client = OpenAI(api_key=api_key)

        image_b64 = base64.b64encode(
            image_bytes
        ).decode("utf-8")

        response = client.chat.completions.create(
            model=get_model(),
            temperature=0,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Pair: {pair}\n"
                                f"Target timeframe: {timeframe}\n"
                                f"Analysis timing: {timing}\n\n"
                                "Analyze this chart screenshot "
                                "for the NEXT candle."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    f"{image_b64}"
                                )
                            },
                        },
                    ],
                },
            ],
        )

        content = (
            response.choices[0]
            .message
            .content
        )

        if not content:
            return wait_response(
                pair,
                timeframe,
                timing,
                "WAIT: Vision engine returned no analysis."
            )

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            return wait_response(
                pair,
                timeframe,
                timing,
                "WAIT: Vision engine returned invalid JSON."
            )

        signal = normalize_signal(
            result.get("signal")
        )

        confidence = clamp_int(
            result.get("confidence", 0),
            0,
            100,
        )

        reason = str(
            result.get(
                "summaryReason",
                "No reliable reason returned."
            )
        ).strip()

        factors = normalize_factors(
            result.get("technicalFactors")
        )

        # Safety rule:
        # High confidence cannot override conflicting evidence.
        if signal == "BUY":
            if factors.buyConfluenceScore <= (
                factors.sellConfluenceScore + 5
            ):
                signal = "WAIT"
                confidence = 0
                reason = (
                    "WAIT: Bullish evidence is not sufficiently "
                    "strong compared with bearish evidence."
                )

        elif signal == "SELL":
            if factors.sellConfluenceScore <= (
                factors.buyConfluenceScore + 5
            ):
                signal = "WAIT"
                confidence = 0
                reason = (
                    "WAIT: Bearish evidence is not sufficiently "
                    "strong compared with bullish evidence."
                )

        if signal == "WAIT":
            confidence = min(confidence, 60)

        return AnalyzeChartResponse(
            id=str(uuid.uuid4()),
            timestamp=int(time.time() * 1000),
            pair=pair,
            timeframe=timeframe,
            timing=timing,
            signal=signal,
            confidence=confidence,
            summaryReason=reason,
            technicalFactors=factors,
        )

    except Exception as exc:
        return wait_response(
            pair,
            timeframe,
            timing,
            f"WAIT: Vision analysis failed safely: {type(exc).__name__}"
        )


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "name": "TAJ QX Overlay Bot Backend",
        "status": "online",
        "version": "2.0.0",
        "engine": "vision",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "taj-qx-overlay-bot-backend",
        "vision_provider": "openai",
        "vision_provider_configured": bool(
            get_api_key()
        ),
        "model": get_model(),
        "otc_data_source": "A11ksa/API-Quotex WebSocket",
        "otc_data_configured": bool(get_quotex_ssid()) and AsyncQuotexClient is not None,
    }


@app.post(
    "/api/analyze-chart",
    response_model=AnalyzeChartResponse,
)
def analyze_chart(
    request: AnalyzeChartRequest
):

    image, raw_bytes = decode_chart_image(
        request.image
    )

    # Force-load image dimensions so corrupted images fail
    # before reaching the vision provider.
    width, height = image.size

    if width < 200 or height < 100:
        return wait_response(
            request.pair,
            request.timeframe,
            request.timing,
            "WAIT: Chart image is too small."
        )

    return analyze_with_vision(
        raw_bytes,
        request.pair,
        request.timeframe,
        request.timing,
    )


@app.get("/api/market-candles", response_model=MarketCandlesResponse)
async def market_candles(asset: str = "EURUSD_otc", timeframe: int = 60, count: int = 60):
    logger.info(
        "Market candle request received asset=%s timeframe=%s count=%s QUOTEX_SSID_configured=%s",
        asset,
        timeframe,
        count,
        bool(get_quotex_ssid()),
    )
    if asset != "EURUSD_otc" or timeframe != 60:
        logger.warning("Unsupported OTC candle request asset=%s timeframe=%s", asset, timeframe)
        raise HTTPException(status_code=400, detail="Only EURUSD_otc at 60 seconds is currently supported.")
    count = max(2, min(count, 200))
    if not get_quotex_ssid():
        logger.error("Market candle request rejected: QUOTEX_SSID configured=false")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "QUOTEX_SSID_NOT_CONFIGURED",
                "status": "NO LIVE DATA",
                "asset": asset,
                "timeframe": timeframe,
                "candles": [],
            },
        )
    if AsyncQuotexClient is None:
        logger.error("Market candle request rejected: api-quotex unavailable")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "API_QUOTEX_NOT_AVAILABLE",
                "status": "NO LIVE DATA",
                "asset": asset,
                "timeframe": timeframe,
                "candles": [],
            },
        )

    candles = await fetch_otc_candles(asset, timeframe, count)
    if not candles:
        logger.warning("Market candle response unavailable asset=%s timeframe=%s", asset, timeframe)
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "LIVE_CANDLE_DATA_UNAVAILABLE",
                "status": "NO LIVE DATA",
                "asset": asset,
                "timeframe": timeframe,
                "candles": [],
            },
        )
    logger.info("Successful live candle response asset=%s timeframe=%s count=%s", asset, timeframe, len(candles))
    return MarketCandlesResponse(success=True, asset=asset, timeframe=timeframe, status="LIVE OTC DATA", candles=candles)
