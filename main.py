import base64
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Create a .env file or Render environment variable."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="HMG Settlement Reader Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradeGuidanceResult(BaseModel):
    reportTime: str
    currentBRTIPrice: float
    pageTimestamp: str

    botsBuildingTowards: str
    settlementAdvantage: str
    environment: str
    phase: str
    marketModeSubtype: str
    anchorControl: str
    biasStrength: str

    signalLight: str
    tradeType: str
    safeSide: str
    bestStrike: float
    confidence: float
    finalAction: str

    why: str
    entryReason: str
    invalidationLevelText: str
    whatBreaksThis: List[str]
    riskControlRule: str

    noTradeZone: str
    anticipationTrigger: str
    confirmationTrigger: str
    shortTrigger: str
    executionInvalidation: str
    oneLineTruth: str

    narrativeAnalysis: str


class ExtractionResult(BaseModel):
    reportTime: str
    currentBRTIPrice: float
    pageTimestamp: str

    extractedBrtiVisible: bool
    extractedLadderVisible: bool
    screenshotsMatchMoment: bool
    screenshotQuality: str

    extractedBattlegroundStrike: float
    extractedPrimaryFloor: float
    extractedPrimaryCeiling: float

    extractedNearestYesPrice: float
    extractedNearestNoPrice: float

    extractedLadderRows: List[Dict[str, float]]

    extractedLadderRead: str
    extractedBrtiRead: str
    extractedObservedBehavior: str
    extractedAnchorEvidence: str

    visibleStrikes: List[float]
    visibleYesPrices: List[float]
    visibleNoPrices: List[float]

    missingCriticalData: bool
    missingDataReason: str


EXTRACTION_SYSTEM_PROMPT = """
You are a screenshot extraction engine for HMG settlement analysis.

You will receive exactly two screenshots:
1. CME CF Bitcoin Real Time Index (BRTI)
2. Robinhood-style Bitcoin contract ladder from approximately the same moment

This step is extraction only.
Do NOT give trading advice yet.
Do NOT return narrative prose outside JSON.
Do NOT invent values.

Read only what is visibly present in the screenshots.

Your job:
1. Extract the visible BRTI price if readable.
2. Extract the visible screenshot time / page time if readable.
3. Determine whether both screenshots appear to be from the same moment.
4. Extract visible ladder rows in a literal structured way.
5. Identify the likely battleground strike, primary floor, and primary ceiling from what is visibly present.
6. State whether critical data is missing.

Very important:
- Prefer literal row extraction over summary prose.
- If a ladder row is visible, include it.
- If a number is not readable, do not invent it.
- Use 0 for unreadable numeric values.
- Use empty string for unreadable text fields when appropriate.

screenshotQuality must be one of:
- CLEAR
- USABLE
- POOR
- UNREADABLE

Return JSON only in exactly this shape:
{
  "reportTime": "string",
  "currentBRTIPrice": 0,
  "pageTimestamp": "string",
  "extractedBrtiVisible": true,
  "extractedLadderVisible": true,
  "screenshotsMatchMoment": true,
  "screenshotQuality": "USABLE",
  "extractedBattlegroundStrike": 0,
  "extractedPrimaryFloor": 0,
  "extractedPrimaryCeiling": 0,
  "extractedNearestYesPrice": 0,
  "extractedNearestNoPrice": 0,
  "extractedLadderRows": [
    {"strike": 0, "yes": 0, "no": 0}
  ],
  "extractedLadderRead": "string",
  "extractedBrtiRead": "string",
  "extractedObservedBehavior": "string",
  "extractedAnchorEvidence": "string",
  "visibleStrikes": [0],
  "visibleYesPrices": [0],
  "visibleNoPrices": [0],
  "missingCriticalData": false,
  "missingDataReason": "string"
}
"""


ANALYSIS_SYSTEM_PROMPT = """
You are HMG Assistant's screenshot-based prediction-market decision engine.

You will receive extracted market facts from:
1. A BRTI screenshot
2. A Robinhood-style contract ladder screenshot

Use those extracted facts to produce an HMG settlement read.

This is live guidance only.
Do not assume the user is already in any position.

Core method:
- Profit by containment
- Trade only edges, not the middle
- Identify pin, defend, reject, reclaim, fail reclaim, breakdown, breakout, compression, trap
- Distinguish descriptive analysis from actionable triggers

You must be decisive and trigger-based, not vague.

Required concepts:
- Environment
- Phase
- Market mode subtype
- Anchor control
- Bias strength
- Signal light
- Trade type
- Safe side
- Best strike
- Execution plan
- One-line truth

Phase definitions:
PHASE 1 — Build Phase
- Bots range
- Liquidity collected
- No commitment yet
- Usually do nothing

PHASE 2 — Pre-Lock Decision Window
- About 3 to 5 minutes before settlement
- Structure starts to shift
- One side gains control
- Traps can be set
- This is the trade decision window

PHASE 3 — Final Window
- No more entries
- Bots execute the outcome
- Price stabilizes or ramps
- Only confirm, do not force entries

Trade type options:
- Directional
- Distance / Extension
- No Trade

Signal options:
- GREEN LIGHT
- YELLOW LIGHT
- RED LIGHT

GREEN LIGHT only when all are true:
1. Price failed a move
2. Direction is clear
3. Strike has distance
4. Contract price is safe
5. The trade is on the safe side, not the exciting side

GREEN LIGHT rule:
Failed move + distance + 92% to 98% safe side = Green Light

RED LIGHT if any one is true:
1. Price is in the middle
2. Strike is too close
3. Bots are switching modes
4. User would be chasing payout
5. Setup feels rushed

RED LIGHT rule:
Close strike + unclear direction + temptation = Red Light

YELLOW LIGHT means wait:
- Direction looks right but not confirmed
- Price is compressing
- Contract is around 50c to 75c
- A spike just happened
- One more move is needed

YELLOW LIGHT rule:
Almost good is still not good

Important interpretation rules:
- Distinguish STRIKE MODE into subtypes, such as:
  - Tradeable Compression
  - Pin Compression
  - Dead Chop
  - Trap Compression
- Explicitly detect anchor control:
  - Floor Anchor Control
  - Ceiling Anchor Control
  - Balanced / No Anchor
- Explicitly express bias strength:
  - Strongly Bullish
  - Slightly Bullish
  - Neutral
  - Slightly Bearish
  - Strongly Bearish
- Provide trigger-based execution, not just descriptive commentary.
- Prefer boring, safe-side analysis over exciting payout-chasing setups.
- If extracted facts are weak or incomplete, be honest and reduce confidence, but still produce the full schema.
- If the extracted facts clearly indicate missing critical data, default to RED LIGHT and NO TRADE.

Output requirements:
- bestStrike should be 0 when there is no valid strike
- finalAction allowed values:
  - ENTER
  - WAIT
  - NO TRADE
- signalLight allowed values:
  - GREEN LIGHT
  - YELLOW LIGHT
  - RED LIGHT

Return JSON only in this exact shape:
{
  "reportTime": "string",
  "currentBRTIPrice": 0,
  "pageTimestamp": "string",
  "botsBuildingTowards": "string",
  "settlementAdvantage": "string",
  "environment": "string",
  "phase": "string",
  "marketModeSubtype": "string",
  "anchorControl": "string",
  "biasStrength": "string",
  "signalLight": "RED LIGHT",
  "tradeType": "No Trade",
  "safeSide": "string",
  "bestStrike": 0,
  "confidence": 0,
  "finalAction": "NO TRADE",
  "why": "string",
  "entryReason": "string",
  "invalidationLevelText": "string",
  "whatBreaksThis": ["string", "string"],
  "riskControlRule": "string",
  "noTradeZone": "string",
  "anticipationTrigger": "string",
  "confirmationTrigger": "string",
  "shortTrigger": "string",
  "executionInvalidation": "string",
  "oneLineTruth": "string",
  "narrativeAnalysis": "string"
}
"""


def encode_image(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(cleaned[start:end + 1])


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = (
                value.replace("$", "")
                .replace(",", "")
                .replace("%", "")
                .replace("c", "")
                .replace("¢", "")
                .strip()
            )
        return float(value)
    except Exception:
        return default


def as_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def as_float_list(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []
    cleaned: List[float] = []
    for item in value:
        val = as_float(item, default=0.0)
        if val != 0:
            cleaned.append(val)
    return cleaned


def normalize_ladder_rows(value: Any) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not isinstance(value, list):
        return rows

    for item in value:
        if not isinstance(item, dict):
            continue
        strike = as_float(item.get("strike", 0))
        yes = as_float(item.get("yes", 0))
        no = as_float(item.get("no", 0))
        if strike != 0:
            rows.append({
                "strike": strike,
                "yes": yes,
                "no": no
            })
    return rows


def normalize_extraction_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    screenshot_quality = as_string(parsed.get("screenshotQuality", "UNREADABLE")).upper()
    if screenshot_quality not in {"CLEAR", "USABLE", "POOR", "UNREADABLE"}:
        screenshot_quality = "UNREADABLE"

    report_time = as_string(parsed.get("reportTime", ""))
    page_timestamp = as_string(parsed.get("pageTimestamp", ""))
    current_brti_price = as_float(parsed.get("currentBRTIPrice", 0))

    ladder_rows = normalize_ladder_rows(parsed.get("extractedLadderRows", []))
    visible_strikes = as_float_list(parsed.get("visibleStrikes", []))
    visible_yes = as_float_list(parsed.get("visibleYesPrices", []))
    visible_no = as_float_list(parsed.get("visibleNoPrices", []))

    if not visible_strikes and ladder_rows:
        visible_strikes = [row["strike"] for row in ladder_rows if row["strike"] != 0]
    if not visible_yes and ladder_rows:
        visible_yes = [row["yes"] for row in ladder_rows if row["yes"] != 0]
    if not visible_no and ladder_rows:
        visible_no = [row["no"] for row in ladder_rows if row["no"] != 0]

    missing_critical = bool(parsed.get("missingCriticalData", False))

    if current_brti_price <= 0:
        missing_critical = True
    if len(visible_strikes) == 0 and len(ladder_rows) == 0:
        missing_critical = True

    return {
        "reportTime": report_time,
        "currentBRTIPrice": current_brti_price,
        "pageTimestamp": page_timestamp,
        "extractedBrtiVisible": bool(parsed.get("extractedBrtiVisible", False)),
        "extractedLadderVisible": bool(parsed.get("extractedLadderVisible", False)),
        "screenshotsMatchMoment": bool(parsed.get("screenshotsMatchMoment", False)),
        "screenshotQuality": screenshot_quality,
        "extractedBattlegroundStrike": as_float(parsed.get("extractedBattlegroundStrike", 0)),
        "extractedPrimaryFloor": as_float(parsed.get("extractedPrimaryFloor", 0)),
        "extractedPrimaryCeiling": as_float(parsed.get("extractedPrimaryCeiling", 0)),
        "extractedNearestYesPrice": as_float(parsed.get("extractedNearestYesPrice", 0)),
        "extractedNearestNoPrice": as_float(parsed.get("extractedNearestNoPrice", 0)),
        "extractedLadderRows": ladder_rows,
        "extractedLadderRead": as_string(parsed.get("extractedLadderRead", "")),
        "extractedBrtiRead": as_string(parsed.get("extractedBrtiRead", "")),
        "extractedObservedBehavior": as_string(parsed.get("extractedObservedBehavior", "")),
        "extractedAnchorEvidence": as_string(parsed.get("extractedAnchorEvidence", "")),
        "visibleStrikes": visible_strikes,
        "visibleYesPrices": visible_yes,
        "visibleNoPrices": visible_no,
        "missingCriticalData": missing_critical,
        "missingDataReason": as_string(parsed.get("missingDataReason", "")),
    }


def build_fallback_narrative(parsed: Dict[str, Any]) -> str:
    report_time = as_string(parsed.get("reportTime", "Current"))
    current_price = as_float(parsed.get("currentBRTIPrice", 0))
    bots = as_string(parsed.get("botsBuildingTowards", "No clear engineered intent."))
    advantage = as_string(parsed.get("settlementAdvantage", "Settlement advantage is unclear."))
    environment = as_string(parsed.get("environment", "No Trade"))
    phase = as_string(parsed.get("phase", "Phase 1 — Build Phase"))
    subtype = as_string(parsed.get("marketModeSubtype", "Undefined"))
    anchor = as_string(parsed.get("anchorControl", "Balanced / No Anchor"))
    bias = as_string(parsed.get("biasStrength", "Neutral"))
    signal = as_string(parsed.get("signalLight", "RED LIGHT"))
    trade_type = as_string(parsed.get("tradeType", "No Trade"))
    safe_side = as_string(parsed.get("safeSide", "No safe side identified."))
    best_strike = as_float(parsed.get("bestStrike", 0))
    why = as_string(parsed.get("why", "The setup is not clear enough."))
    entry_reason = as_string(parsed.get("entryReason", "No A+ entry reason identified."))
    invalidation = as_string(parsed.get("invalidationLevelText", "Invalidation is unclear."))
    final_action = as_string(parsed.get("finalAction", "NO TRADE"))
    risk_rule = as_string(
        parsed.get(
            "riskControlRule",
            "No trade unless dominance is clear, strike is controlled, and flip risk is low."
        )
    )
    no_trade_zone = as_string(parsed.get("noTradeZone", "No-trade zone not identified."))
    anticipation = as_string(parsed.get("anticipationTrigger", "No anticipation trigger identified."))
    confirmation = as_string(parsed.get("confirmationTrigger", "No confirmation trigger identified."))
    short_trigger = as_string(parsed.get("shortTrigger", "No short trigger identified."))
    one_line_truth = as_string(parsed.get("oneLineTruth", "Trade only edges, not the middle."))

    strike_text = f"{best_strike:,.0f}" if best_strike > 0 else "No strike"
    price_text = f"${current_price:,.2f}" if current_price > 0 else "Unavailable"

    return f"""STRUCTURE ANALYSIS
Report time: {report_time}

CURRENT PRICE
BRTI price: {price_text}

WHAT THE BOTS ARE BUILDING TOWARD
{bots}

SETTLEMENT ADVANTAGE
{advantage}

ENVIRONMENT
{environment}

PHASE
{phase}

MODE SUBTYPE
{subtype}

ANCHOR CONTROL
{anchor}

BIAS STRENGTH
{bias}

SIGNAL
{signal}

TRADE TYPE
{trade_type}

SAFE SIDE
{safe_side}

BEST STRIKE
{strike_text}

WHY
{why}

ENTRY REASON
{entry_reason}

NO TRADE ZONE
{no_trade_zone}

ANTICIPATION TRIGGER
{anticipation}

CONFIRMATION TRIGGER
{confirmation}

SHORT TRIGGER
{short_trigger}

INVALIDATION
{invalidation}

RISK CONTROL RULE
{risk_rule}

ONE-LINE TRUTH
{one_line_truth}

FINAL ACTION
{final_action}"""


def normalize_analysis_payload(parsed: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    signal_light = as_string(parsed.get("signalLight", "RED LIGHT")).upper()
    if signal_light not in {"GREEN LIGHT", "YELLOW LIGHT", "RED LIGHT"}:
        signal_light = "RED LIGHT"

    final_action = as_string(parsed.get("finalAction", "NO TRADE")).upper()
    if final_action not in {"ENTER", "WAIT", "NO TRADE"}:
        final_action = "NO TRADE"

    confidence = max(0.0, min(100.0, as_float(parsed.get("confidence", 0))))
    narrative = as_string(parsed.get("narrativeAnalysis", "")).strip()

    normalized = {
        "reportTime": as_string(parsed.get("reportTime", extracted.get("reportTime", ""))),
        "currentBRTIPrice": as_float(parsed.get("currentBRTIPrice", extracted.get("currentBRTIPrice", 0))),
        "pageTimestamp": as_string(parsed.get("pageTimestamp", extracted.get("pageTimestamp", ""))),
        "botsBuildingTowards": as_string(parsed.get("botsBuildingTowards", "No clear engineered intent.")),
        "settlementAdvantage": as_string(parsed.get("settlementAdvantage", "Settlement advantage is unclear.")),
        "environment": as_string(parsed.get("environment", "No Trade")),
        "phase": as_string(parsed.get("phase", "Phase 1 — Build Phase")),
        "marketModeSubtype": as_string(parsed.get("marketModeSubtype", "Undefined")),
        "anchorControl": as_string(parsed.get("anchorControl", "Balanced / No Anchor")),
        "biasStrength": as_string(parsed.get("biasStrength", "Neutral")),
        "signalLight": signal_light,
        "tradeType": as_string(parsed.get("tradeType", "No Trade")),
        "safeSide": as_string(parsed.get("safeSide", "No safe side identified.")),
        "bestStrike": as_float(parsed.get("bestStrike", extracted.get("extractedBattlegroundStrike", 0))),
        "confidence": confidence,
        "finalAction": final_action,
        "why": as_string(parsed.get("why", "No high-quality edge identified.")),
        "entryReason": as_string(parsed.get("entryReason", "Wait for a safer setup.")),
        "invalidationLevelText": as_string(parsed.get("invalidationLevelText", "Invalidation is not clearly defined.")),
        "whatBreaksThis": as_string_list(parsed.get("whatBreaksThis", [])),
        "riskControlRule": as_string(
            parsed.get(
                "riskControlRule",
                "No trade unless dominance is clear, strike is controlled, and flip risk is low."
            )
        ),
        "noTradeZone": as_string(parsed.get("noTradeZone", "No-trade zone not identified.")),
        "anticipationTrigger": as_string(parsed.get("anticipationTrigger", "No anticipation trigger identified.")),
        "confirmationTrigger": as_string(parsed.get("confirmationTrigger", "No confirmation trigger identified.")),
        "shortTrigger": as_string(parsed.get("shortTrigger", "No short trigger identified.")),
        "executionInvalidation": as_string(parsed.get("executionInvalidation", "Execution invalidation not defined.")),
        "oneLineTruth": as_string(parsed.get("oneLineTruth", "Trade only edges, not the middle.")),
        "narrativeAnalysis": narrative,
    }

    if not normalized["narrativeAnalysis"]:
        normalized["narrativeAnalysis"] = build_fallback_narrative(normalized)

    return normalized


def build_analysis_user_prompt(extracted: Dict[str, Any]) -> str:
    return f"""
Use these extracted screenshot facts and produce the final HMG trade analysis.

EXTRACTED FACTS:
- reportTime: {json.dumps(extracted.get("reportTime", ""))}
- currentBRTIPrice: {json.dumps(extracted.get("currentBRTIPrice", 0))}
- pageTimestamp: {json.dumps(extracted.get("pageTimestamp", ""))}
- extractedBrtiVisible: {json.dumps(extracted.get("extractedBrtiVisible", False))}
- extractedLadderVisible: {json.dumps(extracted.get("extractedLadderVisible", False))}
- screenshotsMatchMoment: {json.dumps(extracted.get("screenshotsMatchMoment", False))}
- screenshotQuality: {json.dumps(extracted.get("screenshotQuality", "UNREADABLE"))}
- extractedBattlegroundStrike: {json.dumps(extracted.get("extractedBattlegroundStrike", 0))}
- extractedPrimaryFloor: {json.dumps(extracted.get("extractedPrimaryFloor", 0))}
- extractedPrimaryCeiling: {json.dumps(extracted.get("extractedPrimaryCeiling", 0))}
- extractedNearestYesPrice: {json.dumps(extracted.get("extractedNearestYesPrice", 0))}
- extractedNearestNoPrice: {json.dumps(extracted.get("extractedNearestNoPrice", 0))}
- extractedLadderRows: {json.dumps(extracted.get("extractedLadderRows", []))}
- extractedLadderRead: {json.dumps(extracted.get("extractedLadderRead", ""))}
- extractedBrtiRead: {json.dumps(extracted.get("extractedBrtiRead", ""))}
- extractedObservedBehavior: {json.dumps(extracted.get("extractedObservedBehavior", ""))}
- extractedAnchorEvidence: {json.dumps(extracted.get("extractedAnchorEvidence", ""))}
- visibleStrikes: {json.dumps(extracted.get("visibleStrikes", []))}
- visibleYesPrices: {json.dumps(extracted.get("visibleYesPrices", []))}
- visibleNoPrices: {json.dumps(extracted.get("visibleNoPrices", []))}
- missingCriticalData: {json.dumps(extracted.get("missingCriticalData", False))}
- missingDataReason: {json.dumps(extracted.get("missingDataReason", ""))}

Important:
- Use the extracted facts directly.
- If the extracted facts support a floor / battleground / ceiling read, state it clearly.
- If there is anchor evidence, use it.
- If the ladder rows show strong price asymmetry, use that.
- If there is not enough evidence, lower confidence honestly.
- Prefer an actionable trigger-based read over a vague narrative.
- If extracted facts are weak, still return the full schema, but use RED LIGHT / NO TRADE when appropriate.
Return JSON only.
""".strip()


def run_responses_call(system_prompt: str, user_content: List[Dict[str, Any]]) -> str:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    return response.output_text


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "service": "HMG Settlement Reader Backend"}


@app.get("/debug-schema")
def debug_schema() -> Dict[str, Any]:
    return {
        "reportTime": "",
        "currentBRTIPrice": 0,
        "pageTimestamp": "",
        "botsBuildingTowards": "",
        "settlementAdvantage": "",
        "environment": "",
        "phase": "",
        "marketModeSubtype": "",
        "anchorControl": "",
        "biasStrength": "",
        "signalLight": "",
        "tradeType": "",
        "safeSide": "",
        "bestStrike": 0,
        "confidence": 0,
        "finalAction": "",
        "why": "",
        "entryReason": "",
        "invalidationLevelText": "",
        "whatBreaksThis": [],
        "riskControlRule": "",
        "noTradeZone": "",
        "anticipationTrigger": "",
        "confirmationTrigger": "",
        "shortTrigger": "",
        "executionInvalidation": "",
        "oneLineTruth": "",
        "narrativeAnalysis": ""
    }


@app.post("/debug-analyze")
async def debug_analyze(
    brti_image: UploadFile = File(...),
    ladder_image: UploadFile = File(...)
) -> Dict[str, Any]:
    try:
        brti_bytes = await brti_image.read()
        ladder_bytes = await ladder_image.read()

        if not brti_bytes or not ladder_bytes:
            raise HTTPException(status_code=400, detail="Both screenshots are required.")

        brti_data_url = encode_image(brti_bytes, brti_image.content_type or "image/png")
        ladder_data_url = encode_image(ladder_bytes, ladder_image.content_type or "image/png")

        extraction_raw = run_responses_call(
            EXTRACTION_SYSTEM_PROMPT,
            [
                {
                    "type": "input_text",
                    "text": "Extract the visible hard facts from both screenshots. Do not give trade advice. Return JSON only."
                },
                {
                    "type": "input_image",
                    "image_url": brti_data_url
                },
                {
                    "type": "input_image",
                    "image_url": ladder_data_url
                }
            ]
        )

        extraction_parsed = extract_json(extraction_raw)
        extraction_normalized = normalize_extraction_payload(extraction_parsed)

        analysis_raw = run_responses_call(
            ANALYSIS_SYSTEM_PROMPT,
            [
                {
                    "type": "input_text",
                    "text": build_analysis_user_prompt(extraction_normalized)
                }
            ]
        )

        analysis_parsed = extract_json(analysis_raw)
        analysis_normalized = normalize_analysis_payload(analysis_parsed, extraction_normalized)

        return {
            "extraction_raw": extraction_raw,
            "extraction_parsed": extraction_parsed,
            "extraction_normalized": extraction_normalized,
            "analysis_raw": analysis_raw,
            "analysis_parsed": analysis_parsed,
            "analysis_normalized": analysis_normalized
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze", response_model=TradeGuidanceResult)
async def analyze_trade(
    brti_image: UploadFile = File(...),
    ladder_image: UploadFile = File(...)
) -> TradeGuidanceResult:
    try:
        brti_bytes = await brti_image.read()
        ladder_bytes = await ladder_image.read()

        if not brti_bytes or not ladder_bytes:
            raise HTTPException(status_code=400, detail="Both screenshots are required.")

        brti_data_url = encode_image(brti_bytes, brti_image.content_type or "image/png")
        ladder_data_url = encode_image(ladder_bytes, ladder_image.content_type or "image/png")

        extraction_raw = run_responses_call(
            EXTRACTION_SYSTEM_PROMPT,
            [
                {
                    "type": "input_text",
                    "text": "Extract the visible hard facts from both screenshots. Do not give trade advice. Return JSON only."
                },
                {
                    "type": "input_image",
                    "image_url": brti_data_url
                },
                {
                    "type": "input_image",
                    "image_url": ladder_data_url
                }
            ]
        )

        extracted_parsed = extract_json(extraction_raw)
        extracted = normalize_extraction_payload(extracted_parsed)

        analysis_raw = run_responses_call(
            ANALYSIS_SYSTEM_PROMPT,
            [
                {
                    "type": "input_text",
                    "text": build_analysis_user_prompt(extracted)
                }
            ]
        )

        analysis_parsed = extract_json(analysis_raw)
        normalized = normalize_analysis_payload(analysis_parsed, extracted)

        return TradeGuidanceResult(**normalized)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
