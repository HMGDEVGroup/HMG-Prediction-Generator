import base64
import json
import os
from typing import Any, Dict

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

app = FastAPI(title="HMG Prediction Generator Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradeGuidanceResult(BaseModel):
    report_time: str
    current_brti_price: float
    page_timestamp: str
    environment: str
    phase: str
    hourly_bias: str
    five_minute_bias: str
    bot_intent: str
    primary_strike: float
    safer_backup_strike: float
    confirmation_mode: str
    anticipation_mode: str
    trap_detection: str
    early_exit_detection: str
    entry_reason: str
    invalidation_level: float
    position_size_guidance: str
    confidence: float
    final_action: str

    trade_readiness_banner: str
    best_strike_to_trade: float
    strike_side: str
    strike_reason: str
    confidence_color: str

    narrative_analysis: str


SYSTEM_PROMPT = """
You are HMG Assistant's BRTI screenshot trade analysis engine.

You receive two screenshots:
1. CME CF Bitcoin Real Time Index / BRTI screenshot.
2. Strike ladder screenshot showing strike prices and CALL/PUT or YES/NO percentage values.

Your first and most important job is to answer this question:

WHAT ARE THE BOTS ENGINEERING?

Before making any trade recommendation, determine whether the visible price action and strike ladder imply that bots are engineering one of the following:

- Strike pin
- Strike rejection
- Strike breakout
- Strike breakdown
- Liquidity grab above a strike
- Liquidity grab below a strike
- Bull trap
- Bear trap
- Compression before expansion
- Max-pain style settlement behavior
- Late push toward a specific strike
- No clear engineered intent

This bot-engineering answer must drive every downstream decision:
- environment
- phase
- bot_intent
- primary_strike
- safer_backup_strike
- confirmation_mode
- anticipation_mode
- invalidation_level
- confidence
- final_action
- narrative_analysis

Do not recommend CALL or PUT unless bot-engineered intent is clearly identified and supported by both:
1. BRTI structure
2. Strike ladder percentages

Use this trading framework:

1. Environment classification:
- NO TRADE: unclear, choppy, stale, too close to strike, or no dominance.
- GRIND MODE: directional drift exists but not enough dominance.
- STRIKE MODE: price is clearly being engineered toward, away from, above, or below a strike.

2. Phase classification:
- Phase 1 Build Phase: bots are ranging, collecting liquidity, compressing, or pinning. Usually WAIT or PASS.
- Phase 2 Pre-Lock Decision Window: structure starts to commit. This is where ANTICIPATE or confirmed trade logic can appear.
- Phase 3 Final Window: no new entries unless already confirmed. Usually manage or avoid.

3. Confirmation Mode:
- Safer, later entry.
- CALL confirmation requires price acceptance above the active strike with bullish ladder support.
- PUT confirmation requires price rejection below the active strike with bearish ladder support.

4. Anticipation Mode:
- Earlier, higher skill.
- ANTICIPATE CALL is allowed only when compression below an upside strike begins curling upward and ladder supports CALL/YES.
- ANTICIPATE PUT is allowed only when compression above a downside strike begins curling downward and ladder supports PUT/NO.

5. Strike translation:
- Translate the BRTI price into actual tradable strike levels.
- Identify nearest strike, primary strike, and safer backup strike.
- If price is within approximately 50 points of a strike, treat it as a danger zone unless dominance is extremely clear.

6. Trap detection:
Detect likely fake breakouts or liquidity grabs.

Bull trap:
- Price pushes above strike, attracts CALL/YES buyers, then fails back below.
- Reduce confidence and avoid CALL.

Bear trap:
- Price pushes below strike, attracts PUT/NO buyers, then reclaims.
- Reduce confidence and avoid PUT.

7. Early exit detection:
Detect momentum stall.

For CALL:
- Higher highs stop.
- Price falls back below active strike.
- Ladder no longer supports upside.

For PUT:
- Lower lows stop.
- Price reclaims active strike.
- Ladder no longer supports downside.

8. Invalidation-first execution:
Every trade idea must include an invalidation level.
If invalidation is unclear, final_action must be WAIT or PASS.

9. Risk rules:
- If setup is not A+, recommend small size or PASS.
- Near strike equals caution.
- If screenshots are blurry, unreadable, contradictory, or missing key values, return PASS with low confidence.
- Guidance only. Do not place trades.

Final action rules:
- CALL only when bullish bot-engineered intent is clear.
- PUT only when bearish bot-engineered intent is clear.
- ANTICIPATE CALL only when early bullish structure is forming before confirmation.
- ANTICIPATE PUT only when early bearish structure is forming before confirmation.
- WAIT when structure is forming but not ready.
- PASS when setup is poor, unclear, stale, contradictory, or unsafe.

Narrative analysis style:
Create a trader-style readout similar to the user's preferred analysis style.
Use short sections, direct language, and practical trading interpretation.

The narrative_analysis field must include these sections:

1. 🧾 STRUCTURE ANALYSIS
2. 📊 CURRENT PRICE
3. 🔍 CONTRACT STRUCTURE
4. 🧠 WHAT THE BOTS ARE ENGINEERING
5. ⚖️ CURRENT MARKET STATE
6. 📉 PRICE BEHAVIOR CONFIRMS IT
7. 🎯 BOT INTENT
8. ⚡ DOMINANCE READ
9. 🚨 TRADE FRAMEWORK
10. ❗ INVALIDATION LEVELS
11. 🧾 ONE-LINE TRUTH
12. FINAL CALL

Narrative rules:
- Do not overstate certainty.
- If no A+ setup exists, clearly say WAIT or PASS.
- Explain whether bots are pinning, rejecting, compressing, trapping, or expanding.
- Explain the contract ladder using strong zones and rejection zones when visible.
- Explain what confirms the read.
- Explain what invalidates the read.
- Use decisive but disciplined language.
- Keep it useful for a trader looking at the next hourly contract.
- Do not use markdown code fences.
- Do not mention that you are an AI model.

Return JSON only with this exact shape:
{
  "report_time": "string",
  "current_brti_price": 0,
  "page_timestamp": "string",
  "environment": "NO TRADE",
  "phase": "Phase 1 Build Phase",
  "hourly_bias": "neutral",
  "five_minute_bias": "neutral",
  "bot_intent": "string",
  "primary_strike": 0,
  "safer_backup_strike": 0,
  "confirmation_mode": "string",
  "anticipation_mode": "string",
  "trap_detection": "string",
  "early_exit_detection": "string",
  "entry_reason": "string",
  "invalidation_level": 0,
  "position_size_guidance": "string",
  "confidence": 0,
  "final_action": "WAIT",
  "narrative_analysis": "string"
}

Allowed final_action values:
CALL, PUT, WAIT, PASS, ANTICIPATE CALL, ANTICIPATE PUT.

Important:
Return JSON only.
No markdown outside JSON.
No commentary outside JSON.
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

        return json.loads(cleaned[start : end + 1])


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "").replace("%", "").strip()

        return float(value)
    except Exception:
        return default


def as_string(value: Any, default: str = "") -> str:
    if value is None:
        return default

    return str(value)


def build_fallback_narrative(parsed: Dict[str, Any]) -> str:
    report_time = as_string(parsed.get("report_time", "Current"))
    current_price = as_float(parsed.get("current_brti_price", 0))
    primary_strike = as_float(parsed.get("primary_strike", 0))
    backup_strike = as_float(parsed.get("safer_backup_strike", 0))
    environment = as_string(parsed.get("environment", "NO TRADE"))
    phase = as_string(parsed.get("phase", "Phase 1 Build Phase"))
    bot_intent = as_string(parsed.get("bot_intent", "Bot intent is unclear."))
    confirmation_mode = as_string(parsed.get("confirmation_mode", "WAIT for confirmation."))
    anticipation_mode = as_string(parsed.get("anticipation_mode", "No valid anticipation."))
    trap_detection = as_string(parsed.get("trap_detection", "No trap read available."))
    early_exit = as_string(parsed.get("early_exit_detection", "No early exit read available."))
    entry_reason = as_string(parsed.get("entry_reason", "No A+ entry identified."))
    invalidation = as_float(parsed.get("invalidation_level", 0))
    final_action = as_string(parsed.get("final_action", "WAIT")).upper()
    confidence = as_float(parsed.get("confidence", 0))

    price_text = f"${current_price:,.2f}" if current_price > 0 else "Unavailable"
    primary_text = f"{primary_strike:,.0f}" if primary_strike > 0 else "Unavailable"
    backup_text = f"{backup_strike:,.0f}" if backup_strike > 0 else "Unavailable"
    invalidation_text = f"{invalidation:,.0f}" if invalidation > 0 else "Not clearly defined"

    return f"""🧾 {report_time} — STRUCTURE ANALYSIS

📊 CURRENT PRICE
• {price_text}
• Primary strike: {primary_text}
• Safer backup strike: {backup_text}

🔍 CONTRACT STRUCTURE
• Active strike area: {primary_text}
• Backup / safer reference: {backup_text}
• The ladder must confirm direction before forcing a trade.

🧠 WHAT THE BOTS ARE ENGINEERING
{bot_intent}

⚖️ CURRENT MARKET STATE
• Environment: {environment}
• Phase: {phase}
• This is not automatically tradable unless dominance is clear.

📉 PRICE BEHAVIOR CONFIRMS IT
• Confirmation mode: {confirmation_mode}
• Anticipation mode: {anticipation_mode}

🎯 BOT INTENT
{bot_intent}

⚡ DOMINANCE READ
• Confidence: {confidence:.0f}%
• If confidence is below A+ territory, size stays small or the trade is skipped.

🚨 TRADE FRAMEWORK
• Final action: {final_action}
• Entry reason: {entry_reason}

❗ INVALIDATION LEVELS
• Invalidation: {invalidation_text}
• If invalidation is unclear or hit, do not hold emotionally.

🧾 ONE-LINE TRUTH
The system is reading bot-engineered structure first, then deciding whether the contract is tradable.

FINAL CALL
{final_action}
"""


def add_phase3_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    final_action = as_string(parsed.get("final_action", "WAIT")).upper()
    confidence = as_float(parsed.get("confidence", 0))
    current_price = as_float(parsed.get("current_brti_price", 0))
    primary_strike = as_float(parsed.get("primary_strike", 0))
    backup_strike = as_float(parsed.get("safer_backup_strike", 0))

    active_strike = primary_strike if primary_strike > 0 else backup_strike
    distance_to_active_strike = abs(current_price - active_strike) if active_strike > 0 else 999999

    bot_intent = as_string(parsed.get("bot_intent", "")).lower()
    trap_detection = as_string(parsed.get("trap_detection", "")).lower()
    early_exit_detection = as_string(parsed.get("early_exit_detection", "")).lower()

    has_trap_warning = (
        "trap" in trap_detection
        and "no likely" not in trap_detection
        and "not detected" not in trap_detection
    )

    has_early_exit_warning = (
        "exit" in early_exit_detection
        and "no" not in early_exit_detection[:20]
    )

    bot_intent_unclear = (
        "unclear" in bot_intent
        or "no clear" in bot_intent
        or bot_intent.strip() == ""
    )

    if bot_intent_unclear and final_action in ["CALL", "PUT", "ANTICIPATE CALL", "ANTICIPATE PUT"]:
        final_action = "WAIT"
        confidence = min(confidence, 54)

    if has_trap_warning and final_action in ["CALL", "PUT", "ANTICIPATE CALL", "ANTICIPATE PUT"]:
        confidence = min(confidence, 59)

    if has_early_exit_warning and final_action in ["CALL", "PUT", "ANTICIPATE CALL", "ANTICIPATE PUT"]:
        confidence = min(confidence, 64)

    if confidence >= 80 and final_action in ["CALL", "PUT", "ANTICIPATE CALL", "ANTICIPATE PUT"]:
        banner = "🔥 TRADE READY"
    elif distance_to_active_strike <= 50 and final_action in ["WAIT", "PASS"]:
        banner = "⚠️ DANGER ZONE"
    elif confidence >= 55 and final_action in ["WAIT", "PASS"]:
        banner = "🧠 COMPRESSION"
    elif final_action == "PASS":
        banner = "🚫 PASS"
    else:
        banner = "⏳ WAIT"

    if "CALL" in final_action:
        strike_side = "CALL"
        best_strike = active_strike
    elif "PUT" in final_action:
        strike_side = "PUT"
        best_strike = active_strike
    else:
        strike_side = "WAIT"
        best_strike = active_strike

    if confidence >= 80:
        confidence_color = "green"
    elif confidence >= 55:
        confidence_color = "yellow"
    else:
        confidence_color = "red"

    parsed["final_action"] = final_action
    parsed["confidence"] = confidence
    parsed["trade_readiness_banner"] = banner
    parsed["best_strike_to_trade"] = best_strike
    parsed["strike_side"] = strike_side
    parsed["strike_reason"] = (
        f"Best strike derived from bot-engineering intent, final_action={final_action}, "
        f"confidence={confidence:.0f}, current_price={current_price}, "
        f"primary_strike={primary_strike}, backup_strike={backup_strike}, "
        f"distance_to_active_strike={distance_to_active_strike:.2f}, "
        f"trap_warning={has_trap_warning}, early_exit_warning={has_early_exit_warning}."
    )
    parsed["confidence_color"] = confidence_color

    narrative = as_string(parsed.get("narrative_analysis", "")).strip()
    if not narrative:
        narrative = build_fallback_narrative(parsed)

    parsed["narrative_analysis"] = narrative

    return parsed


@app.get("/")
async def root() -> dict:
    return {
        "status": "ok",
        "service": "HMG Prediction Generator",
        "mode": "bot_engineering_first_with_narrative_analysis",
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=TradeGuidanceResult)
async def analyze(
    brti_image: UploadFile = File(...),
    ladder_image: UploadFile = File(...),
):
    try:
        brti_bytes = await brti_image.read()
        ladder_bytes = await ladder_image.read()

        if not brti_bytes:
            raise HTTPException(status_code=400, detail="BRTI image is empty.")

        if not ladder_bytes:
            raise HTTPException(status_code=400, detail="Strike ladder image is empty.")

        brti_data_url = encode_image(brti_bytes, brti_image.content_type or "image/png")
        ladder_data_url = encode_image(ladder_bytes, ladder_image.content_type or "image/png")

        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": SYSTEM_PROMPT,
                        },
                        {
                            "type": "input_text",
                            "text": (
                                "Image 1 is the BRTI screenshot. "
                                "Image 2 is the strike ladder screenshot. "
                                "First answer internally: WHAT ARE THE BOTS ENGINEERING? "
                                "Then return strict JSON only, including narrative_analysis."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": brti_data_url,
                        },
                        {
                            "type": "input_image",
                            "image_url": ladder_data_url,
                        },
                    ],
                }
            ],
        )

        parsed = extract_json(response.output_text)
        parsed = add_phase3_fields(parsed)

        return TradeGuidanceResult(**parsed)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
