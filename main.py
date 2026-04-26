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
    raise RuntimeError("OPENAI_API_KEY is missing. Create a .env file or Render environment variable.")

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


SYSTEM_PROMPT = """
You are HMG Assistant's BRTI screenshot trade analysis engine.

You receive two screenshots:
1. CME CF Bitcoin Real Time Index / BRTI screenshot.
2. Strike ladder screenshot showing strike prices and CALL/PUT or YES/NO percentage values.

Analyze both images together and return strict JSON only.

Use this trading framework:
- Determine what the bots appear to be building toward first.
- Classify environment as NO TRADE, GRIND MODE, or STRIKE MODE.
- Classify phase as Phase 1 Build Phase, Phase 2 Pre-Lock Decision Window, or Phase 3 Final Window.
- Separate Confirmation Mode from Anticipation Mode.
- Detect trap behavior.
- Detect early exit / momentum stall behavior.
- Always define invalidation before any trade idea.
- If setup is not A+, recommend small size or PASS.
- If price is close to strike, be cautious.
- If evidence is unclear, use WAIT or PASS.
- Do not place trades. Guidance only.

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
  "final_action": "WAIT"
}

Allowed final_action values:
CALL, PUT, WAIT, PASS, ANTICIPATE CALL, ANTICIPATE PUT.
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
            value = value.replace("$", "").replace(",", "").strip()
        return float(value)
    except Exception:
        return default


def add_phase3_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    final_action = str(parsed.get("final_action", "WAIT")).upper()
    confidence = as_float(parsed.get("confidence", 0))
    current_price = as_float(parsed.get("current_brti_price", 0))
    primary_strike = as_float(parsed.get("primary_strike", 0))
    backup_strike = as_float(parsed.get("safer_backup_strike", 0))

    active_strike = primary_strike if primary_strike > 0 else backup_strike
    distance_to_primary = abs(current_price - active_strike) if active_strike > 0 else 999999

    if confidence >= 80 and final_action in ["CALL", "PUT", "ANTICIPATE CALL", "ANTICIPATE PUT"]:
        banner = "🔥 TRADE READY"
    elif distance_to_primary <= 50 and final_action in ["WAIT", "PASS"]:
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
        f"Best strike derived from final_action={final_action}, "
        f"confidence={confidence:.0f}, current_price={current_price}, "
        f"primary_strike={primary_strike}, backup_strike={backup_strike}, "
        f"distance_to_active_strike={distance_to_primary:.2f}."
    )
    parsed["confidence_color"] = confidence_color

    return parsed


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "HMG Prediction Generator"}


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
                        {"type": "input_text", "text": SYSTEM_PROMPT},
                        {
                            "type": "input_text",
                            "text": "Image 1 is the BRTI screenshot. Image 2 is the strike ladder screenshot. Return strict JSON only.",
                        },
                        {"type": "input_image", "image_url": brti_data_url},
                        {"type": "input_image", "image_url": ladder_data_url},
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
