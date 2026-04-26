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
    raise RuntimeError("OPENAI_API_KEY is missing. Create a .env file in the backend folder.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="BRTI Screenshot Trade Analyzer Backend")

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


def encode_image(image_bytes: bytes, content_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


SYSTEM_PROMPT = """
You are HMG Assistant's BRTI screenshot trade analysis engine.

You will receive two screenshots:
1. CME CF Bitcoin Real Time Index / BRTI screenshot.
2. Strike ladder screenshot showing strike prices and CALL/PUT percentage values.

Analyze both images together and return strict JSON only.

Trading framework:
- Determine what the bots appear to be building toward before recommending anything.
- Use strike translation and timing sync.
- Classify the environment:
  NO TRADE, GRIND MODE, STRIKE MODE.
- Classify the phase:
  Phase 1 Build Phase, Phase 2 Pre-Lock Decision Window, Phase 3 Final Window.
- Use Confirmation Mode and Anticipation Mode separately.
- Detect trap behavior.
- Detect early exit / momentum stall behavior.
- Always define invalidation before any trade idea.
- If setup is not A+, recommend small size or PASS.
- If price is close to strike, be cautious.
- If evidence is unclear, use WAIT or PASS.
- Do not place trades. Guidance only.

Decision discipline:
- A CALL recommendation requires bullish BRTI structure and strike ladder support.
- A PUT recommendation requires bearish BRTI structure and strike ladder support.
- ANTICIPATE CALL or ANTICIPATE PUT is only allowed when compression is visible near a strike and the ladder supports the same direction.
- WAIT means structure may be forming but confirmation is not ready.
- PASS means the setup is poor, stale, contradictory, or unsafe.
- If screenshots are blurry or unreadable, return PASS with low confidence.

Return exactly this JSON shape:
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

Important:
Return JSON only.
No markdown.
No explanation outside JSON.
"""


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

        brti_data_url = encode_image(
            brti_bytes,
            brti_image.content_type or "image/png"
        )

        ladder_data_url = encode_image(
            ladder_bytes,
            ladder_image.content_type or "image/png"
        )

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
                                "Image 1 is the BRTI feed screenshot. "
                                "Image 2 is the strike ladder screenshot. "
                                "Analyze both and return strict JSON only."
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

        output_text = response.output_text

        parsed = extract_json(output_text)

        return TradeGuidanceResult(**parsed)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
