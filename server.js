import "dotenv/config";
import express from "express";
import multer from "multer";
import cors from "cors";


const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 15 * 1024 * 1024 } // 15MB
});

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("Missing OPENAI_API_KEY env var");
  process.exit(1);
}

app.get("/", (req, res) => {
  res.json({ ok: true, service: "hmg-ai-image-backend" });
});

// POST /edit-image (multipart: image + prompt)
app.post("/edit-image", upload.single("image"), async (req, res) => {
  try {
    const prompt = (req.body.prompt || "").trim();
    if (!prompt) return res.status(400).send("Missing prompt");
    if (!req.file?.buffer) return res.status(400).send("Missing image file");

    // OpenAI Images Edit: POST /v1/images/edits
    const form = new FormData();
    form.append("model", "gpt-image-1.5");
    form.append("prompt", prompt);
    form.append("image", new Blob([req.file.buffer], { type: req.file.mimetype || "image/jpeg" }), "photo.jpg");
    // Optional tuning:
    form.append("size", "1024x1024");
    form.append("quality", "auto");

    const r = await fetch("https://api.openai.com/v1/images/edits", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`
      },
      body: form
    });

    const json = await r.json();
    if (!r.ok) {
      console.error("OpenAI error:", json);
      return res.status(r.status).json(json);
    }

    // GPT image models return base64 by default
    const imageBase64 = json?.data?.[0]?.b64_json;
    if (!imageBase64) {
      return res.status(500).send("No b64_json returned from OpenAI");
    }

    res.json({
      image_base64: imageBase64,
      prompt
    });
  } catch (err) {
    console.error(err);
    res.status(500).send(err?.message || "Server error");
  }
});

// POST /generate-caption (json: { style, prompt })
app.post("/generate-caption", async (req, res) => {
  try {
    const style = (req.body.style || "").trim();
    const prompt = (req.body.prompt || "").trim();
    if (!style) return res.status(400).send("Missing style");
    if (!prompt) return res.status(400).send("Missing prompt");

    const system = `
You write short social media captions.
Return EXACTLY 3 options, each on its own line. No numbering.
Keep them short and punchy. Add emojis only if the style allows it.
`;
    const styleGuide = {
      "Professional": "Professional, clean, minimal emojis (0-1).",
      "Friendly": "Warm, upbeat, light emojis (1-2).",
      "Vacation/Relaxed": "Chill vacation vibe, island/travel energy, emojis ok.",
      "Funny": "Playful, witty, light humor, emojis ok.",
      "Creative": "More poetic/creative, vivid language, emojis optional."
    }[style] || "Neutral tone.";

    // OpenAI Responses API
    const payload = {
      model: "gpt-4.1-mini",
      input: [
        { role: "system", content: system.trim() },
        { role: "user", content: `Style: ${style}\nGuidelines: ${styleGuide}\nPrompt: ${prompt}` }
      ]
    };

    const r = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify(payload)
    });

    const json = await r.json();
    if (!r.ok) {
      console.error("OpenAI error:", json);
      return res.status(r.status).json(json);
    }

    // Extract text from Responses output
    const text =
      json?.output?.[0]?.content?.[0]?.text ||
      json?.output_text ||
      "";

    const lines = text
      .split("\n")
      .map(s => s.trim())
      .filter(Boolean)
      .slice(0, 3);

    // Fallback if model returns in one paragraph
    const captions = lines.length >= 1 ? lines : [text.trim()].filter(Boolean);

    res.json({ captions: captions.slice(0, 3) });
  } catch (err) {
    console.error(err);
    res.status(500).send(err?.message || "Server error");
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`Server listening on ${PORT}`));
