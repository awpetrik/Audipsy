import json
import os

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
except ImportError:
    genai = None

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
SCHEMA = {
    "type": "object",
    "properties": {
        "stems": {
            "type": "object",
            "properties": {
                name: {
                    "type": "object",
                    "properties": {
                        "detected_instruments": {"type": "array", "items": {"type": "string"}},
                        "likely_fx": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["detected_instruments", "likely_fx", "confidence"],
                }
                for name in STEMS
            },
            "required": ["vocals", "drums", "bass", "other"],
        },
        "full_report": {
            "type": "object",
            "properties": {
                "genre_guess": {"type": "string"},
                "production_style": {"type": "string"},
                "notable_techniques": {"type": "array", "items": {"type": "string"}},
                "similar_artists": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["genre_guess", "production_style", "notable_techniques", "similar_artists"],
        },
    },
    "required": ["stems", "full_report"],
}


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
    return cleaned


def _payload(filename: str, track_features: dict, stem_features: dict) -> dict:
    return {"filename": filename, "track_features": track_features, "stem_features": stem_features, "schema": SCHEMA}


def _fallback_report(track_features: dict, stem_features: dict) -> dict:
    bpm = track_features["bpm"]
    genre = "lo-fi hip-hop" if bpm < 95 else "trap" if bpm < 125 else "house"
    style = "dusty and relaxed" if bpm < 95 else "punchy and modern" if bpm < 125 else "club-focused"
    artists = {
        "lo-fi hip-hop": ["J Dilla", "Nujabes", "idealism"],
        "trap": ["Metro Boomin", "Future", "Travis Scott"],
        "house": ["Fred again..", "Kaytranada", "Disclosure"],
    }[genre]
    defaults = {
        "vocals": "stacked lead vocal",
        "drums": "snare, kick, hi-hat kit",
        "bass": "808 bass",
        "guitar": "processed electric guitar",
        "piano": "layered piano chords",
        "other": "pad synth and harmonic layers",
    }
    stems = {}
    for name, feats in stem_features.items():
        fx = ["compression", "EQ shaping"]
        if feats["spectral_centroid"] > 2500:
            fx.append("bright top-end enhancement")
        if feats["energy_level"] == "high":
            fx.append("aggressive transient control")
        stems[name] = {
            "detected_instruments": [defaults.get(name, name)],
            "likely_fx": fx,
            "confidence": "medium" if name == "other" else "high",
        }
    return {
        "stems": stems,
        "full_report": {
            "genre_guess": genre,
            "production_style": style,
            "notable_techniques": ["layered arrangement", "focused low-end", "stereo depth processing"],
            "similar_artists": artists,
        },
    }


def _try_claude(prompt: dict) -> dict | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or Anthropic is None:
        return None
    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=1200,
            system="You are a precise music production analyst. Return JSON only.",
            messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=True)}],
        )
        return json.loads(_strip_fences(response.content[0].text))
    except Exception:
        return None


# Use Gemini structured output so the response stays machine-parseable.
def _try_gemini(prompt: dict) -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or genai is None:
        return None
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=json.dumps(prompt, ensure_ascii=True),
            config={"response_mime_type": "application/json", "response_json_schema": SCHEMA, "temperature": 0.2},
        )
        return json.loads(_strip_fences(response.text))
    except Exception:
        return None


# Choose the configured AI provider, preferring Gemini by default.
def generate_report(filename: str, track_features: dict, stem_features: dict) -> dict:
    prompt = _payload(filename, track_features, stem_features)
    provider = os.getenv("AI_PROVIDER", "gemini").lower()
    if provider in {"gemini", "auto"}:
        report = _try_gemini(prompt)
        if report:
            return report
    if provider in {"claude", "auto"}:
        report = _try_claude(prompt)
        if report:
            return report
    return _fallback_report(track_features, stem_features)
