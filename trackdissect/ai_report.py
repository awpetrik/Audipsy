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

STEMS = ("vocals", "instrumental", "drums", "bass", "guitar", "piano", "other")
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
                        "mixing_logic": {
                            "type": "object",
                            "properties": {
                                "eq": {"type": "string"},
                                "dynamics": {"type": "string"},
                                "space": {"type": "string"},
                                "saturation": {"type": "string"},
                                "modulation": {"type": "string"},
                            },
                            "required": ["eq", "dynamics", "space", "saturation", "modulation"],
                        },
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["detected_instruments", "likely_fx", "mixing_logic", "confidence"],
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
                "musical_characteristics": {
                    "type": "object",
                    "properties": {
                        "mood": {"type": "string"},
                        "harmonic_complexity": {"type": "string"},
                        "rhythmic_identity": {"type": "string"},
                    },
                    "required": ["mood", "harmonic_complexity", "rhythmic_identity"],
                },
                "arrangement_narrative": {
                    "type": "object",
                    "properties": {
                        "structure_guess": {"type": "string"},
                        "energy_flow": {"type": "string"},
                        "build_up_technique": {"type": "string"},
                    },
                    "required": ["structure_guess", "energy_flow", "build_up_technique"],
                },
                "sound_architecture": {
                    "type": "object",
                    "properties": {
                        "technical_summary": {"type": "string"},
                        "synthesis_tips": {"type": "string"},
                    },
                    "required": ["technical_summary", "synthesis_tips"],
                },
                "pro_workflow_tip": {"type": "string"},
                "notable_techniques": {"type": "array", "items": {"type": "string"}},
                "similar_artists": {"type": "array", "items": {"type": "string"}},
                "reference_tracks": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["genre_guess", "production_style", "musical_characteristics", "arrangement_narrative", "sound_architecture", "pro_workflow_tip", "notable_techniques", "similar_artists", "reference_tracks"],
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
    meta = track_features.get("metadata", {})
    
    # Strip redundant spectral data from stem features to reduce token count
    stripped_stems = {}
    for stem, features in stem_features.items():
        stripped_stems[stem] = {
            "bpm": features.get("bpm"),
            "key": features.get("key"),
            "energy": features.get("energy"),
            "spectral_centroid": features.get("spectral_centroid"),
            "spectral_flatness": features.get("spectral_flatness"),
            "crest_factor": features.get("crest_factor"),
            "stereo_correlation": features.get("stereo_correlation"),
            "sub_bass_ratio": features.get("sub_bass_ratio"),
        }

    context = {
        "filename": filename,
        "track_metadata": {
            "artist": meta.get("artist"),
            "title": meta.get("title"),
            "album": meta.get("album"),
            "original_genre_tag": meta.get("genre")
        },
        "audio_features": {
            "bpm": track_features.get("bpm"),
            "key": track_features.get("key"),
            "energy": track_features.get("energy"),
            "danceability": track_features.get("danceability"),
        },
        "stem_features": stripped_stems,
        "processing_info": {
            "anti_artifact_smoothing": "enabled (High-Shelf Filter @ 14kHz)",
            "peak_normalization": "-0.5dBFS"
        },
        "schema": SCHEMA
    }
    return context


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
        "instrumental": "full instrumental backing track",
        "drums": "snare, kick, hi-hat kit",
        "bass": "808 bass",
        "guitar": "processed electric guitar",
        "piano": "layered piano chords",
        "other": "pad synth and harmonic layers",
    }
    stems = {}
    for name, feats in stem_features.items():
        fx = ["compression", "EQ shaping"]
        if feats.get("zero_crossing_rate", 0) > 0.1:
            fx.append("bright transient enhancement")
        if feats["energy_level"] == "high":
            fx.append("aggressive transient control")
        stems[name] = {
            "detected_instruments": [defaults.get(name, name)],
            "likely_fx": fx,
            "mixing_logic": {
                "eq": "High-pass at 80Hz, broad +2dB bell at fundamental.",
                "dynamics": "4:1 ratio compression, 3-5dB gain reduction.",
                "space": "Parallel plate reverb (0.8s decay) for depth.",
                "saturation": "Subtle tape saturation for pleasant harmonics.",
                "modulation": "Slight chorus on the upper frequencies for width.",
            },
            "confidence": "medium" if name == "other" else "high",
        }
    return {
        "stems": stems,
        "full_report": {
            "genre_guess": genre,
            "production_style": style,
            "musical_characteristics": {
                "mood": "Dynamic and engaging",
                "harmonic_complexity": "Modern tonal balance",
                "rhythmic_identity": "Consistent and energetic"
            },
            "arrangement_narrative": {
                "structure_guess": "Standard contemporary structure",
                "energy_flow": "Well-balanced energy distribution",
                "build_up_technique": "Strategic layering and volume automation"
            },
            "sound_architecture": {
                "technical_summary": "Balanced frequency distribution with centered mono low-end.",
                "synthesis_tips": "Layer sub-oscillator with sawtooth waves for richer texture.",
            },
            "pro_workflow_tip": "Use Logic Pro's Drum Machine Designer and check Phase Correlation on the Stereo Out.",
            "notable_techniques": ["layered arrangement", "focused low-end", "stereo depth processing"],
            "similar_artists": artists,
            "reference_tracks": ["Metro Boomin - Space Cadet", "Disclosure - Latch", "J Dilla - Life"],
        },
    }


def _try_claude(prompt: dict) -> dict | None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or Anthropic is None:
        return None
    client = Anthropic(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
                max_tokens=2000,
                system=(
                    "You are a Senior Sound Designer & Logic Pro Specialist. "
                    "Analyze the provided audio features and track metadata (Artist/Title). "
                    "If the Artist/Title is known, provide context-specific production insights and reference specific songs ('reference_tracks') from the real world. "
                    "Provide professional, highly technical mixing advice. You MUST use specific dB values, Hz frequencies, and ratio/ms times (e.g., 'Cut -3dB at 250Hz', '4:1 compression'). "
                    "Mention specific Logic Pro stock plugins and workflow tips. Use industry terminology. "
                    "Return JSON ONLY matching the provided schema."
                ),
                messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=True)}],
            )
            return json.loads(_strip_fences(response.content[0].text))
        except Exception as e:
            print(f"DEBUG: Claude AI Error on attempt {attempt + 1}: {e}")
    return None


# Use Gemini structured output so the response stays machine-parseable.
def _try_gemini(prompt: dict) -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or genai is None:
        return None
    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
                contents=(
                    "You are a Senior Sound Designer & Logic Pro Specialist. "
                    "Analyze these audio features and track metadata: " + json.dumps(prompt, ensure_ascii=True) + ". "
                    "If the track is recognizable from the metadata, use your knowledge of its real-world production. Suggest specific reference_tracks. "
                    "Provide professional mixing (EQ, Dynamics, Space, Saturation, Modulation), sound architecture (Synthesis/Tips), "
                    "musical characteristics (mood/harmony/rhythm), and arrangement narrative (structure/energy). "
                    "You MUST mandate specific parameter values (e.g., '-2dB Bell at 4kHz', 'Fast attack 2ms'). "
                    "Refer to Logic Pro X features and stock plugins specifically (e.g., Phat FX, Step FX, Alchemy). "
                    "Keep descriptions technically dense but concise. "
                    "Return valid JSON matching the schema."
                ),
                config={"response_mime_type": "application/json", "response_json_schema": SCHEMA, "temperature": 0.4},
            )
            return json.loads(_strip_fences(response.text))
        except Exception as e:
            print(f"DEBUG: Gemini AI Error on attempt {attempt + 1}: {e}")
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
