from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tinytag import TinyTag

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma) -> str:
    key_index = int(np.argmax(np.mean(chroma, axis=1)))
    return KEYS[key_index]


def _extract_metadata(path: str | Path) -> dict:
    try:
        tag = TinyTag.get(path)
        return {
            "artist": tag.artist,
            "title": tag.title,
            "album": tag.album,
            "genre": tag.genre
        }
    except Exception:
        return {}


def _energy_level(rms: float) -> str:
    if rms < 0.05:
        return "low"
    if rms < 0.12:
        return "medium"
    return "high"


def _to_scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _safe_divide(a, b):
    return a / b if b != 0 else 0


def _load_audio(path: str | Path, mono: bool):
    try:
        return librosa.load(path, sr=None, mono=mono)
    except Exception as exc:
        raise ValueError("Unable to decode this audio file. For AAC or M4A, make sure ffmpeg is installed.") from exc


# Create a temporary clip from the chosen time range before detection runs.
def trim_audio(path: str | Path, output_path: str | Path, start_seconds: float, end_seconds: float | None) -> dict:
    y, sr = _load_audio(path, mono=False)
    total = librosa.get_duration(y=y, sr=sr)
    start = max(0.0, float(start_seconds or 0.0))
    end = total if end_seconds is None else min(total, float(end_seconds))
    if end <= start:
        raise ValueError("Trim end must be greater than trim start.")
    clip = y[..., int(start * sr):int(end * sr)]
    if clip.size == 0:
        raise ValueError("The selected trim range is empty.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, clip.T if clip.ndim > 1 else clip, sr)
    return {"path": str(target), "start_seconds": round(start, 2), "end_seconds": round(end, 2), "duration_seconds": round(end - start, 2)}


# Extract the core stats the reporting layer needs for each uploaded track or stem.
def analyze_audio(path: str | Path) -> dict:
    # Load stereo for correlation analysis, then mono for most features.
    y_stereo, sr = _load_audio(path, mono=False)
    if y_stereo.size == 0:
        raise ValueError("The audio file appears to be empty.")

    y = librosa.to_mono(y_stereo) if y_stereo.ndim > 1 else y_stereo

    # Basic features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    rms_scalar = _to_scalar(np.mean(rms))

    # Advanced metrics
    # Crest Factor (Peak-to-RMS ratio in dB)
    peak = np.max(np.abs(y))
    crest_factor = 20 * np.log10(_safe_divide(peak, rms_scalar)) if rms_scalar > 0 else 0

    # Stereo Correlation (-1 to 1)
    correlation = 0.0
    if y_stereo.ndim > 1 and y_stereo.shape[0] == 2:
        l, r = y_stereo[0], y_stereo[1]
        num = np.sum(l * r)
        den = np.sqrt(np.sum(l**2) * np.sum(r**2))
        correlation = _safe_divide(num, den)

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    flatness = _to_scalar(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = _to_scalar(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    # Sub-bass energy (Below 60Hz)
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    sub_bass_mask = freqs < 60
    sub_bass_energy = np.sum(stft[sub_bass_mask, :]) / np.sum(stft) if np.sum(stft) > 0 else 0

    centroid_scalar = _to_scalar(np.mean(centroid))
    bandwidth_scalar = _to_scalar(np.mean(bandwidth))
    low_hz = max(20, int(centroid_scalar - bandwidth_scalar / 2))
    high_hz = max(low_hz + 50, int(centroid_scalar + bandwidth_scalar / 2))

    # Metadata extraction
    meta = _extract_metadata(path)

    # Tech FX Heuristics
    fx_guesses = []
    if correlation < 0.35 and y_stereo.ndim > 1:
        fx_guesses.append("Stereo Ambience")
    if float(crest_factor) < 8.5:
        fx_guesses.append("High Compression")
    if float(sub_bass_energy) > 0.35:
        fx_guesses.append("Sub Heavy")

    return {
        "metadata": meta,
        "bpm": round(_to_scalar(tempo), 1),
        "key": _estimate_key(chroma),
        "rms_energy": round(rms_scalar, 4),
        "crest_factor_db": round(float(crest_factor), 2),
        "stereo_correlation": round(float(correlation), 3),
        "spectral_flatness": round(float(flatness), 4),
        "zero_crossing_rate": round(float(zcr), 4),
        "sub_bass_ratio": round(float(sub_bass_energy), 4),
        "energy_level": _energy_level(rms_scalar),
        "frequency_range": {"low_hz": low_hz, "high_hz": high_hz},
        "duration_seconds": round(len(y) / sr, 2),
        "technical_fx": fx_guesses,
    }
