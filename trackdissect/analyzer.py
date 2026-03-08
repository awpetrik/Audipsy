from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma) -> str:
    key_index = int(np.argmax(np.mean(chroma, axis=1)))
    return KEYS[key_index]


def _energy_level(rms: float) -> str:
    if rms < 0.05:
        return "low"
    if rms < 0.12:
        return "medium"
    return "high"


def _to_scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


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
    y, sr = _load_audio(path, mono=True)
    if y.size == 0:
        raise ValueError("The audio file appears to be empty.")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = _to_scalar(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    bandwidth = _to_scalar(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    rms = _to_scalar(np.mean(librosa.feature.rms(y=y)))
    low_hz = max(20, int(centroid - bandwidth / 2))
    high_hz = max(low_hz + 50, int(centroid + bandwidth / 2))

    return {
        "bpm": round(_to_scalar(tempo), 1),
        "key": _estimate_key(chroma),
        "spectral_centroid": round(centroid, 1),
        "rms_energy": round(rms, 4),
        "energy_level": _energy_level(rms),
        "frequency_range": {"low_hz": low_hz, "high_hz": high_hz},
        "duration_seconds": round(len(y) / sr, 2),
    }
