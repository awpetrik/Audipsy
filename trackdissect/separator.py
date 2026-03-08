import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from scipy.signal import butter, lfilter
import mutagen
from mutagen.wave import WAVE
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, TCON, error as MutagenError

import numpy as np
import soundfile as sf

FULL_STEM_MODELS = {
    "htdemucs_6s": ("vocals", "drums", "bass", "guitar", "piano", "other"),
    "htdemucs": ("vocals", "drums", "bass", "other"),
}
VOCAL_SPLIT_MODELS = ("htdemucs_ft", "htdemucs")
STEM_ORDER = ("vocals", "instrumental", "drums", "kick", "top_drums", "bass", "sub_bass", "mid_bass", "guitar", "piano", "other")


def _resolve_models(preferred: str, candidates: tuple[str, ...] | list[str]) -> list[str]:
    ordered = [preferred] + [name for name in candidates if name != preferred]
    seen = set()
    return [name for name in ordered if name and not (name in seen or seen.add(name))]


def _demucs_command_prefix() -> list[str]:
    configured = os.getenv("DEMUCS_BIN")
    if configured:
        return [configured]
    binary = shutil.which("demucs")
    if binary:
        return [binary]
    venv_binary = Path(sys.executable).with_name("demucs")
    if venv_binary.is_file():
        return [str(venv_binary)]
    return [sys.executable, "-m", "demucs.separate"]


def _build_demucs_command(model: str, source: Path, output_dir: Path, *, two_stems: str | None = None, mode: str = "fast") -> list[str]:
    cmd = [*_demucs_command_prefix(), "-n", model, str(source), "-o", str(output_dir)]
    
    # Mode-based defaults
    default_shifts = "4" if mode == "accurate" else "1"
    default_overlap = "0.25" if mode == "accurate" else "0.1"
    
    shifts = os.getenv("DEMUCS_VOCAL_SHIFTS" if two_stems else "DEMUCS_SHIFTS", default_shifts)
    overlap = os.getenv("DEMUCS_VOCAL_OVERLAP" if two_stems else "DEMUCS_OVERLAP", default_overlap)
    segment = os.getenv("DEMUCS_VOCAL_SEGMENT" if two_stems else "DEMUCS_SEGMENT")
    output_format = os.getenv("DEMUCS_OUTPUT_FORMAT", "wav").lower()

    if two_stems:
        cmd[1:1] = ["--two-stems", two_stems]
    if shifts:
        cmd[1:1] = ["--shifts", shifts]
    if overlap:
        cmd[1:1] = ["--overlap", overlap]
    if segment:
        cmd[1:1] = ["--segment", segment]
    if output_format == "mp3":
        cmd[1:1] = ["--mp3"]
    elif output_format == "flac":
        cmd[1:1] = ["--flac"]
    return cmd


def _run_demucs(model: str, source: Path, output_dir: Path, *, two_stems: str | None = None, mode: str = "fast") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _build_demucs_command(model, source, output_dir, two_stems=two_stems, mode=mode),
        capture_output=True,
        text=True,
        check=False,
        timeout=int(os.getenv("DEMUCS_TIMEOUT", "1800")),
    )


def _result_detail(result: subprocess.CompletedProcess[str]) -> str:
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return lines[-1] if lines else ""


def _emit_technical_output(result: subprocess.CompletedProcess[str], callback=None) -> None:
    if callback is None:
        return
    seen = set()
    for chunk in (result.stdout, result.stderr):
        for raw_line in (chunk or "").splitlines():
            line = raw_line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            callback(line)


def _collect_stems(stem_dir: Path, names: tuple[str, ...]) -> dict[str, Path]:
    stems = {}
    for name in names:
        matches = list(stem_dir.glob(f"{name}.*"))
        if not matches:
            return {}
        stems[name] = matches[0]
    return stems


def _attempt_demucs(
    source: Path,
    output_dir: Path,
    models: list[str],
    *,
    stem_names: dict[str, tuple[str, ...]],
    two_stems: str | None = None,
    event_callback=None,
    technical_callback=None,
    pass_label: str = "separation",
    mode: str = "fast",
) -> dict[str, Path]:
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    for model in models:
        if model not in stem_names:
            continue
        try:
            if event_callback:
                event_callback(f"Starting {pass_label} with {model} ({mode} mode).")
            result = _run_demucs(model, source, output_dir, two_stems=two_stems, mode=mode)
        except FileNotFoundError as exc:
            raise RuntimeError("Demucs is not installed or not available on PATH.") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Stem separation timed out. Try a shorter trim range or increase DEMUCS_TIMEOUT.") from exc

        attempts.append((model, result))
        _emit_technical_output(result, technical_callback)
        if result.returncode != 0:
            if event_callback:
                fallback = " Trying the next fallback model." if model != models[-1] else ""
                event_callback(f"{pass_label.capitalize()} with {model} failed.{fallback}")
            continue
        stems = _collect_stems(output_dir / model / source.stem, stem_names[model])
        if stems:
            if event_callback:
                event_callback(f"{pass_label.capitalize()} complete with {model}.")
            return stems

    detail = _result_detail(attempts[-1][1]) if attempts else ""
    raise RuntimeError(f"Stem separation failed. {detail or 'Please try another file.'}")


def _mix_instrumental(stems: dict[str, Path], output_path: Path) -> Path:
    parts = [path for name, path in stems.items() if name != "vocals"]
    if not parts:
        raise RuntimeError("Instrumental mix could not be created because no accompaniment stems were generated.")

    sample_rate = None
    frame_count = 0
    channel_count = 0
    layers = []
    for path in parts:
        audio, sr = sf.read(path, always_2d=True)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError("Separated stems do not share the same sample rate.")
        layer = audio.astype(np.float32, copy=False)
        layers.append(layer)
        frame_count = max(frame_count, layer.shape[0])
        channel_count = max(channel_count, layer.shape[1])

    mix = np.zeros((frame_count, channel_count), dtype=np.float32)
    for layer in layers:
        padded = np.zeros((frame_count, channel_count), dtype=np.float32)
        padded[: layer.shape[0], : layer.shape[1]] = layer
        mix += padded

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix /= peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mix, sample_rate, subtype="PCM_16")
    return output_path


def _apply_deharsh_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply a gentle high-shelf filter to reduce digital 'sizzle'."""
    # Gentle high-cut around 14kHz to smooth out AI artifacts
    nyquist = 0.5 * sr
    cutoff = 14000
    if cutoff >= nyquist:
        return audio
        
    # Butter filter for smoothing
    b, a = butter(1, cutoff / nyquist, btype='low')
    return lfilter(b, a, audio, axis=0)

def _apply_crossover(audio: np.ndarray, sr: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Linkwitz-Riley 4th order crossover (implemented as stacked 2nd order Butterworth)."""
    nyquist = 0.5 * sr
    if cutoff >= nyquist:
        return audio, np.zeros_like(audio)
    
    # 4th order Linkwitz-Riley is two 2nd order Butterworths in series
    # But for simplicity and stability in 16-bit PCM, a 4th order Butterworth is often preferred 
    # to maintain -24dB/octave slope and phase alignment.
    b_low, a_low = butter(4, cutoff / nyquist, btype='low')
    low_band = lfilter(b_low, a_low, audio, axis=0)
    
    b_high, a_high = butter(4, cutoff / nyquist, btype='high')
    high_band = lfilter(b_high, a_high, audio, axis=0)
    
    return low_band, high_band

def _transfer_metadata(src_path: Path, dst_path: Path, stem_name: str) -> None:
    """Copy tags and artwork from original source to generated stem (WAV ID3)."""
    try:
        # Load source tags
        src_tags = mutagen.File(src_path, easy=False)
        if not src_tags:
            return

        # Prepare WAV for ID3 tagging (WAV files can host ID3 chunks)
        try:
            audio = WAVE(dst_path)
            audio.add_tags()
        except Exception:
            try:
                audio = WAVE(dst_path)
            except Exception:
                return

        # Copy standard tags if they exist
        tag_map = {
            "TIT2": ("title", "TIT2"), # Title
            "TPE1": ("artist", "TPE1"), # Artist
            "TALB": ("album", "TALB"), # Album
            "TCON": ("genre", "TCON"), # Genre
        }
        
        # Determine the destination tag type
        for id3_field, (src_field, alt_id3) in tag_map.items():
            val = None
            if id3_field in src_tags:
                val = src_tags[id3_field]
            elif src_field in src_tags:
                val = src_tags[src_field]
            
            if val:
                audio.tags[id3_field] = val

        # Handle Album Art specifically (APIC)
        # Search for APIC frame or raw image data in different formats
        art_frames = [f for f in src_tags.keys() if f.startswith("APIC") or f.startswith("covr")]
        if art_frames:
            # Inject artwork if present
            if art_frames[0].startswith("APIC"):
                audio.tags["APIC"] = src_tags[art_frames[0]]
            elif art_frames[0] == "covr":
                # MP4/M4A cover art handling
                try:
                    cover_data = src_tags["covr"][0]
                    audio.tags.add(APIC(
                        encoding=3,
                        mime='image/jpeg',
                        type=3,
                        desc=u'Cover',
                        data=cover_data
                    ))
                except Exception:
                    pass

        # Update title to include stem name for better UX
        title = None
        if "TIT2" in audio.tags:
            title = str(audio.tags["TIT2"])
        
        stem_label = stem_name.replace("_", " ").capitalize()
        if title:
            audio.tags["TIT2"] = TIT2(encoding=3, text=f"{title} ({stem_label})")
        else:
            audio.tags["TIT2"] = TIT2(encoding=3, text=f"{src_path.stem} ({stem_label})")

        audio.save()
    except Exception as e:
        # Silently fail if metadata transfer doesn't work; essential audio is prioritized
        print(f"DEBUG: Metadata transfer error: {e}")

# Run Demucs and return the generated stem files for the current job.
def separate_audio(input_path: str | Path, output_root: str | Path, *, mode: str = "fast", event_callback=None, technical_callback=None) -> dict:
    source = Path(input_path)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_models = _resolve_models(os.getenv("DEMUCS_MODEL", "htdemucs_6s"), tuple(FULL_STEM_MODELS))
    full_stems = _attempt_demucs(
        source,
        output_dir / "full",
        full_models,
        stem_names=FULL_STEM_MODELS,
        event_callback=event_callback,
        technical_callback=technical_callback,
        pass_label="full stem pass",
        mode=mode,
    )

    split_models = _resolve_models(os.getenv("DEMUCS_VOCAL_MODEL", "htdemucs_ft"), VOCAL_SPLIT_MODELS)
    split_stems: dict[str, Path] = {}
    try:
        split_stems = _attempt_demucs(
            source,
            output_dir / "vocal_split",
            split_models,
            stem_names={name: ("vocals", "no_vocals") for name in VOCAL_SPLIT_MODELS},
            two_stems="vocals",
            event_callback=event_callback,
            technical_callback=technical_callback,
            pass_label="dedicated vocal split",
        )
    except RuntimeError:
        if event_callback:
            event_callback("Dedicated vocal split did not complete. Rebuilding instrumental from the accompaniment stems instead.")
        split_stems = {}

    combined = dict(full_stems)
    if split_stems.get("vocals"):
        combined["vocals"] = split_stems["vocals"]
    if split_stems.get("no_vocals"):
        combined["instrumental"] = split_stems["no_vocals"]
        if event_callback:
            event_callback("Instrumental exported directly from the Demucs no_vocals output.")
    else:
        combined["instrumental"] = _mix_instrumental(full_stems, output_dir / "derived" / "instrumental.wav")
        if event_callback:
            event_callback("Instrumental mix rebuilt from the non-vocal stems.")

    # Generate EDM sub-stems and add to combined dictionary first
    if "bass" in combined:
        try:
            if event_callback:
                event_callback("Isolating Sub-Bass vs Mid-Bass crossover...")
            audio, sr = sf.read(combined["bass"], always_2d=True)
            low, high = _apply_crossover(audio, sr, 80)
            sub_path = output_dir / "sub_bass.wav"
            mid_path = output_dir / "mid_bass.wav"
            sf.write(sub_path, low, sr, subtype="PCM_16")
            sf.write(mid_path, high, sr, subtype="PCM_16")
            combined["sub_bass"] = sub_path
            combined["mid_bass"] = mid_path
        except Exception as e:
            if technical_callback:
                technical_callback(f"Bass crossover failed: {e}")

    if "drums" in combined:
        try:
            if event_callback:
                event_callback("Isolating Kick vs Top-Drums crossover...")
            audio, sr = sf.read(combined["drums"], always_2d=True)
            low, high = _apply_crossover(audio, sr, 120)
            kick_path = output_dir / "kick.wav"
            top_path = output_dir / "top_drums.wav"
            sf.write(kick_path, low, sr, subtype="PCM_16")
            sf.write(top_path, high, sr, subtype="PCM_16")
            combined["kick"] = kick_path
            combined["top_drums"] = top_path
        except Exception as e:
            if technical_callback:
                technical_callback(f"Drum crossover failed: {e}")

    # Final pass: Smoothing, Metadata Injection, and ordering
    ordered = OrderedDict()
    for name in STEM_ORDER:
        path = combined.get(name)
        if not path:
            continue
            
        try:
            # Post-processing: Anti-Artifact Smoothing
            if event_callback:
                event_callback(f"Finalizing {name} stem...")
            audio, sr = sf.read(path, always_2d=True)
            audio = _apply_deharsh_filter(audio, sr)
            
            # Peak normalization to prevent clipping after filter
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak) * 0.95 # -0.5dBFS
            
            sf.write(path, audio, sr, subtype="PCM_16")
            
            # Metadata Injection
            _transfer_metadata(source, path, name)
        except Exception as e:
            if technical_callback:
                technical_callback(f"Finalization failed for {name}: {e}")
        
        ordered[name] = path

    return ordered
