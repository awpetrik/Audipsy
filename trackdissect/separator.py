import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf

FULL_STEM_MODELS = {
    "htdemucs_6s": ("vocals", "drums", "bass", "guitar", "piano", "other"),
    "htdemucs": ("vocals", "drums", "bass", "other"),
}
VOCAL_SPLIT_MODELS = ("htdemucs_ft", "htdemucs")
STEM_ORDER = ("vocals", "instrumental", "drums", "bass", "guitar", "piano", "other")


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


def _build_demucs_command(model: str, source: Path, output_dir: Path, *, two_stems: str | None = None) -> list[str]:
    cmd = [*_demucs_command_prefix(), "-n", model, str(source), "-o", str(output_dir)]
    shifts = os.getenv("DEMUCS_VOCAL_SHIFTS" if two_stems else "DEMUCS_SHIFTS")
    overlap = os.getenv("DEMUCS_VOCAL_OVERLAP" if two_stems else "DEMUCS_OVERLAP")
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


def _run_demucs(model: str, source: Path, output_dir: Path, *, two_stems: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _build_demucs_command(model, source, output_dir, two_stems=two_stems),
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
) -> dict[str, Path]:
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    for model in models:
        if model not in stem_names:
            continue
        try:
            if event_callback:
                event_callback(f"Starting {pass_label} with {model}.")
            result = _run_demucs(model, source, output_dir, two_stems=two_stems)
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


# Run Demucs and return the generated stem files for the current job.
def separate_audio(input_path: str | Path, output_root: str | Path, *, event_callback=None, technical_callback=None) -> dict:
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

    ordered = OrderedDict()
    for name in STEM_ORDER:
        path = combined.get(name)
        if path:
            ordered[name] = path
    return ordered
