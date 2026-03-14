#!/usr/bin/env python3
"""Cross-platform quick-native launcher for Audipsy.

Features:
- Creates .venv automatically (if missing)
- Installs Python dependencies automatically (with hash-based cache)
- Detects busy port and auto-finds the next free port
- Validates required system tools (ffmpeg)
- Starts uvicorn with clear runtime diagnostics
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
DEPS_STAMP = VENV_DIR / ".deps.sha256"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MIN_PYTHON = (3, 10)


class LaunchError(RuntimeError):
    pass


def info(msg: str) -> None:
    print(f"[quick-native] {msg}")


def fail(msg: str, *, hint: str | None = None) -> None:
    print(f"[quick-native][error] {msg}", file=sys.stderr)
    if hint:
        print(f"[quick-native][hint] {hint}", file=sys.stderr)
    raise LaunchError(msg)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    process = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if process.returncode != 0:
        joined = " ".join(cmd)
        fail(f"Command failed (exit {process.returncode}): {joined}")


def verify_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required. Current: {sys.version.split()[0]}",
            hint="Install a newer Python version, then rerun this launcher.",
        )


def venv_python_path() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> Path:
    py = venv_python_path()
    if py.exists():
        return py

    info("Creating virtual environment (.venv)...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT)
    if not py.exists():
        fail("Virtual environment created but Python executable was not found.")
    return py


def requirements_hash() -> str:
    if not REQ_FILE.exists():
        fail("requirements.txt not found.")
    return hashlib.sha256(REQ_FILE.read_bytes()).hexdigest()


def ensure_python_deps(py: Path, *, force: bool = False, skip: bool = False) -> None:
    if skip:
        info("Skipping Python dependency installation (--skip-deps).")
        return

    req_hash = requirements_hash()
    installed_hash = DEPS_STAMP.read_text().strip() if DEPS_STAMP.exists() else ""

    if not force and installed_hash == req_hash:
        info("Python dependencies already up to date.")
        return

    info("Installing/updating Python dependencies...")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT)
    run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)], cwd=ROOT)

    DEPS_STAMP.write_text(req_hash)
    info("Dependency installation completed.")


def ensure_system_tools() -> None:
    if shutil.which("ffmpeg"):
        return

    os_name = platform.system()
    if os_name == "Darwin":
        hint = "Install ffmpeg with Homebrew: brew install ffmpeg"
    elif os_name == "Windows":
        hint = "Install ffmpeg and ensure it is on PATH: winget install Gyan.FFmpeg"
    else:
        hint = "Install ffmpeg from your package manager, e.g. apt install ffmpeg"

    fail("ffmpeg is required but not found on PATH.", hint=hint)


def can_bind(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def select_port(host: str, preferred: int, max_tries: int) -> tuple[int, bool]:
    if can_bind(host, preferred):
        return preferred, False

    port = preferred
    for _ in range(max_tries):
        port += 1
        if can_bind(host, port):
            return port, True

    fail(
        f"Could not find a free port after trying range {preferred}-{preferred + max_tries}.",
        hint="Stop other local servers or provide a custom --port.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick native launcher for Audipsy")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Server host (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", DEFAULT_PORT)),
        help=f"Preferred server port (default: env PORT or {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--max-port-tries",
        type=int,
        default=50,
        help="Max number of incremental ports to try when preferred port is occupied",
    )
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation checks")
    parser.add_argument("--force-deps", action="store_true", help="Force reinstall of Python dependencies")
    parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn auto-reload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        verify_python_version()
        ensure_system_tools()
        py = ensure_venv()
        ensure_python_deps(py, force=args.force_deps, skip=args.skip_deps)

        port, used_fallback = select_port(args.host, args.port, args.max_port_tries)
        if used_fallback:
            info(f"Port {args.port} is busy. Using free port {port} instead.")

        url = f"http://{args.host}:{port}"
        info(f"Starting Audipsy at {url}")

        cmd = [str(py), "-m", "uvicorn", "main:app", "--host", args.host, "--port", str(port)]
        if not args.no_reload:
            cmd.append("--reload")

        run(cmd, cwd=ROOT)
        return 0
    except LaunchError:
        return 1
    except KeyboardInterrupt:
        info("Interrupted by user.")
        return 130
    except Exception as exc:  # pragma: no cover
        fail(
            f"Unexpected launcher failure: {exc}",
            hint="Rerun with --force-deps, or use Docker fallback if issue persists.",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
