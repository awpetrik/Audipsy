# Audipsy

[![Stage](https://img.shields.io/badge/Stage-MVP-ff8a2a?style=for-the-badge)](README.md)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-0f172a?style=for-the-badge&logo=fastapi)](trackdissect/main.py)
[![Frontend](https://img.shields.io/badge/Frontend-Vanilla%20HTML%2FCSS%2FJS-1f2937?style=for-the-badge)](trackdissect/index.html)
[![Separation](https://img.shields.io/badge/Stem%20Engine-Demucs-111827?style=for-the-badge)](trackdissect/separator.py)
[![Analysis](https://img.shields.io/badge/Audio%20Features-Librosa-1e3a8a?style=for-the-badge)](trackdissect/analyzer.py)
[![AI](https://img.shields.io/badge/AI-Gemini%20Primary%20%7C%20Claude%20Fallback-374151?style=for-the-badge)](trackdissect/ai_report.py)
[![Python](https://img.shields.io/badge/Python-3.10%2B-2b6cb0?style=for-the-badge&logo=python)](trackdissect/requirements.txt)

---

![App Preview](https://github.com/user-attachments/assets/45a5ba95-cabb-4fcb-837b-a952937075c8)

| Build Context | Value |
|---|---|
| Runtime | FastAPI + Uvicorn |
| Launch Mode | Quick Native (auto-setup, auto-port fallback) |
| Input Formats | MP3, WAV, AAC, M4A |
| Upload Limit | 12 MB |

> **Design note** — This repository prioritizes a low-friction local trial flow: single-command launch, self-healing setup, and predictable API behavior.

---

Audipsy is an MVP web application for AI-assisted music production analysis.

It accepts a single audio upload, renders a waveform preview in the browser, separates stems with Demucs, extracts audio features with Librosa, and generates a production report with Gemini as the primary AI provider (with Claude fallback support).

The app uses:
- FastAPI backend
- Single-file frontend (HTML, CSS, vanilla JavaScript)
- In-memory job state with throttled atomic snapshots for local stability

## Features

- **12 MB Upload Limit**
- Input support: MP3, WAV, AAC, M4A
- Browser-based waveform preview with selectable trim window
- **Processing Quality Modes**:
  - **Fast**: Rapid 1-shift separation
  - **Accurate**: High-fidelity 4-shift separation
- **Skip AI Report**: Option to bypass AI generation for faster stem-only workflows
- **EDM-Specialized Separation**:
  - **Sub-Bass vs Mid-Bass** (precision 80 Hz crossover)
  - **Kick vs Top-Drums** (precision 120 Hz isolation)
  - **Lead Vocal vs Vocal Layers** via mid-side (MS) decomposition
- **Anti-Artifact Smoothing**: 14 kHz high-shelf filter to reduce digital harshness
- **Metadata Preservation**: Copies artist, title, album, and album art to all exported stems
- AI-generated production analysis with Gemini (Claude optional fallback)
- Throttled atomic persistence and capped job history for stable local runs

## Tech Stack

- Python
- FastAPI
- Librosa
- Demucs
- Mutagen (metadata and album art)
- Google Gemini API (primary)
- Anthropic Claude API (optional fallback)
- HTML, CSS, vanilla JavaScript

## Project Structure

```text
trackdissect/
├── ai_report.py      # AI prompting and report generation
├── analyzer.py       # Audio feature extraction and trimming
├── index.html        # Frontend UI
├── main.py           # FastAPI app and job management
├── quick_native.py   # Cross-platform launcher with auto-setup
├── requirements.txt
├── run-native.bat    # Windows wrapper
├── run-native.sh     # macOS/Linux wrapper
└── separator.py      # Demucs processing and split logic
```

## Quick Start (Native, Cross-Platform)

### macOS / Linux

```bash
cd trackdissect
./run-native.sh
```

### Windows (cmd)

```bat
cd trackdissect
run-native.bat
```

The launcher automatically:
- verifies Python version
- checks ffmpeg availability
- creates .venv if missing
- installs dependencies from requirements.txt when needed
- resolves port conflicts by selecting the next free port

## Custom Port and Runtime Options

### Prefer a specific port

```bash
./run-native.sh --port 9000
```

If the selected port is occupied, Audipsy auto-falls back to the next available port.

### Useful flags

```bash
./run-native.sh --help
```

- `--host 127.0.0.1`
- `--port 8000`
- `--max-port-tries 50`
- `--skip-deps`
- `--force-deps`
- `--no-reload`

## How It Works

1. User uploads one track (up to 12 MB).
2. User selects a trim window and processing quality (Fast / Accurate).
3. Backend trims audio and performs Demucs separation.
4. **Vocal & EDM splits**: Specialized filters and MS decomposition split vocals (Lead / Layers), bass (Sub / Mid), and drums (Kick / Top).
5. **Post-processing**: All stems are smoothed, peak-normalized (−0.5 dBFS), and tagged with original metadata.
6. Librosa features and AI analysis build the production report.
7. Frontend renders a dark-mode dashboard with stem cards and a full arrangement narrative.

## Configuration

Audipsy reads configuration from environment variables.

### Required for Gemini

```bash
export GEMINI_API_KEY="your_api_key"
```

### Optional

```bash
export AI_PROVIDER=auto   # gemini, claude, or auto
export GEMINI_MODEL=gemini-3-flash-preview
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## API Endpoints

### `POST /upload`
Accepts a single file upload and returns a job identifier.

### `GET /status/{job_id}`
Returns current state, progress, activity logs, and technical logs.

### `GET /results/{job_id}`
Returns the full analysis payload once processing is complete.

## Analysis Output

Each completed job includes:
- **Track-level features**: BPM, key, energy, danceability.
- **Per-stem features**: Spectral centroid, flatness, crest factor, stereo correlation.
- **AI production report**:
  - Genre and similar artists
  - Musical characteristics (mood, harmony, rhythm)
  - Arrangement narrative (structure, energy flow)
  - Sound architecture and synthesis tips
  - Workflow tip

## Requirements

- Python 3.10+
- ffmpeg

### ffmpeg install hints

- macOS: `brew install ffmpeg`
- Windows: `winget install Gyan.FFmpeg`
- Linux: install from your distro package manager (e.g. `apt install ffmpeg`)

## Manual Run (Advanced)

```bash
cd trackdissect
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Windows PowerShell activation:

```powershell
cd trackdissect
.\.venv\Scripts\Activate.ps1
```

## Current Limitations

- Job history is capped for memory stability.
- File size is capped at 12 MB for local processing reliability.
- Accurate mode is significantly more resource-intensive than Fast mode.

## Troubleshooting

### Upload fails immediately

Check the file type and ensure it does not exceed 12 MB.

### Stem separation fails

Check Demucs and ffmpeg availability:

```bash
demucs --help
ffmpeg -version
```

### Gemini responses are missing

Ensure `GEMINI_API_KEY` is set in the same shell session used to launch the app.

### App falls back to local analysis

This usually means the configured AI provider failed or no valid API key was available.

### Port already in use

No manual action required when using quick-native launchers; they auto-select a free port.

### Dependency install fails

Retry with:

```bash
./run-native.sh --force-deps
```

## Development Notes

- Frontend is intentionally kept as a single static HTML file served by FastAPI.
- Generated uploads and outputs are stored under `trackdissect/data/`.

## Future Improvements

- Persistent job storage
- Better progress granularity from the separation pipeline
- Richer feature extraction and genre classification
- Multi-file history and session management
- Safer background job execution for larger workloads
