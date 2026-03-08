# Audipsy

Audipsy is an MVP web application for AI-assisted music production analysis. It accepts a single MP3 or WAV upload, generates a waveform preview in the browser, separates the track into four stems with Demucs, extracts audio features with Librosa, and produces a production report with Gemini as the primary AI provider.

The application uses a FastAPI backend and a single-file HTML, CSS, and JavaScript frontend. Job state is stored in memory, which keeps the implementation simple and easy to run locally.

## Features

- **12 MB Upload Limit** (Expanded for better quality)
- Support for MP3, WAV, AAC, and M4A inputs
- **EDM Specialized Stems**: Beyond standard separation, Audipsy isolates:
  - **Sub-Bass vs Mid-Bass** (Precision 80Hz Crossover)
  - **Kick vs Top-Drums** (Precision 120Hz Isolation)
- **Anti-Artifact Smoothing**: 14kHz high-shelf filter to reduce digital harshness for AirPods/critical listening
- **Metadata Preservation**: Copies Artist, Title, Album, and **Album Art** to all downloaded stems
- **Processing Quality Modes**:
  - **Fast**: Rapid 1-shift separation
  - **Accurate**: High-fidelity 4-shift separation
- **Ruthless Performance**: Throttled atomic persistence and capped job history for stability
- Browser-based waveform preview and selectable trim window
- AI-generated production analysis with Gemini (Claude fallback)

## How It Works

1. The user uploads an audio file (up to 12MB).
2. The user selects a trim window and processing quality (Fast/Accurate).
3. The backend trims the audio and performs Demucs separation.
4. **EDM Crossover**: specialized filters split the Bass and Drums into sub-components.
5. **Post-Processing**: All stems are smoothed, peak-normalized (-0.5dBFS), and tagged with original metadata.
6. Librosa/Gemini analyze the sonic profile to generate a "Senior Sound Designer" report.
7. The frontend renders a premium dark-mode dashboard with SVG-iconed stem cards and a full arrangement narrative.

## Requirements

- Python 3.10 or newer
- `pip`
- `ffmpeg` (Required for metadata injection and format support)
- `demucs`
- Gemini API key

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r trackdissect/requirements.txt
```

## Running the Application

```bash
uvicorn trackdissect.main:app --reload
```

## Current Limitations

- Session limits: History is capped at the last 20 jobs to preserve memory.
- File size capped at 12 MB for local processing stability.
- Processing depends on local CPU/GPU performance (Accurate mode is resource-intensive).

## Troubleshooting

### Upload fails immediately

Check that the file is either `.mp3` or `.wav` and does not exceed 10 MB.

### Stem separation fails

Confirm that `demucs` is installed and available:

```bash
demucs --help
```

### Gemini responses are missing

Check that `GEMINI_API_KEY` is set in the same shell session used to launch Uvicorn.

### The app falls back to local analysis

This usually means the configured AI provider failed or no valid API key was available.

## Development Notes

- The backend is intentionally small and keeps each Python file under 150 lines
- The frontend is a single static HTML file served by FastAPI
- Generated uploads and stems are stored under `trackdissect/data/`

## Future Improvements

- Persistent job storage
- Better progress granularity from the separation pipeline
- Richer feature extraction and genre classification
- Multi-file history and session management
- Safer background job execution for larger workloads
