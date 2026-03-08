# TrackDissect

TrackDissect is an MVP web application for AI-assisted music production analysis. It accepts a single MP3 or WAV upload, generates a waveform preview in the browser, separates the track into four stems with Demucs, extracts audio features with Librosa, and produces a production report with Gemini as the primary AI provider.

The application uses a FastAPI backend and a single-file HTML, CSS, and JavaScript frontend. Job state is stored in memory, which keeps the implementation simple and easy to run locally.

## Features

- Single audio upload with a 10 MB limit
- Support for MP3, WAV, AAC, and M4A inputs
- Browser-based waveform preview using the Web Audio API
- Selectable trim window before analysis
- Stem separation into `vocals`, `drums`, `bass`, and `other`
- Extended separation with `guitar` and `piano` when the 6-stem Demucs model is available
- Audio feature extraction for the full track and each stem
- AI-generated production analysis with Gemini
- Optional Claude support as a fallback provider
- Stem preview and download links in the UI
- Progress polling during processing

## Tech Stack

- Python
- FastAPI
- Librosa
- Demucs
- Google Gemini API
- Anthropic Claude API as optional fallback
- HTML, CSS, and vanilla JavaScript

## Project Structure

```text
trackdissect/
├── ai_report.py
├── analyzer.py
├── index.html
├── main.py
├── requirements.txt
└── separator.py
```

## How It Works

1. The user uploads an MP3, WAV, AAC, or M4A file from the frontend.
2. The user selects the exact time window to analyze.
3. The backend stores the file locally and creates an in-memory job record.
4. A background task trims the audio to the selected clip.
5. Librosa extracts high-level features from the chosen clip.
6. Demucs separates the selected clip into four stems, or six stems when the extended model succeeds.
7. Librosa analyzes each stem for tempo, key, spectral centroid, RMS energy, and estimated frequency range.
8. Gemini converts those features into production-oriented observations such as likely instruments, effects, genre, and stylistic references.
9. The frontend polls job status and renders the completed report with stem playback and downloads.

## Requirements

Before running the app, make sure the following are available:

- Python 3.10 or newer
- `pip`
- `demucs` installed through Python dependencies and available on `PATH`
- A Gemini API key for cloud analysis

Depending on your environment, Demucs may also require additional audio tooling for full format support.

## Installation

Clone the repository, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r trackdissect/requirements.txt
```

## Configuration

TrackDissect reads configuration from environment variables.

### Required for Gemini

```bash
export GEMINI_API_KEY="your_api_key"
```

### Optional

```bash
export AI_PROVIDER=gemini
export GEMINI_MODEL=gemini-2.5-flash
export ANTHROPIC_API_KEY="your_anthropic_key"
export CLAUDE_MODEL=claude-sonnet-4-20250514
```

### Provider Behavior

- `AI_PROVIDER=gemini` uses Gemini only
- `AI_PROVIDER=claude` uses Claude only
- `AI_PROVIDER=auto` tries Gemini first, then Claude
- If no cloud provider succeeds, the app falls back to a local heuristic report

Gemini is the default provider if `AI_PROVIDER` is not set.

## Running the Application

Start the FastAPI server from the repository root:

```bash
uvicorn trackdissect.main:app --reload
```

Open the app in your browser:

```text
http://127.0.0.1:8000
```

## API Endpoints

### `POST /upload`

Accepts a single file upload and returns a job identifier.

Example response:

```json
{
  "job_id": "abc123",
  "filename": "track.wav"
}
```

### `GET /status/{job_id}`

Returns the current processing state, progress percentage, and status message.

Example response:

```json
{
  "status": "processing",
  "progress": 47,
  "message": "Analyzing drums stem...",
  "filename": "track.wav"
}
```

### `GET /results/{job_id}`

Returns the full analysis payload once processing is complete.

### `GET /stems/{job_id}`

Returns stem download URLs for completed jobs.

### `GET /media/{job_id}/{stem}`

Streams an individual stem file for playback or download.

## Analysis Output

Each completed job includes:

- The selected analysis window
- Track-level features such as BPM and key
- Per-stem feature summaries
- AI-detected instruments
- Likely effects and processing clues
- Confidence labels
- A full production report with:
  - genre guess
  - production style
  - notable techniques
  - similar artists

## Current Limitations

- Jobs are stored in memory only and will be lost when the server restarts
- There is no persistent database or queue
- Upload size is capped at 10 MB
- AAC and M4A decoding depends on local media support such as ffmpeg
- Processing time depends heavily on Demucs performance
- The 6-stem Demucs model may fall back to the standard 4-stem model depending on environment support
- The fallback local report is heuristic and less detailed than cloud output

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
