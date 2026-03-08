from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    from .ai_report import generate_report
    from .analyzer import analyze_audio, trim_audio
    from .separator import separate_audio
except ImportError:
    from ai_report import generate_report
    from analyzer import analyze_audio, trim_audio
    from separator import separate_audio

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
for folder in (UPLOAD_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".mp3", ".wav", ".aac", ".m4a"}
ICONS = {"vocals": "mic", "instrumental": "inst", "drums": "drum", "bass": "sub", "guitar": "gtr", "piano": "key", "other": "wave"}
JOBS: dict[str, dict] = {}
app = FastAPI(title="TrackDissect")
ACTIVITY_LIMIT = 8
TECHNICAL_LOG_LIMIT = 80


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _update(job_id: str, **fields) -> None:
    fields.setdefault("updated_at", _timestamp())
    JOBS[job_id].update(fields)


def _append_log(job_id: str, text: str, *, phase: str | None = None, technical: bool = False) -> None:
    key = "technical_logs" if technical else "activity"
    limit = TECHNICAL_LOG_LIMIT if technical else ACTIVITY_LIMIT
    entry = {"at": _timestamp(), "phase": phase or JOBS[job_id].get("stage", "queue"), "text": text}
    log = JOBS[job_id].setdefault(key, [])
    if log and log[-1]["text"] == text and log[-1]["phase"] == entry["phase"]:
        log[-1] = entry
    else:
        log.append(entry)
        del log[:-limit]


def _set_stage(job_id: str, *, stage: str, progress: int, message: str, event: str | None = None) -> None:
    _update(job_id, stage=stage, progress=progress, message=message)
    if event:
        _append_log(job_id, event, phase=stage)


# Run the full separation and analysis workflow in the background after upload.
def process_job(job_id: str) -> None:
    job = JOBS[job_id]
    if job["status"] != "queued":
        return
    try:
        _update(job_id, status="processing", started_at=_timestamp())
        _set_stage(job_id, stage="prepare", progress=10, message="Preparing selected clip...", event="Trim window locked. Preparing the analysis clip.")
        clip = trim_audio(job["upload_path"], OUTPUT_DIR / job_id / "selection.wav", job["trim_start"], job["trim_end"])
        _append_log(job_id, f"Selection ready: {clip['duration_seconds']}s clip exported for analysis.", phase="prepare")
        _set_stage(job_id, stage="features", progress=20, message="Extracting track features...", event="Extracting tempo, key, energy, and spectral profile.")
        track_features = analyze_audio(clip["path"])
        _append_log(job_id, f"Track features captured at {track_features['bpm']} BPM in key {track_features['key']}.", phase="features")
        _set_stage(job_id, stage="separate", progress=38, message="Separating vocals and stems with Demucs...", event="Demucs is loading models and splitting the clip. This is usually the longest step.")
        stems = separate_audio(
            clip["path"],
            OUTPUT_DIR / job_id,
            event_callback=lambda text: _append_log(job_id, text, phase="separate"),
            technical_callback=lambda text: _append_log(job_id, text, phase="separate", technical=True),
        )
        analyses = {}
        analysis_start = 48
        analysis_end = 84
        total_stems = max(1, len(stems))
        _set_stage(job_id, stage="analyze", progress=analysis_start, message="Inspecting separated stems...", event=f"Demucs finished. {len(stems)} outputs are ready for stem analysis.")
        for index, (stem, path) in enumerate(stems.items(), start=1):
            progress = analysis_start + round((index - 1) * (analysis_end - analysis_start) / total_stems)
            _update(job_id, progress=progress, message=f"Analyzing {stem} stem...")
            _append_log(job_id, f"Analyzing {stem} profile and frequency range.", phase="analyze")
            analyses[stem] = analyze_audio(path)
        _set_stage(job_id, stage="report", progress=88, message="Generating production report...", event="Writing the production summary and instrument notes.")
        report = generate_report(job["filename"], track_features, analyses)
        stem_data = {}
        for stem, features in analyses.items():
            stem_data[stem] = {
                "icon": ICONS.get(stem, stem[:3]),
                "preview_url": f"/media/{job_id}/{stem}",
                "download_url": f"/media/{job_id}/{stem}",
                "features": features,
                "analysis": report["stems"].get(stem, {"detected_instruments": [stem], "likely_fx": [], "confidence": "low"}),
            }
        _update(
            job_id,
            status="done",
            stage="done",
            progress=100,
            message="Analysis complete.",
            stem_files={name: str(path) for name, path in stems.items()},
            results={
                "job_id": job_id,
                "filename": job["filename"],
                "selection": clip,
                "track_features": track_features,
                "stems": stem_data,
                "full_report": report["full_report"],
            },
        )
        _append_log(job_id, f"Analysis complete. {len(stem_data)} downloadable outputs are ready.", phase="done")
    except Exception as exc:
        _update(job_id, status="error", stage="error", progress=100, message=str(exc))
        _append_log(job_id, f"Processing stopped: {exc}", phase="error")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "index.html")


@app.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    trim_start: float = Form(0),
    trim_end: float | None = Form(None),
):
    if not file.filename or Path(file.filename).suffix.lower() not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Upload an MP3, WAV, AAC, or M4A file.")
    payload = await file.read()
    if len(payload) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be 10MB or less.")
    job_id = uuid4().hex
    filename = Path(file.filename).name
    upload_path = UPLOAD_DIR / f"{job_id}{Path(filename).suffix.lower()}"
    upload_path.write_bytes(payload)
    JOBS[job_id] = {
        "status": "queued",
        "stage": "queue",
        "progress": 5,
        "message": "Upload received.",
        "filename": filename,
        "file_size_bytes": len(payload),
        "upload_path": str(upload_path),
        "trim_start": trim_start,
        "trim_end": trim_end,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "started_at": None,
        "activity": [{"at": _timestamp(), "phase": "queue", "text": "Upload received. Waiting to start the background job."}],
        "technical_logs": [],
    }
    background_tasks.add_task(process_job, job_id)
    return {"job_id": job_id, "filename": filename, "trim_start": trim_start, "trim_end": trim_end}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = _job(job_id)
    return {
        "status": job["status"],
        "stage": job.get("stage", "queue"),
        "progress": job["progress"],
        "message": job["message"],
        "filename": job["filename"],
        "file_size_bytes": job.get("file_size_bytes"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "trim_start": job.get("trim_start"),
        "trim_end": job.get("trim_end"),
        "source_url": f"/source/{job_id}",
        "activity": job.get("activity", []),
        "technical_logs": job.get("technical_logs", []),
    }


@app.get("/results/{job_id}")
def job_results(job_id: str):
    job = _job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Results are not ready yet.")
    return job["results"]


@app.get("/source/{job_id}")
def source_media(job_id: str):
    job = _job(job_id)
    path = Path(job["upload_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Source file not found.")
    return FileResponse(path, filename=job["filename"])


@app.get("/stems/{job_id}")
def job_stems(job_id: str):
    job = _job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Stem downloads are not ready yet.")
    return {stem: f"/media/{job_id}/{stem}" for stem in job["stem_files"]}


@app.get("/media/{job_id}/{stem}")
def media(job_id: str, stem: str):
    job = _job(job_id)
    source = job.get("stem_files", {}).get(stem)
    path = Path(source) if source else None
    if not path or not path.is_file():
        raise HTTPException(status_code=404, detail="Stem file not found.")
    return FileResponse(path, filename=path.name)
