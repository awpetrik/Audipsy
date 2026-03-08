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
ICONS = {"vocals": "mic", "drums": "drum", "bass": "sub", "guitar": "gtr", "piano": "key", "other": "wave"}
JOBS: dict[str, dict] = {}
app = FastAPI(title="TrackDissect")

def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _update(job_id: str, **fields) -> None:
    JOBS[job_id].update(fields)


# Run the full separation and analysis workflow in the background after upload.
def process_job(job_id: str) -> None:
    job = JOBS[job_id]
    if job["status"] != "queued":
        return
    try:
        _update(job_id, status="processing", progress=10, message="Preparing selected clip...")
        clip = trim_audio(job["upload_path"], OUTPUT_DIR / job_id / "selection.wav", job["trim_start"], job["trim_end"])
        _update(job_id, progress=20, message="Extracting track features...")
        track_features = analyze_audio(clip["path"])
        _update(job_id, progress=38, message="Separating stems with Demucs...")
        stems = separate_audio(clip["path"], OUTPUT_DIR / job_id)
        analyses = {}
        for index, (stem, path) in enumerate(stems.items(), start=1):
            _update(job_id, progress=38 + index * 11, message=f"Analyzing {stem} stem...")
            analyses[stem] = analyze_audio(path)
        _update(job_id, progress=88, message="Generating production report...")
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
    except Exception as exc:
        _update(job_id, status="error", progress=100, message=str(exc))


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
        "progress": 5,
        "message": "Upload received.",
        "filename": filename,
        "upload_path": str(upload_path),
        "trim_start": trim_start,
        "trim_end": trim_end,
    }
    background_tasks.add_task(process_job, job_id)
    return {"job_id": job_id, "filename": filename, "trim_start": trim_start, "trim_end": trim_end}


@app.get("/status/{job_id}")
def job_status(job_id: str):
    job = _job(job_id)
    return {key: job[key] for key in ("status", "progress", "message", "filename")}


@app.get("/results/{job_id}")
def job_results(job_id: str):
    job = _job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Results are not ready yet.")
    return job["results"]


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
