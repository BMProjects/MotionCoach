"""FastAPI service for the MVP.

Single process, one analysis worker thread, filesystem storage under data/. The API shape
(upload -> job -> poll -> report/poses) is the one a production deployment keeps; only the
executor (queue + GPU workers) and storage (object store + DB) change.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import settings
from ..apps.analysis import AnalysisOptions, run_analysis
from ..apps.assets import get_reference, list_references, load_profile
from ..apps.scope import ScopeRequest
from ..perception.registry import DEFAULT_BACKEND, SPECS, list_backends

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_UPLOAD_BYTES = 2 * 1024 ** 3
STAGE_WEIGHTS = {"queued": 0, "ingest": 0.05, "locate": 0.15, "poses_ref": 0.25, "poses_user": 0.4, "features": 0.03,
                 "align": 0.06, "compare": 0.06}

app = FastAPI(title="motioncoach MVP")
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _jobs_dir() -> Path:
    d = settings.DATA_DIR / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_job(job: dict) -> None:
    d = _jobs_dir() / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")


def _load_job(job_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        raise HTTPException(404, "job not found")
    with _lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    p = _jobs_dir() / job_id / "job.json"
    if not p.exists():
        raise HTTPException(404, "job not found")
    return json.loads(p.read_text(encoding="utf-8"))


class UserVideo(BaseModel):
    upload_id: str | None = None
    local: str | None = None         # file name inside assets/ (development convenience)


class JobRequest(BaseModel):
    reference_id: str
    user: UserVideo
    sample_fps: float = Field(10.0, ge=2, le=30)
    backend: str = DEFAULT_BACKEND
    user_mirrored: str = Field("auto", pattern="^(auto|yes|no)$")
    mirror_practice: bool = False
    scope_mode: str = Field("auto", pattern="^(auto|segments|manual)$")
    seg_from: str | None = None
    seg_to: str | None = None
    user_start: float | None = Field(None, ge=0)
    user_end: float | None = Field(None, ge=0)


def _resolve_user_video(u: UserVideo) -> Path:
    if u.upload_id:
        if not re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{2,5}", u.upload_id):
            raise HTTPException(400, "bad upload id")
        p = settings.DATA_DIR / "uploads" / u.upload_id
    elif u.local:
        p = (settings.ASSETS_DIR / u.local).resolve()
        if p.parent != settings.ASSETS_DIR.resolve():
            raise HTTPException(400, "local video must be inside assets/")
    else:
        raise HTTPException(400, "user video required")
    if not p.exists() or p.suffix.lower() not in VIDEO_EXT:
        raise HTTPException(404, "user video not found")
    return p


def _run(job_id: str, req: JobRequest, user_path: Path) -> None:
    order = list(STAGE_WEIGHTS)

    def progress(stage: str, frac: float) -> None:
        with _lock:
            job = _jobs[job_id]
            if stage == "done":
                return
            done = sum(STAGE_WEIGHTS[s] for s in order[: order.index(stage)]) if stage in order else 0
            job.update(stage=stage, progress=round(done + STAGE_WEIGHTS.get(stage, 0) * frac, 3))

    try:
        with _lock:
            _jobs[job_id]["status"] = "running"
        ref = get_reference(req.reference_id, settings.REFERENCES_DIR, settings.ROOT)
        opts = AnalysisOptions(sample_fps=req.sample_fps, backend=req.backend,
                               user_mirrored=req.user_mirrored, mirror_practice=req.mirror_practice,
                               scope=ScopeRequest(req.scope_mode, req.seg_from, req.seg_to, req.user_start, req.user_end))
        run_analysis(ref, user_path, _jobs_dir() / job_id, settings.DATA_DIR, settings.MODELS_DIR, opts, progress)
        with _lock:
            _jobs[job_id].update(status="done", stage="done", progress=1.0, finished=time.time())
    except Exception as exc:  # report to the client; keep the traceback in job.json
        with _lock:
            _jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}",
                                 traceback=traceback.format_exc(limit=8))
    finally:
        with _lock:
            _save_job(_jobs[job_id])


@app.get("/api/references")
def references() -> list[dict]:
    out = []
    for r in list_references(settings.REFERENCES_DIR, settings.ROOT):
        prof = load_profile(r.raw["profile"])
        out.append({"id": r.id, "title": r.raw.get("title"), "status": r.raw.get("status"),
                    "profile": prof.id, "profile_title": prof.raw["title"], "view": r.raw.get("view"),
                    "available": r.video.exists(),
                    "segments": [{"id": s.id, "name": s.name, "start": s.start, "end": s.end} for s in r.segments(prof)]})
    return out


@app.get("/api/local-videos")
def local_videos() -> list[dict]:
    """Development helper: videos already in assets/."""
    return [{"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
            for p in sorted(settings.ASSETS_DIR.iterdir()) if p.suffix.lower() in VIDEO_EXT]


@app.post("/api/uploads")
async def upload(file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported file type {ext!r}")
    d = settings.DATA_DIR / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    uid = f"{uuid.uuid4().hex}{ext}"
    size = 0
    with open(d / uid, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                (d / uid).unlink(missing_ok=True)
                raise HTTPException(413, "file too large")
            f.write(chunk)
    return {"upload_id": uid, "filename": file.filename, "size_mb": round(size / 1e6, 1)}


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    get_reference(req.reference_id, settings.REFERENCES_DIR, settings.ROOT)
    if req.backend not in SPECS:
        raise HTTPException(400, f"unknown backend {req.backend}")
    if req.backend != DEFAULT_BACKEND and not settings.DEV_MODE:
        raise HTTPException(403, "非默认姿态算法仅在开发者模式下可用（MOTIONCOACH_DEV=1），算法对比请用 motioncoach bench")
    if req.scope_mode != "auto" and not req.seg_from:
        raise HTTPException(400, "seg_from required for segments/manual scope")
    if req.scope_mode == "manual" and (req.user_start is None or req.user_end is None or req.user_end <= req.user_start):
        raise HTTPException(400, "manual scope needs user_start < user_end")
    user_path = _resolve_user_video(req.user)
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0, "created": time.time(),
           "request": req.model_dump(), "user_filename": req.user.local or req.user.upload_id}
    with _lock:
        _jobs[job_id] = job
        _save_job(job)
    _executor.submit(_run, job_id, req, user_path)
    return job


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    jobs = []
    for p in sorted(_jobs_dir().glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        j = json.loads(p.read_text(encoding="utf-8"))
        with _lock:
            j = dict(_jobs.get(j["id"], j))
        j.pop("traceback", None)
        jobs.append(j)
    return jobs


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _load_job(job_id)


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str) -> FileResponse:
    _load_job(job_id)
    p = _jobs_dir() / job_id / "report.json"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    return FileResponse(p, media_type="application/json")


@app.get("/api/jobs/{job_id}/poses/{side}")
def get_poses(job_id: str, side: str) -> FileResponse:
    if side not in ("ref", "user"):
        raise HTTPException(404, "unknown side")
    _load_job(job_id)
    p = _jobs_dir() / job_id / f"poses_{side}.json.gz"
    if not p.exists():
        raise HTTPException(404, "poses not ready")
    return FileResponse(p, media_type="application/json", headers={"Content-Encoding": "gzip"})


@app.get("/api/config")
def config() -> dict:
    return {"dev_mode": settings.DEV_MODE, "default_backend": DEFAULT_BACKEND}


@app.get("/api/backends")
def backends() -> list[dict]:
    return list_backends()


@app.get("/api/jobs/{job_id}/export.csv")
def export_csv(job_id: str) -> Response:
    """Per-moment aligned data for researchers: reference time, matched user time, segment, values."""
    _load_job(job_id)
    p = _jobs_dir() / job_id / "report.json"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    rep = json.loads(p.read_text(encoding="utf-8"))
    ser, feats = rep["series"], rep["features"]
    seg_of = []
    for t in ser["t_ref"]:
        seg_of.append(next((g["id"] for g in rep["segments"] if g["ref_start"] <= t < g["ref_end"]), ""))
    labels = rep["patterns"]["labels"]
    head = ["t_ref_s", "t_user_s", "segment", "pattern"]
    for f in feats:
        u = "deg" if f["unit"] == "deg" else "torso" if f["name"] != "head_turn" else "shoulder"
        head += [f"{f['name']}_ref_{u}", f"{f['name']}_user_{u}", f"{f['name']}_delta_{u}"]
    lines = [",".join(head)]
    fmt = lambda v: "" if v is None else f"{v:.4g}"
    for i, t in enumerate(ser["t_ref"]):
        row = [fmt(t), fmt(ser["user_t"][i]), seg_of[i], "" if labels[i] < 0 else str(labels[i] + 1)]
        for f in feats:
            a, b = ser["ref"][f["name"]][i], ser["user"][f["name"]][i]
            row += [fmt(a), fmt(b), fmt(None if a is None or b is None else b - a)]
        lines.append(",".join(row))
    body = "\ufeff" + "\n".join(lines) + "\n"   # BOM so Excel opens UTF-8 correctly
    return Response(body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="motioncoach_{job_id}.csv"'})


def _bench_dir() -> Path:
    return settings.DATA_DIR / "bench"


@app.get("/api/bench")
def bench_list() -> list[dict]:
    out = []
    for p in sorted(_bench_dir().glob("*/bench.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        b = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": p.parent.name, "created": b["created"], "reference": b["reference"], "user": b["user"],
                    "backends": [r["backend"] for r in b["rows"]]})
    return out


@app.get("/api/bench/{bench_id}")
def bench_get(bench_id: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", bench_id):
        raise HTTPException(404, "not found")
    p = _bench_dir() / bench_id / "bench.json"
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="application/json")


@app.get("/media/{name}")
def media(name: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{16}_n\d+\.mp4", name):
        raise HTTPException(404, "not found")
    p = settings.DATA_DIR / "videos" / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="video/mp4")   # supports HTTP Range for seeking


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
