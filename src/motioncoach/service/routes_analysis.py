"""Role A - high-precision analysis service: references, uploads, analysis jobs, reports, media.

See docs/MOBILE_DESIGN.md (roles A/B/C). The mobile side (role C) will consume the same contracts.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .. import settings
from ..apps.assets import get_reference, list_references, load_profile
from ..perception.registry import DEFAULT_BACKEND, GROUP_BACKEND, SPECS, availability
from .jobs import (MAX_UPLOAD_BYTES, VIDEO_EXT, GroupOptions, IdentityEdit, JobRequest, _executor, _job_summary, _jobs,
                   _jobs_dir, _load_job, _lock, _resolve_user_video, _run, _save_job)

router = APIRouter()


@router.get("/api/references")
def references() -> list[dict]:
    out = []
    for r in list_references(settings.REFERENCES_DIR, settings.ROOT):
        prof = load_profile(r.raw["profile"])
        out.append({"id": r.id, "title": r.raw.get("title"), "status": r.raw.get("status"),
                    "profile": prof.id, "profile_title": prof.raw["title"], "view": r.raw.get("view"),
                    "available": r.video.exists(),
                    "segments": [{"id": s.id, "name": s.name, "start": s.start, "end": s.end} for s in r.segments(prof)]})
    return out


@router.get("/api/local-videos")
def local_videos() -> list[dict]:
    """Development helper: videos already in assets/."""
    return [{"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)}
            for p in sorted(settings.ASSETS_DIR.iterdir()) if p.suffix.lower() in VIDEO_EXT]


@router.post("/api/uploads")
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


@router.post("/api/jobs")
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
    if req.mode == "group" and SPECS[req.backend].family == "mediapipe":   # MediaPipe gives no per-person boxes
        if not availability(GROUP_BACKEND)[0]:
            raise HTTPException(400, "多人分析需要 RTMPose/RTMW 或 YOLO 姿态算法（uv sync --extra rtm）")
        req.backend = GROUP_BACKEND
    user_path = _resolve_user_video(req.user)
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0, "created": time.time(),
           "request": req.model_dump(), "user_filename": req.user.local or req.user.filename or req.user.upload_id}
    with _lock:
        _jobs[job_id] = job
        _save_job(job)
    _executor.submit(_run, job_id, req, user_path)
    return job



@router.get("/api/jobs")
def list_jobs(limit: int = 200) -> list[dict]:
    jobs = []
    for p in sorted(_jobs_dir().glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        j = json.loads(p.read_text(encoding="utf-8"))
        with _lock:
            j = dict(_jobs.get(j["id"], j))
        j.pop("traceback", None)
        if j.get("status") == "done":
            j["summary"] = _job_summary(p.parent)
        jobs.append(j)
    return jobs


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _load_job(job_id)


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Delete one analysis (job record, report, poses) and its uploaded video when no other analysis uses it.

    Shared caches (pose cache, browser-compatible video copies) stay: later analyses reuse them.
    A queued/running job is refused unless this process no longer knows it (left over from a restart).
    """
    job = _load_job(job_id)
    with _lock:
        live = job_id in _jobs
        if live and job.get("status") in ("queued", "running"):
            raise HTTPException(409, "分析进行中，完成后才能删除")
        _jobs.pop(job_id, None)
    shutil.rmtree(_jobs_dir() / job_id)
    upload_id = (job.get("request") or {}).get("user", {}).get("upload_id")
    if upload_id and re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{2,5}", upload_id):
        still_used = any(upload_id in q.read_text(encoding="utf-8") for q in _jobs_dir().glob("*/job.json"))
        if not still_used:
            (settings.DATA_DIR / "uploads" / upload_id).unlink(missing_ok=True)
    return {"deleted": job_id}


def _result_dir(job_id: str, person: int | None) -> Path:
    """A single analysis keeps its report in the job folder; in a group job each performer has a sub-folder."""
    _load_job(job_id)
    return _jobs_dir() / job_id / (f"person_{int(person)}" if person is not None else "")


def _gzip_json(p: Path, request: Request) -> FileResponse:
    if "gzip" not in request.headers.get("accept-encoding", ""):
        return FileResponse(p, media_type="application/json")
    gz = p.with_suffix(".json.gz")
    if not gz.exists() or gz.stat().st_mtime < p.stat().st_mtime:
        tmp = gz.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(p.read_bytes(), 6))
        tmp.replace(gz)
    return FileResponse(gz, media_type="application/json", headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"})


@router.get("/api/jobs/{job_id}/group")
def get_group(job_id: str, request: Request) -> FileResponse:
    p = _result_dir(job_id, None) / "group.json"
    if not p.exists():
        raise HTTPException(404, "group result not ready")
    return _gzip_json(p, request)


@router.post("/api/jobs/{job_id}/identities")
def edit_identities(job_id: str, edits: list[IdentityEdit]) -> dict:
    """Append identity edits (merge / split / role) to a finished group job and re-run it in place; the pose
    cache makes the re-run take seconds to a minute."""
    job = _load_job(job_id)
    req = JobRequest(**job["request"])
    if req.mode != "group":
        raise HTTPException(400, "not a multi-person analysis")
    if job.get("status") in ("queued", "running"):
        raise HTTPException(409, "分析进行中")
    g = req.group or GroupOptions()
    g.edits = [*g.edits, *edits]
    req.group = g
    job.update(status="queued", stage="queued", progress=0.0, request=req.model_dump(by_alias=True),
               rerun=time.time(), error=None)
    with _lock:
        _jobs[job_id] = job
        _save_job(job)
    (_jobs_dir() / job_id / "summary.json").unlink(missing_ok=True)
    _executor.submit(_run, job_id, req, _resolve_user_video(req.user))
    return job


class SectionFix(BaseModel):
    sections: dict[str, tuple[float, float]] | None = None    # None: back to the automatic vote


@router.post("/api/jobs/{job_id}/sections")
def fix_sections(job_id: str, fix: SectionFix) -> dict:
    """"修正动作段" for a group job: fix (or clear) the common section bounds and re-run in place."""
    job = _load_job(job_id)
    req = JobRequest(**job["request"])
    if req.mode != "group":
        raise HTTPException(400, "not a multi-person analysis; single analyses are corrected with a new manual-scope job")
    if job.get("status") in ("queued", "running"):
        raise HTTPException(409, "分析进行中")
    for k, (a, b) in (fix.sections or {}).items():
        if not (b > a >= 0):
            raise HTTPException(400, f"{k}: 结束时间应晚于开始时间")
    g = req.group or GroupOptions()
    g.sections = fix.sections or None
    req.group = g
    job.update(status="queued", stage="queued", progress=0.0, request=req.model_dump(by_alias=True), rerun=time.time(), error=None)
    with _lock:
        _jobs[job_id] = job
        _save_job(job)
    (_jobs_dir() / job_id / "summary.json").unlink(missing_ok=True)
    _executor.submit(_run, job_id, req, _resolve_user_video(req.user))
    return job


@router.get("/api/jobs/{job_id}/report")
def get_report(job_id: str, request: Request, person: int | None = None) -> FileResponse:
    """The report is a few MB of JSON; serve a gzip copy (made once, next to it) when the client accepts it."""
    p = _result_dir(job_id, person) / "report.json"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    return _gzip_json(p, request)


@router.get("/api/jobs/{job_id}/poses/{side}")
def get_poses(job_id: str, side: str, person: int | None = None) -> FileResponse:
    if side not in ("ref", "user"):
        raise HTTPException(404, "unknown side")
    p = _result_dir(job_id, person) / f"poses_{side}.json.gz"
    if not p.exists():
        raise HTTPException(404, "poses not ready")
    return FileResponse(p, media_type="application/json", headers={"Content-Encoding": "gzip"})



@router.get("/api/jobs/{job_id}/export.csv")
def export_csv(job_id: str, person: int | None = None) -> Response:
    """Per-moment aligned data for researchers: reference time, matched user time, segment, values."""
    p = _result_dir(job_id, person) / "report.json"
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
                    headers={"Content-Disposition": f'attachment; filename="motioncoach_{job_id}{"" if person is None else f"_p{person}"}.csv"'})



@router.get("/media/{name}")
def media(name: str) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{16}_n\d+(_view)?\.mp4", name):   # n1 single proxy; n2 display proxy (_view)
        raise HTTPException(404, "not found")
    p = settings.DATA_DIR / "videos" / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="video/mp4")   # supports HTTP Range for seeking

