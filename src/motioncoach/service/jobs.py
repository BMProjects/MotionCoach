"""Analysis jobs: request model, filesystem-backed records, a thread-pool executor, and history digests.

Shared by the API routes; one process, ``MOTIONCOACH_JOB_WORKERS`` analysis threads, storage under data/.
A production deployment keeps this interface and swaps the executor (queue + GPU workers) and storage.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .. import settings
from ..apps.analysis import AnalysisOptions, run_analysis
from ..apps.group import run_group
from ..apps.assets import get_reference
from ..apps.scope import ScopeRequest
from ..engine.errors import NoPersonError
from ..engine.group import GroupParams
from ..media.presence import require_person
from ..perception.registry import DEFAULT_BACKEND, LOCATE_BACKEND, get_backend

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
MAX_UPLOAD_BYTES = 2 * 1024 ** 3
STAGE_WEIGHTS = {"queued": 0, "check": 0.03, "ingest": 0.07, "locate": 0.6, "poses_ref": 0.03, "poses_user": 0.03, "features": 0.04,
                 "align": 0.08, "compare": 0.12}   # locate = the one full-length pose pass (fine passes then hit the cache)
GROUP_STAGE_WEIGHTS = {"queued": 0, "check": 0.03, "ingest": 0.07, "poses": 0.55, "identities": 0.05, "persons": 0.30}

_executor = ThreadPoolExecutor(max_workers=settings.JOB_WORKERS)
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
    filename: str | None = Field(None, max_length=200)   # original name of an upload, for display only


class IdentityEdit(BaseModel):
    model_config = ConfigDict(populate_by_name=True)   # "from" on the wire, from_ when re-read from job.json
    op: str = Field(pattern="^(merge|split|role)$")
    id: int | None = Field(None, ge=0)
    into: int | None = Field(None, ge=0)
    from_: int | None = Field(None, ge=0, alias="from")
    at: float | None = Field(None, ge=0)
    performer: bool | None = None

    def as_dict(self) -> dict:
        d = {"op": self.op, "id": self.id, "into": self.into, "from": self.from_, "at": self.at, "performer": self.performer}
        return {k: v for k, v in d.items() if v is not None}


class GroupOptions(BaseModel):
    baseline_s: float = Field(10.0, ge=2, le=60)
    max_persons: int = Field(12, ge=1, le=30)
    edits: list[IdentityEdit] = Field(default_factory=list, max_length=200)
    sections: dict[str, tuple[float, float]] | None = None   # "修正动作段": section id -> (start, end) seconds


class JobRequest(BaseModel):
    reference_id: str
    user: UserVideo
    mode: str = Field("single", pattern="^(single|group)$")    # group: several people in one video (apps/group.py)
    group: GroupOptions | None = None
    sample_fps: float = Field(10.0, ge=2, le=30)
    backend: str = DEFAULT_BACKEND
    user_mirrored: str = Field("auto", pattern="^(auto|yes|no)$")
    mirror_practice: bool = False
    scope_mode: str = Field("auto", pattern="^(auto|segments|manual)$")
    seg_from: str | None = None
    seg_to: str | None = None
    user_start: float | None = Field(None, ge=0)
    user_end: float | None = Field(None, ge=0)
    height_cm: float | None = Field(None, ge=80, le=230)   # practitioner's own height: enables centimetre wording


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
    weights = GROUP_STAGE_WEIGHTS if req.mode == "group" else STAGE_WEIGHTS
    order = list(weights)

    def progress(stage: str, frac: float) -> None:
        with _lock:
            job = _jobs[job_id]
            if stage == "done":
                return
            done = sum(weights[s] for s in order[: order.index(stage)]) if stage in order else 0
            job.update(stage=stage, progress=round(done + weights.get(stage, 0) * frac, 3))

    try:
        with _lock:
            _jobs[job_id]["status"] = "running"
        ref = get_reference(req.reference_id, settings.REFERENCES_DIR, settings.ROOT)
        opts = AnalysisOptions(sample_fps=req.sample_fps, backend=req.backend,
                               user_mirrored=req.user_mirrored, mirror_practice=req.mirror_practice,
                               scope=ScopeRequest(req.scope_mode, req.seg_from, req.seg_to, req.user_start, req.user_end),
                               height_cm=req.height_cm)
        if req.mode == "group":
            g = req.group or GroupOptions()
            params = GroupParams(baseline_s=g.baseline_s, max_persons=g.max_persons,
                                 edits=tuple(e.as_dict() for e in g.edits), sections=g.sections)
            progress("check", 0.0)
            require_person(user_path, get_backend(LOCATE_BACKEND, settings.MODELS_DIR))
            run_group(ref, user_path, _jobs_dir() / job_id, settings.DATA_DIR, settings.MODELS_DIR, opts, params, progress)
        else:
            run_analysis(ref, user_path, _jobs_dir() / job_id, settings.DATA_DIR, settings.MODELS_DIR, opts, progress)
        with _lock:
            _jobs[job_id].update(status="done", stage="done", progress=1.0, finished=time.time())
    except NoPersonError as exc:   # an outcome for the user (e.g. an empty video), not a crash
        with _lock:
            _jobs[job_id].update(status="rejected", stage="check", reason=str(exc), presence=exc.presence,
                                 finished=time.time())
    except Exception as exc:  # report to the client; keep the traceback in job.json
        with _lock:
            _jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}",
                                 traceback=traceback.format_exc(limit=8))
    finally:
        with _lock:
            _save_job(_jobs[job_id])



def _group_summary(job_dir: Path) -> dict | None:
    g_p, sum_p = job_dir / "group.json", job_dir / "summary.json"
    if sum_p.exists() and sum_p.stat().st_mtime >= g_p.stat().st_mtime:
        return json.loads(sum_p.read_text(encoding="utf-8"))
    try:
        g = json.loads(g_p.read_text(encoding="utf-8"))
        done = [p for p in g["persons"] if p.get("status") == "done"]
        rates = []
        for p in done:
            c = p.get("counts") or {}
            ev = c.get("good", 0) + c.get("warn", 0) + c.get("bad", 0)
            if ev:
                rates.append(c.get("good", 0) / ev)
        summ = {"schema": g.get("schema"), "mode": "group", "reference": g["reference"].get("title"),
                "segments": f"多人 · 识别 {len(g['identities'])} 人，练习者 {sum(i['performer'] for i in g['identities'])} 人",
                "pass_text": (f"{len(done)} 位练习者，重点指标平均达标 {round(100 * sum(rates) / len(rates))}%" if rates else "没有完成比较的练习者"),
                "backend": g.get("model", {}).get("backend"), "created": g.get("created")}
    except (ValueError, KeyError, ZeroDivisionError):
        return None
    sum_p.write_text(json.dumps(summ, ensure_ascii=False), encoding="utf-8")
    return summ


def _job_summary(job_dir: Path) -> dict | None:
    """Small digest of a finished report for the history page; cached next to report.json."""
    if (job_dir / "group.json").exists():
        return _group_summary(job_dir)
    rep_p, sum_p = job_dir / "report.json", job_dir / "summary.json"
    if not rep_p.exists():
        return None
    if sum_p.exists() and sum_p.stat().st_mtime >= rep_p.stat().st_mtime:
        return json.loads(sum_p.read_text(encoding="utf-8"))
    try:
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        names = {sg["id"]: sg["name"] for sg in rep.get("segments", [])}
        sc = rep.get("scope", {})
        segs = [names.get(i, i) for i in sc.get("segments", [])]
        summ = {
            "schema": rep.get("schema"),
            "title": rep.get("profile", {}).get("title"),
            "reference": rep.get("reference", {}).get("title"),
            "segments": f"{segs[0]} — {segs[-1]}" if len(segs) > 1 else (segs[0] if segs else ""),
            "ref_range": sc.get("ref_range"), "user_range": sc.get("user_range"),
            "pass_text": rep.get("summary", {}).get("pass_text"),
            "top_issue": (rep.get("summary", {}).get("top_issues") or [{}])[0].get("text"),
            "tempo_ratio": rep.get("timing", {}).get("tempo_ratio"),
            "backend": rep.get("model", {}).get("backend"), "device": rep.get("model", {}).get("device"),
            "created": rep.get("created"),
        }
    except (ValueError, KeyError, IndexError):
        return None
    sum_p.write_text(json.dumps(summ, ensure_ascii=False), encoding="utf-8")
    return summ
