"""Role B - algorithm testing service: pose backends and benchmark results (developer instance).

The heavier studies (research.align_compare, research.pose_bench) run as commands on the GPU server;
these routes only expose what the developer web pages show. See docs/MOBILE_DESIGN.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .. import settings
from ..perception.registry import list_backends

router = APIRouter()


@router.get("/api/backends")
def backends() -> list[dict]:
    return list_backends()



def _bench_dir() -> Path:
    return settings.DATA_DIR / "bench"


@router.get("/api/bench")
def bench_list() -> list[dict]:
    out = []
    for p in sorted(_bench_dir().glob("*/bench.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        b = json.loads(p.read_text(encoding="utf-8"))
        out.append({"id": p.parent.name, "created": b["created"], "reference": b["reference"], "user": b["user"],
                    "backends": [r["backend"] for r in b["rows"]]})
    return out


@router.get("/api/bench/{bench_id}")
def bench_get(bench_id: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", bench_id):
        raise HTTPException(404, "not found")
    p = _bench_dir() / bench_id / "bench.json"
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="application/json")

