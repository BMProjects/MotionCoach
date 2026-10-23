"""FastAPI application: assembles the role routers (A analysis, B dev tools) and the web client.

Roles are logical (docs/MOBILE_DESIGN.md): both run in one process today; the routers can be served
separately later without changing their contracts.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .. import settings
from ..contracts import REPORT_SCHEMA
from ..engine.scale import DEFAULT_HEIGHT_CM
from ..perception.registry import BOUNDS_BACKEND, DEFAULT_BACKEND, SPECS
from . import routes_analysis, routes_devtools

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="motioncoach")
app.include_router(routes_analysis.router)
app.include_router(routes_devtools.router)


@app.middleware("http")
async def revalidate_web_client(request: Request, call_next):
    """The page and its scripts must not be reused from a browser cache after a deploy: without a
    Cache-Control header browsers keep them for hours on heuristics (docs/ITERATION_LOG.md I19)."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/config")
def config() -> dict:
    return {"dev_mode": settings.DEV_MODE, "default_backend": DEFAULT_BACKEND,
            "default_backend_label": SPECS[DEFAULT_BACKEND].label, "bounds_backend": BOUNDS_BACKEND, "default_height_cm": DEFAULT_HEIGHT_CM,
            "report_schema": REPORT_SCHEMA}


def _versioned(match: re.Match) -> str:
    path = WEB_DIR / match.group(2)
    tag = hashlib.sha1(path.read_bytes()).hexdigest()[:10] if path.is_file() else "0"
    return f'{match.group(1)}/static/{match.group(2)}?v={tag}"'


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """index.html with a content hash on each script and stylesheet URL, so a deploy is picked up at once."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(re.sub(r'((?:src|href)=")/static/([\w./-]+)"', _versioned, html))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
