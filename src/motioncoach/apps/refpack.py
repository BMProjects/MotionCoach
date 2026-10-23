"""Reference package (role A -> role C): everything a client needs to compare against one reference
video without the server - rules, annotated repetition templates, and the reference's features and
same-body skeleton measured with one named pose model (contracts/schemas/refpack_manifest.schema.json).

    motioncoach refpack --reference baduanjin_demo_a [--backend mediapipe-full] [--with-video]
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..contracts import REFPACK_SCHEMA
from ..engine.features import compute_features
from ..engine.filters import one_euro_clean
from ..engine.normalize import CANON, bone_lengths, normalized
from ..media.extract import extract_poses, prepared_video
from ..media.ingest import view_path
from ..perception.registry import DEFAULT_BACKEND, get_backend
from .assets import PROFILES_DIR, Reference, load_profile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_refpack(reference: Reference, out_dir: Path, data_dir: Path, models_dir: Path,
                  backend: str = DEFAULT_BACKEND, sample_fps: float = 10.0, with_video: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(reference.raw["profile"])
    names = profile.features
    be = get_backend(backend, models_dir)
    vid, proxy = prepared_video(reference.video, data_dir)
    raw = extract_poses(proxy, vid, be, sample_fps, data_dir)
    seq = raw.unmirror() if reference.raw.get("mirrored_video") else raw
    seq = one_euro_clean(seq, conf_thr=seq.meta.get("conf_thr", 0.5))     # same cleaning as the analysis
    F, _, _ = compute_features(seq, names)
    L = bone_lengths(seq)
    N = normalized(seq, L, L, L[("hip_mid", "shoulder_mid")])

    files: dict[str, dict] = {}

    def put(key: str, name: str, arr: np.ndarray | None = None, src: Path | None = None) -> None:
        p = out_dir / name
        if arr is not None:
            arr.tofile(p)
        else:
            shutil.copyfile(src, p)
        entry = {"path": name, "sha256": _sha(p), "bytes": p.stat().st_size}
        if arr is not None:
            entry.update(dtype=str(arr.dtype), shape=list(arr.shape))
        files[key] = entry

    model = be.model_id
    tag = backend.replace("/", "_")
    put("t", "t.f32", np.asarray(seq.t, np.float32))
    put("features", f"features_{tag}.f16", np.asarray(F, np.float16))                  # NaN = not measurable
    put("skeleton", f"skeleton_{tag}.f16", np.asarray(N, np.float16))
    put("profile", "profile.yaml", src=PROFILES_DIR / profile.id / "profile.yaml")
    if with_video:
        put("video", "reference_view.mp4", src=view_path(proxy))

    templates = reference.rep_templates()
    segments = [{"id": s.id, "name": s.name, "start": s.start, "end": s.end, "reps": templates.get(s.id)}
                for s in reference.segments(profile)]
    content = json.dumps([reference.raw, profile.raw, model, sample_fps], sort_keys=True, ensure_ascii=False, default=str)
    manifest = {
        "schema": REFPACK_SCHEMA, "reference_id": reference.id, "title": reference.raw.get("title"),
        "version": hashlib.sha256(content.encode()).hexdigest()[:16], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine_version": __version__, "profile": profile.id, "status": reference.raw.get("status"),
        "model": model, "sample_fps": sample_fps, "frames": int(len(seq)), "features": list(names),
        "joints": list(CANON), "segments": segments, "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest
