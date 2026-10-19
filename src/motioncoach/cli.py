"""Command line: run an analysis headless, or start the local web service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import settings


def _analyze(args: argparse.Namespace) -> int:
    from .apps.analysis import AnalysisOptions, run_analysis
    from .apps.assets import get_reference

    ref = get_reference(args.reference, Path(args.references_dir), settings.ROOT)
    user = Path(args.user)
    if not user.exists():
        print(f"User video not found: {user}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else settings.DATA_DIR / "jobs" / f"cli_{ref.id}_{user.stem[:40]}"
    last = {}

    def progress(stage: str, frac: float) -> None:
        if last.get(stage) is None or frac - last[stage] >= 0.1 or frac >= 1.0:
            last[stage] = frac
            print(f"  {stage:<11s} {frac * 100:5.1f}%", flush=True)

    opts = _options(args)
    rep = run_analysis(ref, user, out, settings.DATA_DIR, settings.MODELS_DIR, opts, progress)
    s = rep["summary"]
    sc = rep["scope"]
    print(f"\nreport: {out / 'report.json'}  ({rep['elapsed_s']} s; {rep['timings_s']})")
    print(f"scope: {sc['reason']} -> segments {sc['segments']}, ref {sc['ref_range']} s, user "
          f"{[round(x, 1) for x in sc['user_range']]} s; backend {rep['model']['backend']} ({rep['model']['device']})")
    for c in sc["candidates"]:
        print(f"  {'✓' if c['eligible'] else '✗'} {c['name']:<16s} user {c['user_range'][0]:6.1f}-{c['user_range'][1]:6.1f}  {'; '.join(c['reasons'])}")
    print(f"pass rate: {s['pass_rate'] if s['pass_rate'] is None else round(s['pass_rate'] * 100, 1)}%"
          f"  tempo ratio: {rep['timing']['tempo_ratio']:.2f}  align cost: {rep['timing']['align_cost']:.2f}")
    if rep["left_right"]["text"]:
        print("left/right:", rep["left_right"]["text"])
    for w in rep["quality"]["warnings"]:
        print("warning:", w)
    print("segments (ref -> user seconds):")
    for seg in rep["segments"]:
        pr = "-" if seg["pass_rate"] is None else f"{seg['pass_rate'] * 100:.0f}%"
        print(f"  {seg['name']:<16s} ref {seg['ref_start']:6.1f}-{seg['ref_end']:6.1f}  user {seg['user_start']:6.1f}-{seg['user_end']:6.1f}"
              f"  pass {pr:>4s}  cost {seg['align_cost']:.2f} {' | '.join(seg['flags'])}")
    print("top issues:")
    for it in s["top_issues"]:
        print(f"  [{it['segment_name']}] {it['text']}")
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=1))
    return 0


def _options(args: argparse.Namespace):
    from .apps.analysis import AnalysisOptions
    from .apps.scope import ScopeRequest

    scope = ScopeRequest(args.scope, args.seg_from, args.seg_to, args.user_start, args.user_end)
    return AnalysisOptions(sample_fps=args.fps, backend=args.backend, user_mirrored=args.mirrored,
                           mirror_practice=args.mirror_practice, scope=scope)


def _bench(args: argparse.Namespace) -> int:
    from .apps.assets import get_reference
    from .apps.bench import run_bench

    ref = get_reference(args.reference, Path(args.references_dir), settings.ROOT)
    out = Path(args.out) if args.out else settings.DATA_DIR / "bench" / time.strftime("%Y%m%d-%H%M%S")
    res = run_bench(ref, Path(args.user), args.backends.split(","), out, settings.DATA_DIR, settings.MODELS_DIR,
                    _options(args), progress=lambda s, f: print(f"  {s:<28s} {f * 100:5.1f}%", flush=True) if s.endswith(("done", "align")) else None)
    print((out / "bench.md").read_text(encoding="utf-8"))
    return 0


def _backends(args: argparse.Namespace) -> int:
    from .perception.registry import list_backends

    for b in list_backends():
        mark = "✓" if b["available"] else ("○" if b["status"] == "candidate" else "✗")
        print(f"{mark} {b['name']:<16s} {b['dims']:<6s} {b['keypoints']:>3d} kp  {b['license']:<12s} {b['label']}"
              f"{'  [默认]' if b['default'] else ''}")
        print(f"      GPU: {b['gpu']}")
        if b["reason"]:
            print(f"      {b['reason']}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("motioncoach.service.app:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="motioncoach")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="compare a user video against a reference asset")
    a.add_argument("--reference", required=True, help="reference id (references/*.yaml)")
    a.add_argument("--references-dir", default=str(settings.REFERENCES_DIR))
    a.add_argument("--user", required=True, help="user video path")
    a.add_argument("--out", help="output directory (default data/jobs/cli_*)")
    a.add_argument("--fps", type=float, default=10.0, help="analysis sample rate")
    a.add_argument("--backend", default="mediapipe-full", help="pose backend (see `motioncoach backends`)")
    a.add_argument("--mirrored", default="auto", choices=["auto", "yes", "no"], help="user video is a selfie mirror")
    a.add_argument("--mirror-practice", action="store_true", help="user performs the opposite side on purpose")
    a.add_argument("--json", action="store_true", help="also print the summary JSON")
    a.set_defaults(func=_analyze)
    b = sub.add_parser("bench", help="compare pose backends on the same interval")
    b.add_argument("--reference", required=True)
    b.add_argument("--references-dir", default=str(settings.REFERENCES_DIR))
    b.add_argument("--user", required=True)
    b.add_argument("--backends", default="mediapipe-full,rtmw-l,yolo26m-pose")
    b.add_argument("--out")
    b.add_argument("--fps", type=float, default=10.0)
    b.add_argument("--backend", default="mediapipe-full", help=argparse.SUPPRESS)
    b.add_argument("--mirrored", default="auto", choices=["auto", "yes", "no"])
    b.add_argument("--mirror-practice", action="store_true")
    b.set_defaults(func=_bench)
    for sp in (a, b):
        sp.add_argument("--scope", default="auto", choices=["auto", "segments", "manual"],
                        help="auto: longest shared, continuous, quality-checked interval")
        sp.add_argument("--from", dest="seg_from", help="first segment id (segments/manual)")
        sp.add_argument("--to", dest="seg_to", help="last segment id (segments/manual)")
        sp.add_argument("--user-start", type=float, help="manual: start in the user video (s)")
        sp.add_argument("--user-end", type=float, help="manual: end in the user video (s)")
    lb = sub.add_parser("backends", help="list pose backends and availability")
    lb.set_defaults(func=_backends)
    s = sub.add_parser("serve", help="start the local web service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=_serve)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
