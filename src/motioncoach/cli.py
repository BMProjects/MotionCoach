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

    from .engine.errors import NoPersonError

    opts = _options(args)
    try:
        rep = run_analysis(ref, user, out, settings.DATA_DIR, settings.MODELS_DIR, opts, progress)
    except NoPersonError as exc:   # an outcome, not a crash
        print(f"\n无法分析：{exc}", file=sys.stderr)
        return 3
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


def _group(args: argparse.Namespace) -> int:
    from .apps.assets import get_reference
    from .apps.group import run_group
    from .engine.group import GroupParams

    ref = get_reference(args.reference, Path(args.references_dir), settings.ROOT)
    out = Path(args.out) if args.out else settings.DATA_DIR / "group" / time.strftime("%Y%m%d-%H%M%S")
    ids = lambda s: tuple(int(x) for x in s.split(",") if x != "") if s else ()   # noqa: E731
    params = GroupParams(baseline_s=args.baseline, max_persons=args.max_persons, max_depth_ratio=args.max_depth_ratio,
                         include=ids(args.include), exclude=ids(args.exclude))
    from .apps.analysis import AnalysisOptions
    opts = AnalysisOptions(sample_fps=args.fps, backend=args.backend, user_mirrored=args.mirrored)
    g = run_group(ref, Path(args.video), out, settings.DATA_DIR, settings.MODELS_DIR, opts, params,
                  progress=lambda s, f: None)
    print(f"{len(g['identities'])} 个身份，练习者 {sum(i['performer'] for i in g['identities'])} 人 → {out / 'group.json'}")
    for i in g["identities"]:
        sc = i["screening"]
        print(f"  #{i['id']:<2d} {'练习者' if i['performer'] else '排除  '}  身高 {i['stature_px']:6.0f}px  "
              f"x {i['layout'].get('x_m')} m  z {i['layout'].get('z_m')} m  身高约 {i['layout'].get('height_m')} m  "
              f"出现 {sc['presence'] * 100:3.0f}%  "
              f"运动 {sc['motion_rel']}  相似 {sc['similarity']}  {'；'.join(sc['reasons'])}")
    for p in g["persons"]:
        sy = g["group"].get("sync", {}).get(p["id"], {})
        print(f"  #{p['id']:<2d} {p.get('pass_text') or p.get('error')}  同步偏移中位 {sy.get('offset_median_s')} s")
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
    if args.probe:   # run every installed backend on a few seconds: the device actually used and its speed
        from .media.extract import prepared_video
        from .perception.base import run_on_video
        from .perception.registry import get_backend

        _, proxy = prepared_video(Path(args.probe), settings.DATA_DIR)
        print(f"\n实测（{Path(args.probe).name} 前 {args.seconds:g} 秒，10 fps）")
        for b in list_backends():
            if not b["available"]:
                continue
            try:
                be = get_backend(b["name"], settings.MODELS_DIR)
                seq = run_on_video(be, proxy, 10.0, 0.0, args.seconds)
                det = float(seq.detected.mean())
                print(f"  {b['name']:<16s} device={seq.meta.get('device')}  {seq.meta.get('ms_per_frame', 0):6.1f} ms/帧  "
                      f"检出 {det:.0%}  {seq.meta.get('device_note', '')}")
            except Exception as exc:   # report and continue with the next backend
                print(f"  {b['name']:<16s} 失败：{type(exc).__name__}: {str(exc)[:160]}")
    return 0


def _refpack(args: argparse.Namespace) -> int:
    from .apps.assets import get_reference
    from .apps.refpack import build_refpack

    ref = get_reference(args.reference, Path(args.references_dir), settings.ROOT)
    out = Path(args.out) if args.out else settings.DATA_DIR / "refpacks" / f"{ref.id}__{args.backend}"
    m = build_refpack(ref, out, settings.DATA_DIR, settings.MODELS_DIR, args.backend, args.fps, args.with_video)
    size = sum(f["bytes"] for f in m["files"].values())
    print(f"{out}  ({m['frames']} frames, {len(m['segments'])} segments, {size / 1e6:.2f} MB, version {m['version']})")
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
    g = sub.add_parser("group", help="multi-person video: identities, performer screening, per-person reports, group metrics")
    g.add_argument("--reference", required=True)
    g.add_argument("--references-dir", default=str(settings.REFERENCES_DIR))
    g.add_argument("--video", required=True)
    g.add_argument("--backend", default="rtmw-l", help="a multi-person backend (rtmlib or YOLO)")
    g.add_argument("--fps", type=float, default=10.0)
    g.add_argument("--mirrored", default="auto", choices=["auto", "yes", "no"])
    g.add_argument("--baseline", type=float, default=10.0, help="opening seconds that set up the identities")
    g.add_argument("--max-persons", type=int, default=12)
    g.add_argument("--max-depth-ratio", type=float, default=1.8, help="farther than this x the group's depth = behind the group")
    g.add_argument("--include", default="", help="identity ids forced to be performers, e.g. 3,5")
    g.add_argument("--exclude", default="", help="identity ids forced out")
    g.add_argument("--out")
    g.set_defaults(func=_group)
    lb = sub.add_parser("backends", help="list pose backends and availability")
    lb.add_argument("--probe", metavar="VIDEO", help="also run every installed backend on the first seconds of VIDEO")
    lb.add_argument("--seconds", type=float, default=5.0)
    lb.set_defaults(func=_backends)
    rp = sub.add_parser("refpack", help="build a reference package (role A -> mobile clients)")
    rp.add_argument("--reference", required=True)
    rp.add_argument("--references-dir", default=str(settings.REFERENCES_DIR))
    rp.add_argument("--backend", default="mediapipe-full", help="pose model the client will use")
    rp.add_argument("--fps", type=float, default=10.0)
    rp.add_argument("--with-video", action="store_true", help="include the reference display proxy")
    rp.add_argument("--out")
    rp.set_defaults(func=_refpack)
    s = sub.add_parser("serve", help="start the local web service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=_serve)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
