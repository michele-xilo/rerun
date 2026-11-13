"""
Per-archetype `construct + rr.log` cost in µs, for A/B comparison between two builds.

Run once per build (best of 5 runs, memory sink, no viewer):

    uvpy tests/python/log_benchmark/bench_all_archetypes.py --out before.json
    # switch branch, rebuild (`pixi run py-build-release`), then:
    uvpy tests/python/log_benchmark/bench_all_archetypes.py --out after.json

    uvpy tests/python/log_benchmark/bench_all_archetypes.py --compare before.json after.json

Use a release build and run both builds back to back on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

import rerun as rr

rng = np.random.default_rng(0)
img_small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
img_vga = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
depth = rng.random((120, 160), dtype=np.float32)
pts_1k = rng.random((1000, 3), dtype=np.float32)
pts_100k = rng.random((100_000, 3), dtype=np.float32)
tensor = rng.random((16, 16), dtype=np.float32)
m3 = np.eye(3, dtype=np.float32)
verts = rng.random((8, 3), dtype=np.float32)
tris = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.uint32)
strip = rng.random((10, 3), dtype=np.float32)

CASES = {
    # small, numeric
    "Scalars": lambda: rr.Scalars(42.0),
    "Transform3D t+mat": lambda: rr.Transform3D(translation=[1, 2, 3], mat3x3=m3),
    "Transform3D t+quat": lambda: rr.Transform3D(translation=[1, 2, 3], rotation=rr.Quaternion(xyzw=[0, 0, 0, 1])),
    "Transform3D t+relation": lambda: rr.Transform3D(
        translation=[1, 2, 3], relation=rr.TransformRelation.ChildFromParent
    ),
    "Points3D 1 pt": lambda: rr.Points3D([[1, 2, 3]], colors=[0xFF0000FF], radii=[0.1]),
    "Points3D 1 pt + label": lambda: rr.Points3D([[1, 2, 3]], labels=["a"]),
    "Points2D 1 pt": lambda: rr.Points2D([[1, 2]], radii=[0.1]),
    "Boxes3D": lambda: rr.Boxes3D(half_sizes=[[1, 2, 3]], colors=[0xFF0000FF]),
    "Boxes2D": lambda: rr.Boxes2D(half_sizes=[[1, 2]]),
    "Arrows3D": lambda: rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[1, 2, 3]]),
    "LineStrips3D 10 pts": lambda: rr.LineStrips3D([strip]),
    "Pinhole": lambda: rr.Pinhole(focal_length=500, width=640, height=480),
    "Clear": lambda: rr.Clear(recursive=False),
    "TextLog": lambda: rr.TextLog("hello"),
    "TextDocument": lambda: rr.TextDocument("hello"),
    "AnnotationContext": lambda: rr.AnnotationContext([(1, "a", (255, 0, 0))]),
    # medium / large payloads (zero-copy path stays above the 4 KB threshold)
    "Points3D 1k pts": lambda: rr.Points3D(pts_1k),
    "Points3D 100k pts": lambda: rr.Points3D(pts_100k),
    "Mesh3D 8 verts": lambda: rr.Mesh3D(vertex_positions=verts, triangle_indices=tris),
    "Tensor 16x16": lambda: rr.Tensor(tensor),
    "Image 8x8": lambda: rr.Image(img_small),
    "Image 640x480": lambda: rr.Image(img_vga),
    "DepthImage 160x120": lambda: rr.DepthImage(depth),
}


def bench(name: str, make: Callable[[], Any]) -> float:
    rr.log("probe", make())  # fail fast on a broken case
    t = time.perf_counter()
    for _ in range(200):
        rr.log("e", make())
    per = (time.perf_counter() - t) / 200
    n = int(min(30000, max(200, 0.4 / per)))
    if "100k" in name or "640x480" in name:
        n = min(n, 150)  # the memory sink retains everything
    best = float("inf")
    for _ in range(5):
        rr.memory_recording()
        t = time.perf_counter()
        for _ in range(n):
            rr.log("e", make())
        best = min(best, time.perf_counter() - t)
    return best / n * 1e6


def compare(before_path: str, after_path: str) -> None:
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)
    print(f"{'case':28} {'before µs':>10} {'after µs':>10} {'change':>8}")
    for name, b in before.items():
        a = after.get(name)
        if b is None or a is None:
            continue
        print(f"{name:28} {b:10.2f} {a:10.2f} {(b - a) / b * 100:+7.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="write results to this JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    rr.init("rerun_example_all_archetypes", spawn=False)
    out = {}
    for name, make in CASES.items():
        try:
            out[name] = bench(name, make)
            print(f"{name:28} {out[name]:10.2f} µs", file=sys.stderr)
        except Exception as e:
            out[name] = None
            print(f"SKIP {name}: {type(e).__name__}: {e}", file=sys.stderr)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
