"""Distance from the embryo surface, quantised and stored as its own multiscale group.

The distance transform must be given physical spacing. At level 1 the voxel is
(2.0, 0.52, 0.52) um, so omitting ``sampling`` would report depth in voxels and be wrong
along z by a factor of ~4 -- and since depth is the axis the whole tool samples on, that
error would be invisible and everywhere.

Stored as ``uint8`` at 0.5 um per unit. The field is smooth and bounded, and the mask it
comes from was segmented at level 4, so its surface is only accurate to ~4 um; 0.5 um is
already ~8x finer than that, and float32 would spend 4 bytes a voxel storing noise.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .store import (
    DEFAULT_STORE,
    DEPTH_LEVELS,
    DEPTH_SCALE_UM,
    Dataset,
    Level,
    quantize_depth,
)

#: How the six faces of the imaged volume are treated by the distance transform.
#:
#: "ignore"  -- the crop plane is not a surface. A voxel cut off by the edge of the
#:              imaged volume gets its distance to the nearest *real* tissue boundary,
#:              measured laterally. Nothing fake is introduced, but tissue that is in
#:              truth just below an unimaged surface reads as deeper than it is.
#: "surface" -- the crop plane counts as surface, so depth falls to 0 there. This invents
#:              a flat shell that would appear as a disc in every shallow-depth view.
BOUNDARY_MODES = ("ignore", "surface")


def surface_distance_um(
    mask: np.ndarray,
    voxel_um: tuple[float, float, float],
    boundary: str = "ignore",
) -> np.ndarray:
    """Euclidean distance in micrometres from each body voxel to the embryo surface."""
    if boundary not in BOUNDARY_MODES:
        raise ValueError(f"unknown boundary mode {boundary!r}, expected one of {BOUNDARY_MODES}")
    if boundary == "surface":
        # One layer of background on every face makes the crop plane a surface.
        padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
        dist = ndi.distance_transform_edt(padded, sampling=voxel_um)
        return dist[1:-1, 1:-1, 1:-1].astype(np.float32)
    # No padding: scipy measures to the nearest zero *inside* the array, so a body running
    # off the edge is not treated as having a surface there.
    return ndi.distance_transform_edt(mask.astype(bool), sampling=voxel_um).astype(np.float32)


def touched_faces(mask: np.ndarray) -> dict[str, int]:
    """Which faces of the imaged volume the body runs into, and by how many voxels."""
    faces = {
        "z0": mask[0], "z1": mask[-1],
        "y0": mask[:, 0], "y1": mask[:, -1],
        "x0": mask[:, :, 0], "x1": mask[:, :, -1],
    }
    return {k: int(v.sum()) for k, v in faces.items() if v.any()}


def crop_bounded(
    depth_um: np.ndarray, level: Level, faces: dict[str, int]
) -> np.ndarray:
    """Voxels whose depth is only an upper bound, because the crop plane is nearer.

    Derived, not stored: it is a pure function of the depth field, the voxel geometry and
    which faces the body touches, so the viewer can recompute it for free rather than
    carrying another 32 GiB array. Where this is true the tissue may sit just under a
    surface that was never imaged, and its reported depth is too large.
    """
    zs, ys, xs = level.voxel_size_um
    nz, ny, nx = depth_um.shape
    limit = np.full(depth_um.shape, np.inf, dtype=np.float32)
    axes = {
        "z0": (np.arange(nz) * zs, 0), "z1": ((nz - 1 - np.arange(nz)) * zs, 0),
        "y0": (np.arange(ny) * ys, 1), "y1": ((ny - 1 - np.arange(ny)) * ys, 1),
        "x0": (np.arange(nx) * xs, 2), "x1": ((nx - 1 - np.arange(nx)) * xs, 2),
    }
    for face in faces:
        coords, axis = axes[face]
        shape = [1, 1, 1]
        shape[axis] = -1
        limit = np.minimum(limit, coords.reshape(shape).astype(np.float32))
    return (depth_um > 0) & (limit < depth_um)


def _worker(args):
    t, store_path, level, boundary = args
    ds = Dataset(store_path)
    lv = ds.levels[level]
    mask = np.asarray(ds.label(level)[t, 0]).astype(bool)
    depth = surface_distance_um(mask, lv.voxel_size_um, boundary)
    ds.depth(level, mode="a")[t, 0] = quantize_depth(depth)
    faces = touched_faces(mask)
    bounded = int(crop_bounded(depth, lv, faces).sum()) if faces else 0
    return t, {
        "max_depth_um": float(depth.max()),
        "body_voxels": int(mask.sum()),
        "saturated_voxels": int((depth / DEPTH_SCALE_UM > 255).sum()),
        "touched_faces": faces,
        "crop_bounded_voxels": bounded,
        "crop_bounded_fraction": bounded / max(int(mask.sum()), 1),
    }


def build_depth(
    store_path: Path,
    levels: list[str] = list(DEPTH_LEVELS),
    boundary: str = "ignore",
    workers: int = 4,
) -> dict[str, list[dict]]:
    ds = Dataset(store_path)
    ds.create_depth_group(levels)
    out: dict[str, list[dict]] = {}
    for level in sorted(levels, key=int, reverse=True):
        lv = ds.levels[level]
        n_t = ds.n_timepoints
        print(f"level {level} {lv.shape} voxel={lv.voxel_size_um} um, boundary={boundary}")
        jobs = [(t, store_path, level, boundary) for t in range(n_t)]
        started = time.time()
        stats: list[dict | None] = [None] * n_t
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, (t, st) in enumerate(pool.map(_worker, jobs), start=1):
                stats[t] = st
                if done % 10 == 0 or done == n_t:
                    rate = done / (time.time() - started)
                    print(
                        f"  {done:3d}/{n_t} ({rate:.2f}/s, "
                        f"{(n_t - done) / rate / 60:.1f} min left)",
                        flush=True,
                    )
        out[level] = stats  # type: ignore[assignment]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--levels", nargs="+", default=list(DEPTH_LEVELS))
    p.add_argument("--boundary", default="ignore", choices=BOUNDARY_MODES)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    stats = build_depth(args.store, args.levels, args.boundary, args.workers)
    for level, st in stats.items():
        depths = np.array([s["max_depth_um"] for s in st])
        bounded = np.array([s["crop_bounded_fraction"] for s in st])
        sat = sum(s["saturated_voxels"] for s in st)
        print(
            f"level {level}: max depth {depths.min():.1f}..{depths.max():.1f} um, "
            f"saturated voxels {sat}, crop-bounded {bounded.min() * 100:.1f}.."
            f"{bounded.max() * 100:.1f}% of body"
        )


if __name__ == "__main__":
    main()
