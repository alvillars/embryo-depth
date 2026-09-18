"""Persist the depth-bucketed index to disk, chunked, instead of rebuilding it in RAM.

``viewer.py``'s ``build_index`` sorts one timepoint's body voxels by depth every time a
session starts, so that a depth-band query is a contiguous slice instead of a full-volume
scan. That sort is what turns depth into a queryable axis at all -- unlike z/y/x, depth is a
per-voxel *derived* value, not a physical storage axis, so there is no byte range on disk
that "is" depth 40 until something sorts the voxels that way.

This module does that sort once, at precompute time, and writes the result into
``depth_index/{level}`` chunked in small row-blocks (see ``store.create_depth_index_group``).
A reader then needs only a lookup in the tiny, always-resident ``offsets`` table plus one
small chunked zarr read to answer "the shell at depth d, timepoint t" -- genuine chunk-level
laziness, the same shape of cheapness z-chunk reads already have, with no need to hold every
timepoint's index in RAM for the whole session.

Must run after ``depth.py``'s ``build_depth``: this composes the already-computed
``labels/embryo/{level}`` and ``depth/{level}`` arrays (plus the base image, for the stored
intensity values), it does not recompute the distance transform.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .depth import crop_bounded, touched_faces
from .store import DEFAULT_STORE, DEPTH_LEVELS, DEPTH_SCALE_UM, Dataset, Level


def _max_body_voxels(ds: Dataset, level: str) -> int:
    """Widest per-timepoint body-voxel count, from the already-computed label array.

    A boolean sum over already-segmented data, not a new distance transform -- used only to
    size the depth_index arrays before any data is written.
    """
    arr = ds.label(level)
    return max(int(np.count_nonzero(arr[t, 0])) for t in range(ds.n_timepoints))


def build_index_rows(
    image: np.ndarray, depth_units: np.ndarray, level: Level
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Counting-sort one timepoint's body voxels by depth.

    Exactly ``viewer.build_index``'s logic, returning the four arrays ready to write into
    one timepoint's row-span of ``depth_index/{level}`` rather than a ``DepthIndex`` object.
    """
    body = depth_units > 0
    coords = np.argwhere(body).astype(np.int16)
    d = depth_units[body]
    order = np.argsort(d, kind="stable")

    coords = coords[order]
    values = image[body][order].astype(np.uint16)
    counts = np.bincount(d, minlength=256)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    faces = touched_faces(body)
    if faces:
        flagged = crop_bounded(depth_units.astype(np.float32) * DEPTH_SCALE_UM, level, faces)
        reliable = (~flagged[body][order]).astype(np.uint8)
    else:
        reliable = np.ones(len(values), dtype=np.uint8)
    return coords, values, reliable, offsets


def _worker(args) -> int:
    t, store_path, level = args
    ds = Dataset(store_path)
    lv = ds.levels[level]
    image = np.asarray(ds.image(level)[t, 0])
    depth_units = np.asarray(ds.depth(level)[t, 0])
    coords, values, reliable, offsets = build_index_rows(image, depth_units, lv)
    n = len(values)
    ds.depth_index(level, "coords", "a")[t, :n] = coords
    ds.depth_index(level, "values", "a")[t, :n] = values
    ds.depth_index(level, "reliable", "a")[t, :n] = reliable
    ds.depth_index(level, "offsets", "a")[t] = offsets
    return t


def build_depth_index(
    store_path: Path,
    levels: list[str] = list(DEPTH_LEVELS),
    workers: int = 4,
) -> None:
    ds = Dataset(store_path)
    for level in levels:
        if not ds.has_depth():
            raise SystemExit(f"no depth/{level} -- run `python -m embryo_depth.depth` first")

    print("sizing depth_index arrays from labels/embryo (a voxel count, not a new EDT)...")
    max_body = {level: _max_body_voxels(ds, level) for level in levels}
    for level in levels:
        print(f"  level {level}: max {max_body[level]} body voxels/timepoint")

    ds.create_depth_index_group(levels, max_body)

    for level in sorted(levels, key=int, reverse=True):
        n_t = ds.n_timepoints
        jobs = [(t, store_path, level) for t in range(n_t)]
        started = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, _ in enumerate(pool.map(_worker, jobs), start=1):
                if done % 10 == 0 or done == n_t:
                    rate = done / (time.time() - started)
                    print(
                        f"  level {level}: {done:3d}/{n_t} ({rate:.2f}/s, "
                        f"{(n_t - done) / rate:.0f}s left)",
                        flush=True,
                    )


class PersistedDepthIndex:
    """Disk-backed equivalent of ``viewer.py``'s ``DepthIndex``, reading ``depth_index/{level}``.

    Where ``DepthIndex`` is built by scanning and sorting a whole timepoint's volume in RAM
    at every viewer launch, this reads the already-sorted layout straight off disk:
    ``.offsets`` is tiny and kept fully resident (like ``DepthIndex.offsets``), and
    ``.band()`` touches only the on-disk chunks overlapping the requested row range -- no
    full-timepoint decompression, no full-series RAM residency.
    """

    def __init__(self, ds: Dataset, level: str):
        self.ds = ds
        self.level = level
        self.coords = ds.depth_index(level, "coords")
        self.values = ds.depth_index(level, "values")
        self.reliable = ds.depth_index(level, "reliable")
        self.offsets = np.asarray(ds.depth_index(level, "offsets"))  # tiny; eager is fine
        self.shape = ds.levels[level].spatial_shape

    def band(self, t: int, lo: int, hi: int):
        lo = int(np.clip(lo, 0, 255))
        hi = int(np.clip(hi, 0, 255))
        r0, r1 = int(self.offsets[t, lo]), int(self.offsets[t, hi + 1])
        return (
            np.asarray(self.coords[t, r0:r1]),
            np.asarray(self.values[t, r0:r1]),
            np.asarray(self.reliable[t, r0:r1]).astype(bool),
        )

    def shell(self, t: int, depth: int, half_width: int, reliable_only: bool = False):
        """Coordinates and intensities of the shell at ``depth`` +/- ``half_width``."""
        coords, values, reliable = self.band(t, depth - half_width, depth + half_width)
        if reliable_only:
            coords, values = coords[reliable], values[reliable]
        return coords, values

    def shell_volume(self, t: int, depth: int, half_width: int, reliable_only: bool = False):
        """The same shell rendered as a dense 3D volume for napari's volume renderer."""
        coords, values = self.shell(t, depth, half_width, reliable_only)
        out = np.zeros(self.shape, dtype=np.uint16)
        if len(coords):
            out[coords[:, 0], coords[:, 1], coords[:, 2]] = values
        return out

    @property
    def max_depth_unit(self) -> int:
        """Deepest bucket that actually holds voxels, across every timepoint."""
        occupied = np.nonzero(np.diff(self.offsets, axis=1).any(axis=0))[0]
        return int(occupied.max()) if len(occupied) else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--levels", nargs="+", default=list(DEPTH_LEVELS))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    build_depth_index(args.store, args.levels, args.workers)
    print("done")


if __name__ == "__main__":
    main()
