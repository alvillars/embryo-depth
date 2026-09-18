"""Upscale the level-4 mask into a full NGFF label pyramid.

Only the mask is upscaled -- segmentation runs once, at level 4. A level-0 label
therefore has 0.26 um voxels but still carries only level-4 information, so its true
surface accuracy stays around 4 um. Nothing downstream should assume otherwise.

The resampling is done in physical coordinates rather than by a fixed factor, because
this pyramid cannot be walked with one. Its shapes were produced by ``floor(n/2)``
(2303 -> 1151 -> 575 -> 287 -> 143), so the ratios are not exactly 2 going back up, and
each level also carries a sub-voxel ``translation`` recording where its grid sits. Mapping
target voxel centres through micrometres and back handles both at once.

Two resampling methods share that same coordinate mapping:

* ``upsample_to`` -- nearest-neighbour. Every source voxel becomes an axis-aligned block
  of identical target voxels (up to 8x8 in xy at the level4->level1 magnification), which
  is a visible staircase at that ratio. Kept as the default for level 0, where its cost
  (~1.6 GB/timepoint) is far cheaper than the alternative below and the level is only ever
  touched to validate the final upscale.
* ``smooth_upsample_to`` -- resamples the source mask's *signed distance field* linearly
  instead of the raw 0/1 field, then re-thresholds at zero. This is not the "interpolate
  the binary mask and re-threshold" shortcut warned about above -- that shifts the surface
  by an amount that depends on local shape, because a 0/1 field has no meaningful slope
  near the surface to interpolate. The signed distance field does: its zero level set *is*
  the surface, and it varies smoothly and roughly linearly nearby, so linearly resampling
  it and re-thresholding removes the staircase without moving the surface or inventing
  information beyond the source mask's own ~4 um accuracy. Default for levels 1-3.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .segment import SEGMENT_LEVEL, signed_distance
from .store import DEFAULT_STORE, Dataset, Level


def _source_coords(source: Level, target: Level, axis: int) -> np.ndarray:
    """Continuous (unrounded) source-index coordinate for each target voxel centre.

    ``axis`` is 0, 1 or 2 for z, y, x. Target voxel centres are converted to micrometres
    using the target's own scale and translation, then back through the source's, so the
    result is correct whatever the shape ratio and whatever sub-voxel offset each level
    records. Shared by the nearest-neighbour and linear-interpolation resamplers below.
    """
    ax = axis + 2  # transformations are in tczyx order
    n_target = target.shape[ax]
    centres_um = target.translation[ax] + (np.arange(n_target) + 0.5) * target.scale[ax]
    return (centres_um - source.translation[ax]) / source.scale[ax] - 0.5


def nearest_indices(source: Level, target: Level, axis: int) -> np.ndarray:
    """Index into ``source`` for each voxel of ``target`` along one spatial axis."""
    n_source = source.shape[axis + 2]
    src = _source_coords(source, target, axis)
    return np.clip(np.rint(src).astype(np.int64), 0, n_source - 1)


def upsample_to(volume: np.ndarray, source: Level, target: Level) -> np.ndarray:
    """Nearest-neighbour resample of one timepoint from ``source`` to ``target`` geometry.

    Nearest-neighbour is not a shortcut here: interpolating a binary mask linearly and
    re-thresholding would move the surface by a sub-voxel amount that varies with the
    local shape, and the surface is the one thing this pipeline is measuring from. See
    ``smooth_upsample_to`` for the correct way to interpolate it smoothly.
    """
    iz = nearest_indices(source, target, 0)
    iy = nearest_indices(source, target, 1)
    ix = nearest_indices(source, target, 2)
    return volume[np.ix_(iz, iy, ix)]


def linear_axis_weights(
    source: Level, target: Level, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bracketing source indices and blend weight for linear interpolation along one axis.

    Returns ``(lo, hi, weight_hi)``: ``lo``/``hi`` are the two source indices to blend for
    each target index along ``axis``, both clipped to ``[0, n_source - 1]``; ``weight_hi``
    is the blend weight for ``hi`` in ``[0, 1]`` (the weight for ``lo`` is ``1 - weight_hi``).
    Reduces to ``lo == hi`` (``weight_hi`` all zero) wherever source and target share scale
    and translation on this axis -- e.g. z between levels 1-4, which this pyramid never
    magnifies in z -- so that axis costs nothing extra in ``_blend_axis``.
    """
    n_source = source.shape[axis + 2]
    src = np.clip(_source_coords(source, target, axis), 0, n_source - 1)
    lo = np.floor(src).astype(np.int64)
    hi = np.clip(lo + 1, 0, n_source - 1)
    weight_hi = (src - lo).astype(np.float32)
    return lo, hi, weight_hi


def _blend_axis(
    volume: np.ndarray, lo: np.ndarray, hi: np.ndarray, weight_hi: np.ndarray, axis: int
) -> np.ndarray:
    """Blend two gathers along one axis. Skips the blend entirely when it would be a no-op."""
    if not weight_hi.any():
        return np.take(volume, lo, axis=axis)
    lo_vals = np.take(volume, lo, axis=axis)
    hi_vals = np.take(volume, hi, axis=axis)
    shape = [1] * volume.ndim
    shape[axis] = -1
    w = weight_hi.reshape(shape)
    lo_vals *= 1.0 - w  # in place, to cap the number of full-size temporaries alive at once
    lo_vals += hi_vals * w
    return lo_vals


def _smooth_resample(sdf: np.ndarray, source: Level, target: Level) -> np.ndarray:
    """Separably interpolate a signed-distance field onto ``target``'s grid and re-threshold.

    Three 1-D passes (z, then y, then x) rather than one N-D ``map_coordinates`` call:
    building an explicit ``(3, *target_shape)`` coordinate array would be ~19 GB in float32
    alone for level 0. The axis mapping here has no shear or rotation (see module docstring),
    so it is exactly separable. z goes first because it is the least-magnified axis across
    this pyramid (identity for levels 1-4, 2x for level 4->0), keeping the smaller
    intermediate array around longest and deferring the expensive xy blow-up to the end.
    """
    out = sdf
    for axis in (0, 1, 2):
        lo, hi, weight_hi = linear_axis_weights(source, target, axis)
        out = _blend_axis(out, lo, hi, weight_hi, axis)
    return (out > 0).astype(np.uint8)


def smooth_upsample_to(mask: np.ndarray, source: Level, target: Level) -> np.ndarray:
    """Resample ``mask`` from ``source`` to ``target`` geometry via its signed distance field.

    Unlike ``upsample_to`` (block-replication nearest-neighbour, which literally copies each
    source voxel into an up-to-8x8 axis-aligned block of target voxels), this interpolates
    the continuous, implicit surface the mask represents, removing the staircase artifact
    without inventing new surface information -- the source mask's own accuracy (~4 um for a
    level-4 mask, per the module docstring) is unchanged, only how it is resampled.
    """
    sdf = signed_distance(mask.astype(bool), source.voxel_size_um)
    return _smooth_resample(sdf, source, target)


def _worker(args) -> tuple[int, float]:
    t, store_path, source_level, target_levels, smooth_levels = args
    ds = Dataset(store_path)
    src = ds.levels[source_level]
    started = time.time()
    mask = np.asarray(ds.label(source_level)[t, 0])
    sdf = signed_distance(mask.astype(bool), src.voxel_size_um) if smooth_levels else None
    for level in target_levels:
        if level in smooth_levels:
            out = _smooth_resample(sdf, src, ds.levels[level])
        else:
            out = upsample_to(mask, src, ds.levels[level])
        ds.label(level, mode="a")[t, 0] = out
    return t, time.time() - started


def build_pyramid(
    store_path: Path,
    source_level: str = SEGMENT_LEVEL,
    target_levels: list[str] | None = None,
    workers: int = 4,
    smooth_levels: frozenset[str] | None = None,
) -> None:
    """Build the label pyramid. ``smooth_levels`` selects which target levels use
    ``smooth_upsample_to`` instead of ``upsample_to``; defaults to every target level except
    "0" (level 0's cost is much higher and it is only ever touched to validate the upscale --
    see module docstring). Pass an empty ``frozenset()`` to disable smoothing entirely.
    """
    ds = Dataset(store_path)
    if target_levels is None:
        target_levels = [lv for lv in ds.levels if lv != source_level]
    target_levels = sorted(target_levels, key=int, reverse=True)
    if smooth_levels is None:
        smooth_levels = frozenset(target_levels) - {"0"}

    # (Re)create the group so the multiscales metadata lists every level that now exists.
    ds.create_label_group(sorted(ds.levels, key=int))

    n_t = ds.n_timepoints
    jobs = [(t, store_path, source_level, target_levels, smooth_levels) for t in range(n_t)]
    started = time.time()
    # Fewer workers than cores: a level-0 timepoint is 1.6e9 voxels and the write is
    # I/O bound, so more processes mostly contend for memory bandwidth.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (t, dt) in enumerate(pool.map(_worker, jobs), start=1):
            if done % 5 == 0 or done == n_t:
                rate = done / (time.time() - started)
                print(
                    f"  {done:3d}/{n_t} timepoints ({rate:.2f}/s, "
                    f"{(n_t - done) / rate / 60:.1f} min left)",
                    flush=True,
                )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--source-level", default=SEGMENT_LEVEL)
    p.add_argument("--levels", nargs="+", default=None, help="target levels, e.g. 0 1 2 3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--smooth-levels",
        nargs="*",
        default=None,
        help=(
            "levels to resample via signed-distance interpolation instead of "
            "nearest-neighbour; default: every target level except 0. Pass with no "
            "values to disable smoothing entirely, or include 0 to opt it in."
        ),
    )
    p.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="upscale only the first N timepoints, without writing, and print a comparison",
    )
    args = p.parse_args()

    ds = Dataset(args.store)
    targets = args.levels or [lv for lv in ds.levels if lv != args.source_level]
    targets = sorted(targets, key=int, reverse=True)
    smooth_levels = (
        None if args.smooth_levels is None else frozenset(args.smooth_levels)
    )
    print(f"upscaling labels/embryo/{args.source_level} -> {sorted(targets, key=int)}")
    for lv in sorted(targets, key=int):
        print(f"  level {lv}: {ds.levels[lv].shape}  voxel {ds.levels[lv].voxel_size_um} um")

    if args.dry_run:
        resolved_smooth = (
            frozenset(targets) - {"0"} if smooth_levels is None else smooth_levels
        )
        print(
            f"dry run: {args.dry_run} timepoints, "
            f"smoothing levels {sorted(resolved_smooth, key=int)}"
        )
        src = ds.levels[args.source_level]
        for t in range(args.dry_run):
            mask = np.asarray(ds.label(args.source_level)[t, 0])
            for level in targets:
                tgt = ds.levels[level]
                nn = upsample_to(mask, src, tgt)
                line = f"  t={t} level={level} nn_voxels={int(nn.sum())}"
                if level in resolved_smooth:
                    sm = smooth_upsample_to(mask, src, tgt)
                    line += f" smooth_voxels={int(sm.sum())}"
                print(line)
        return

    build_pyramid(args.store, args.source_level, targets, args.workers, smooth_levels)
    print("done")


if __name__ == "__main__":
    main()
