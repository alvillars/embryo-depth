"""Segment the lumen (and its complement, the tissue shell) from the embryo body.

Run on level 4, same as segment.py, and requires labels/embryo/{level} to already exist --
the embryo mask is the region searched for a lumen, not something this module derives itself.

The reporter labels cell boundaries (see segment.py's module docstring), so within the embryo
body the lumen -- the central, largely membrane-free cavity/interior -- is the *dark* region:
thresholding the image intensity inside the embryo mask and keeping voxels below the threshold
isolates it from the brighter tissue around it.

Otsu's threshold is computed only over intensities *inside* the embryo mask. Including the
zero-valued background outside the embryo (as in a naive `mask = embryo * image` on the full
volume) would badly skew the split, because in 3D the embryo occupies a much smaller fraction
of the full imaged volume than it does of one central 2D slice.

Thresholding alone does not isolate the lumen: every individual cell's cytoplasm is also
dark relative to its own membrane, so `vol < threshold` picks up cell-sized dark blobs
throughout the whole tissue, not just the one large central cavity. This gets worse as the
embryo cellularises -- on this dataset the raw component count grows from ~11 at t=0 to
~900 by t=131. Left unfiltered, the default closing radius (sized to seal membrane-mesh
gaps, not to distinguish a cavity from a cell) bridges many of those per-cell blobs into one
large connected mass, and the segmented "lumen" ends up including a large chunk of tissue.
``opening_um`` runs first, before closing or component selection, specifically to erase
blobs on the scale of one cell while leaving a genuinely larger cavity intact -- this is a
different step from ``surface_opening_um``, which (like segment.py's) shapes the final
lumen's own outer surface *after* closing.

Which surviving candidate is "the lumen" is picked by depth, not size: at early timepoints
the embryo has fewer, larger blastomeres, so the biggest dark blob after opening can be a
single peripheral cell rather than the true central cavity. See
``_select_lumen_component``.

``tissue`` is not independently thresholded -- it is defined as everything in the embryo body
that is not lumen (``embryo & ~lumen``), then upscaled through the pyramid the same way as
``lumen`` and ``embryo`` via ``upscale.py --label tissue``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from .depth import surface_distance_um
from .segment import (
    SEGMENT_LEVEL,
    _padded_morphology,
    ellipsoid_footprint,
    fill_interior,
    smooth_surface,
)
from .store import Dataset

LUMEN_LEVEL = SEGMENT_LEVEL

# Same defaults as segment.py's embryo conditioning -- see its module docstring for why these
# physical sizes (not voxel counts) are the right units to carry across pyramid levels.
# DEFAULT_LUMEN_OPENING_UM is a radius bigger than one cell but smaller than the true cavity --
# unlike segment.py's own (unused-by-default) opening_um, lumen needs this on by default. Tuned
# against Position_10 by sweeping and watching lumen_fraction_of_embryo / n_components: 12 um
# left many timepoints with several components and lumen_fraction_of_embryo up to 0.5 (cell
# cytoplasm still bridged in); 20 um collapses 116/132 timepoints to a single component, none
# flagged too_small, and lumen_fraction_of_embryo settles to 0.05-0.35 (mean 0.25). Too small
# and per-cell blobs still get through (fraction stays large, many components); too large and it
# starts eroding the true cavity too (fraction drops toward 0, or too_small trips early on).
DEFAULT_LUMEN_OPENING_UM = 20.0
DEFAULT_LUMEN_CLOSING_UM = 12.0
DEFAULT_LUMEN_SURFACE_OPENING_UM = 12.0
DEFAULT_LUMEN_SURFACE_SMOOTHING_UM = 6.0
DEFAULT_MIN_LUMEN_VOLUME_UM3 = 1e5


def lumen_threshold(vol: np.ndarray, embryo_mask: np.ndarray) -> float:
    """Otsu threshold over the image intensity inside the embryo only."""
    return float(threshold_otsu(vol[embryo_mask]))


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, n = ndi.label(mask)
    if n > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(counts.argmax())
    elif n == 0:
        mask = np.zeros_like(mask)
    return mask, n


def _select_lumen_component(
    mask: np.ndarray, embryo_bool: np.ndarray, voxel_um: tuple[float, float, float]
) -> tuple[np.ndarray, int, bool]:
    """Pick the candidate component containing the embryo's own deepest interior point.

    Picking "largest by voxel count" picks whichever dark blob happens to be biggest, which
    at early timepoints -- fewer, larger blastomeres before the embryo fully cellularises --
    can be a single peripheral cell rather than the true central cavity. The point of the
    embryo farthest from its own outer surface necessarily sits inside whatever structure
    occupies its core: if that is a hollow cavity, the point is inside the cavity, not inside
    a cell nearer the surface.

    If no candidate contains that point, no dark structure occupies the embryo's actual core
    -- e.g. before a lumen has opened up developmentally -- and an all-zero mask is returned
    rather than guessing among peripheral candidates. On Position_10 this is never a matter
    of the opening radius: swept 0-20 um, the deepest point fell inside a candidate at every
    early timepoint tested or never did, regardless of how little erosion was applied.

    Returns ``(mask, n_components, found)``.
    """
    labels, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask), n, False
    embryo_depth = surface_distance_um(embryo_bool, voxel_um)
    deepest = np.unravel_index(np.argmax(embryo_depth), embryo_depth.shape)
    target = labels[deepest]
    if target == 0:
        return np.zeros_like(mask), n, False
    return labels == target, n, True


def segment_lumen_timepoint(
    vol: np.ndarray,
    embryo_mask: np.ndarray,
    voxel_um: tuple[float, float, float],
    opening_um: float = DEFAULT_LUMEN_OPENING_UM,
    closing_um: float = DEFAULT_LUMEN_CLOSING_UM,
    surface_opening_um: float = DEFAULT_LUMEN_SURFACE_OPENING_UM,
    surface_smoothing_um: float = DEFAULT_LUMEN_SURFACE_SMOOTHING_UM,
    min_volume_um3: float = DEFAULT_MIN_LUMEN_VOLUME_UM3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Segment one timepoint's lumen and tissue mask within an existing embryo mask.

    Returns ``(lumen_mask, tissue_mask, stats)``, both uint8.
    """
    embryo_bool = embryo_mask.astype(bool)
    threshold = lumen_threshold(vol, embryo_bool)

    mask = embryo_bool & (vol < threshold)
    raw_voxels = int(mask.sum())
    raw_components = int(ndi.label(mask)[1])

    # Erase cell-sized dark blobs before anything else touches the mask -- closing or
    # component selection run on the unfiltered candidate would otherwise bridge or count
    # per-cell cytoplasm as if it were part of one large cavity. See module docstring.
    if opening_um > 0:
        mask = _padded_morphology(
            ndi.binary_opening, mask, ellipsoid_footprint(opening_um, voxel_um)
        )
    opened_voxels = int(mask.sum())

    mask, n_components, lumen_found = _select_lumen_component(mask, embryo_bool, voxel_um)

    if closing_um > 0:
        mask = _padded_morphology(
            ndi.binary_closing, mask, ellipsoid_footprint(closing_um, voxel_um)
        )
    # Closing can push the mask outward past the embryo's inner surface.
    mask &= embryo_bool
    closed_voxels = int(mask.sum())

    presmooth_voxels = closed_voxels
    if surface_opening_um > 0:
        mask = _padded_morphology(
            ndi.binary_opening, mask, ellipsoid_footprint(surface_opening_um, voxel_um)
        )
    if surface_smoothing_um > 0:
        mask = smooth_surface(mask, voxel_um, surface_smoothing_um)
    mask &= embryo_bool

    if surface_opening_um > 0 or surface_smoothing_um > 0:
        # Conditioning can shave off specks or reconnect the mask to a disconnected speck;
        # re-establish the single-component invariant afterwards, same as segment.py.
        mask, _ = _largest_component(mask)
        mask &= embryo_bool

    lumen_mask = fill_interior(mask) & embryo_bool
    body_voxels = int(lumen_mask.sum())
    tissue_mask = embryo_bool & ~lumen_mask

    voxel_volume = float(np.prod(voxel_um))
    stats = {
        "threshold_used": threshold,
        "raw_voxels": raw_voxels,
        "raw_components": raw_components,
        "opened_voxels": opened_voxels,
        "closed_voxels": closed_voxels,
        "body_voxels": body_voxels,
        "lumen_volume_um3": body_voxels * voxel_volume,
        "tissue_voxels": int(tissue_mask.sum()),
        "embryo_voxels": int(embryo_bool.sum()),
        "lumen_fraction_of_embryo": (
            body_voxels / int(embryo_bool.sum()) if embryo_bool.any() else 0.0
        ),
        "n_components": int(n_components),
        "lumen_found": lumen_found,
        "surface_conditioning_loss": (
            (presmooth_voxels - body_voxels) / presmooth_voxels if presmooth_voxels else 0.0
        ),
        "too_small": body_voxels * voxel_volume < min_volume_um3,
    }
    return lumen_mask.astype(np.uint8), tissue_mask.astype(np.uint8), stats


def _worker(args):
    t, store_path, level, kwargs = args
    ds = Dataset(store_path)
    vol = np.asarray(ds.image(level)[t, 0])
    embryo_mask = np.asarray(ds.label(level, name="embryo")[t, 0])
    lumen_mask, tissue_mask, stats = segment_lumen_timepoint(
        vol, embryo_mask, ds.levels[level].voxel_size_um, **kwargs
    )
    return t, lumen_mask, tissue_mask, stats


def segment_lumen_all(
    store_path: Path,
    level: str = LUMEN_LEVEL,
    workers: int = 8,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Segment every timepoint's lumen/tissue. Timepoints are fully independent."""
    ds = Dataset(store_path)
    if not ds.has_labels(name="embryo"):
        raise RuntimeError(
            f"labels/embryo not found in {store_path} -- run `python -m embryo_depth.segment` "
            "first, lumen segmentation reads the embryo mask as its search region"
        )
    lv = ds.levels[level]
    n_t = lv.n_timepoints
    lumen_masks = np.zeros((n_t, *lv.spatial_shape), dtype=np.uint8)
    tissue_masks = np.zeros((n_t, *lv.spatial_shape), dtype=np.uint8)
    stats: list[dict | None] = [None] * n_t

    jobs = [(t, store_path, level, kwargs) for t in range(n_t)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (t, lumen_mask, tissue_mask, st) in enumerate(pool.map(_worker, jobs), start=1):
            lumen_masks[t] = lumen_mask
            tissue_masks[t] = tissue_mask
            stats[t] = st
            if done % 10 == 0 or done == n_t:
                rate = done / (time.time() - started)
                print(
                    f"  {done:3d}/{n_t} timepoints  ({rate:.1f}/s, "
                    f"{(n_t - done) / rate:.0f}s left)",
                    flush=True,
                )
    return lumen_masks, tissue_masks, stats  # type: ignore[return-value]


def write_lumen_masks(
    ds: Dataset, lumen_masks: np.ndarray, tissue_masks: np.ndarray, level: str = LUMEN_LEVEL
) -> None:
    ds.create_label_group([level], name="lumen")
    ds.create_label_group([level], name="tissue")
    lumen_arr = ds.label(level, mode="a", name="lumen")
    tissue_arr = ds.label(level, mode="a", name="tissue")
    for t in range(lumen_masks.shape[0]):
        lumen_arr[t, 0] = lumen_masks[t]
        tissue_arr[t, 0] = tissue_masks[t]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=None)
    p.add_argument("--level", default=LUMEN_LEVEL)
    p.add_argument("--opening-um", type=float, default=DEFAULT_LUMEN_OPENING_UM)
    p.add_argument("--closing-um", type=float, default=DEFAULT_LUMEN_CLOSING_UM)
    p.add_argument("--surface-opening-um", type=float, default=DEFAULT_LUMEN_SURFACE_OPENING_UM)
    p.add_argument(
        "--surface-smoothing-um", type=float, default=DEFAULT_LUMEN_SURFACE_SMOOTHING_UM
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stats", type=Path, default=Path("segmentation_qc/lumen_stats.json"))
    p.add_argument("--dry-run", type=int, default=0, help="segment only the first N timepoints")
    args = p.parse_args()

    from .store import DEFAULT_STORE

    store = args.store or DEFAULT_STORE
    ds = Dataset(store)
    lv = ds.levels[args.level]
    print(
        f"segmenting lumen: {store.name} level {args.level} {lv.shape} "
        f"voxel={lv.voxel_size_um} um"
    )

    kwargs = dict(
        opening_um=args.opening_um,
        closing_um=args.closing_um,
        surface_opening_um=args.surface_opening_um,
        surface_smoothing_um=args.surface_smoothing_um,
    )
    if args.dry_run:
        if not ds.has_labels(name="embryo"):
            raise RuntimeError("labels/embryo not found -- run embryo_depth.segment first")
        lumen_masks = np.zeros((args.dry_run, *lv.spatial_shape), dtype=np.uint8)
        stats = []
        for t in range(args.dry_run):
            _, lumen_masks[t], _, st = _worker((t, store, args.level, kwargs))
            stats.append(st)
            print(
                f"  t={t} thr={st['threshold_used']:.0f} "
                f"frac={st['lumen_fraction_of_embryo']:.4f} "
                f"raw_comps={st['raw_components']} comps={st['n_components']} "
                f"lumen_found={st['lumen_found']}"
            )
    else:
        lumen_masks, tissue_masks, stats = segment_lumen_all(
            store, args.level, workers=args.workers, **kwargs
        )
        write_lumen_masks(ds, lumen_masks, tissue_masks, args.level)
        print(f"wrote labels/lumen/{args.level}")
        print(f"wrote labels/tissue/{args.level}")

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps({"level": args.level, "params": kwargs, "stats": stats}, indent=1)
    )
    print(f"wrote {args.stats}")

    no_lumen = [t for t, s in enumerate(stats) if not s["lumen_found"]]
    if no_lumen:
        print(
            f"NOTE: {len(no_lumen)} timepoints had no dark structure at the embryo's own "
            f"core (empty lumen/full tissue written): {no_lumen}"
        )
    flagged = [t for t, s in enumerate(stats) if s["too_small"] and s["lumen_found"]]
    if flagged:
        print(
            f"WARNING: {len(flagged)} timepoints found a lumen but it looks degenerate: {flagged}"
        )


if __name__ == "__main__":
    main()
