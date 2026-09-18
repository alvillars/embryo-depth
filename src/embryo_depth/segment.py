"""Segment the embryo from background at the coarsest pyramid level.

Run on level 4: the whole 4D stack is 301 MB, so all 132 timepoints fit in RAM and the
parameters can be iterated on interactively. The mask is later upscaled to the finer
levels rather than being recomputed there.

Two measured properties of this dataset shape the implementation:

* The Otsu threshold drifts strongly over the series (1294 / 898 / 944 at t = 0 / 66 /
  131) as the embryo develops and the foreground fraction grows from 3% to 16%. The
  threshold is therefore computed per timepoint; a single global value fails at one end.

* The reporter labels cell boundaries, so the bright signal is membranes and the dark
  regions are cell interiors and, at the centre, the embryo's cavity. The embryo is a
  hollow shell: thresholding traces both its outer and inner surface. The cavity must be
  filled, or the distance transform measures depth from the inner surface as well and
  tissue beside the cavity reads as shallow when it is in fact deep. Skipping this does
  not look like a bug -- the resulting depth field is smooth and plausible, just wrong.
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

from .store import Dataset

SEGMENT_LEVEL = "4"

# Defaults in micrometres, so they carry over unchanged if segmentation is ever re-run at
# a finer level. At level 4 the voxel is 2.0 um in z but 4.16 um in xy -- z is the *finer*
# axis here, the opposite of level 0 -- so footprints must be built from physical sizes.
DEFAULT_CLOSING_UM = 12.0
DEFAULT_OPENING_UM = 0.0
DEFAULT_MIN_VOLUME_UM3 = 1e5
# Surface conditioning: opening deletes the bright protrusions, smoothing relaxes the rest.
DEFAULT_SURFACE_OPENING_UM = 12.0
DEFAULT_SURFACE_SMOOTHING_UM = 6.0


def ellipsoid_footprint(radius_um: float, voxel_um: tuple[float, float, float]) -> np.ndarray:
    """A binary ellipsoid of the given physical radius, sampled on an anisotropic grid."""
    radii = [max(1, int(round(radius_um / v))) for v in voxel_um]
    zz, yy, xx = np.ogrid[
        -radii[0] : radii[0] + 1, -radii[1] : radii[1] + 1, -radii[2] : radii[2] + 1
    ]
    return ((zz / radii[0]) ** 2 + (yy / radii[1]) ** 2 + (xx / radii[2]) ** 2) <= 1.0


def _padded_morphology(op, mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Apply a scipy morphology op with the array padded out by the footprint radius.

    ``ndi.binary_closing`` and ``binary_opening`` run an erosion whose ``border_value``
    defaults to 0, so anything within a footprint radius of the array boundary gets eroded
    away. On this dataset plain closing discards 7-8% of the body -- it is not extensive,
    which a closing must be. ``border_value=1`` overcorrects, inventing tissue along the
    border; padding is the only variant that both preserves the input and adds nothing.
    """
    pad = [(s // 2, s // 2) for s in footprint.shape]
    padded = np.pad(mask, pad)
    result = op(padded, footprint)
    return result[tuple(slice(lo, -hi if hi else None) for lo, hi in pad)]


def fill_interior(mask: np.ndarray) -> np.ndarray:
    """Fill cavities enclosed by tissue, including ones cut open by the volume boundary.

    The embryo is a hollow shell, and its central cavity must be filled: left open, the
    distance transform would measure depth from the *inner* surface as well, so tissue
    beside the cavity would read as shallow when it is in fact deep.

    Plain ``binary_fill_holes`` floods inward from the array border, so it fills the
    cavity only while the cavity is fully inside the imaged volume. From t~90 the embryo
    has grown until the cavity reaches the z=0 face, at which point the cavity becomes
    reachable from the border and is left open -- 16-20% of the body. Enlarging the
    closing does not help, because there is no neck to seal: the cavity is enclosed by
    tissue at every timepoint and only escapes across the crop plane.

    Sealing each face with its own 2D hole-filled version fixes this. A cavity that is
    ringed by tissue within the face plane cannot leak across it, while true exterior
    background stays background and remains connected to the padded border.
    """
    padded = np.zeros(tuple(s + 2 for s in mask.shape), dtype=bool)
    padded[1:-1, 1:-1, 1:-1] = mask
    padded[0, 1:-1, 1:-1] = ndi.binary_fill_holes(mask[0])
    padded[-1, 1:-1, 1:-1] = ndi.binary_fill_holes(mask[-1])
    padded[1:-1, 0, 1:-1] = ndi.binary_fill_holes(mask[:, 0])
    padded[1:-1, -1, 1:-1] = ndi.binary_fill_holes(mask[:, -1])
    padded[1:-1, 1:-1, 0] = ndi.binary_fill_holes(mask[:, :, 0])
    padded[1:-1, 1:-1, -1] = ndi.binary_fill_holes(mask[:, :, -1])
    return ndi.binary_fill_holes(padded)[1:-1, 1:-1, 1:-1]


def signed_distance(mask: np.ndarray, voxel_um: tuple[float, float, float]) -> np.ndarray:
    """Signed Euclidean distance field, positive inside the mask, in micrometres.

    The zero level set of this field *is* the mask surface, so resampling this field onto a
    finer grid and re-thresholding at zero is a mathematically principled way to interpolate
    an implicit binary surface -- unlike linearly interpolating the raw 0/1 field, which
    biases where the surface ends up. Shared by ``smooth_surface`` below (which then
    Gaussian-blurs the field, a curvature-flow operation) and ``upscale.smooth_upsample_to``
    (which linearly resamples the field onto a different grid, no blurring involved, so none
    of the curvature-flow volume-shrink bias documented below applies there).
    """
    inside = ndi.distance_transform_edt(mask, sampling=voxel_um)
    outside = ndi.distance_transform_edt(~mask, sampling=voxel_um)
    return (inside - outside).astype(np.float32)


def smooth_surface(
    mask: np.ndarray, voxel_um: tuple[float, float, float], sigma_um: float
) -> np.ndarray:
    """Fair the surface by smoothing the signed distance field, not the binary mask.

    Blurring a binary mask and re-thresholding biases against thin features and against
    regions of high curvature in a way that depends on voxel size. Smoothing the signed
    distance and re-thresholding at zero is the level-set equivalent and behaves like
    curvature flow. Distances are in micrometres, so ``sigma_um`` means the same thing at
    any level.

    Being curvature flow, it shrinks convex bodies: a sphere of radius R loses a shell of
    thickness sigma^2 / R, so its volume drops by 3 * sigma^2 / R^2. That is 12% for a
    30 um sphere but only ~0.4% for this embryo (R ~ 160 um) at sigma = 6 um, which is
    well under a level-4 voxel. Check it again before raising sigma much, or before
    reusing this on anything smaller or sharply curved.
    """
    signed = signed_distance(mask, voxel_um)
    smoothed = ndi.gaussian_filter(signed, [sigma_um / v for v in voxel_um], mode="nearest")
    return smoothed > 0


def background_mad_threshold(vol: np.ndarray, k: float = 12.0) -> float:
    """Threshold at ``k`` robust deviations above the background mode.

    An alternative to Otsu for timepoints where the embryo is small: Otsu's split is
    pulled upward when the histogram is overwhelmingly background, which can clip a dim
    early embryo. Background here is tight (median ~117, MAD ~10 counts) and signal starts
    several hundred counts above it, so this is well separated.
    """
    sample = vol[::4, ::4, ::4].astype(np.float32).ravel()
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median))) or 1.0
    return median + k * 1.4826 * mad


def compute_thresholds(vol: np.ndarray) -> dict[str, float]:
    """All candidate thresholds, so the review step can compare them."""
    return {
        "otsu": float(threshold_otsu(vol)),
        "mad": float(background_mad_threshold(vol)),
    }


def segment_timepoint(
    vol: np.ndarray,
    voxel_um: tuple[float, float, float],
    method: str = "otsu",
    closing_um: float = DEFAULT_CLOSING_UM,
    opening_um: float = DEFAULT_OPENING_UM,
    min_volume_um3: float = DEFAULT_MIN_VOLUME_UM3,
    fill: str = "interior",
    surface_opening_um: float = DEFAULT_SURFACE_OPENING_UM,
    surface_smoothing_um: float = DEFAULT_SURFACE_SMOOTHING_UM,
) -> tuple[np.ndarray, dict]:
    """Segment one timepoint into a solid body mask.

    Returns the mask and a stats dict for the per-timepoint QC series.
    """
    thresholds = compute_thresholds(vol)
    if method not in thresholds:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(thresholds)}")
    threshold = thresholds[method]

    mask = vol > threshold
    raw_voxels = int(mask.sum())

    if opening_um > 0:
        mask = _padded_morphology(
            ndi.binary_opening, mask, ellipsoid_footprint(opening_um, voxel_um)
        )

    # Close before filling. binary_fill_holes only fills fully enclosed cavities, so any
    # gap in the surface leaves the interior connected to the outside and unfilled.
    # Closing seals those gaps first; the order of these two steps is load-bearing.
    if closing_um > 0:
        mask = _padded_morphology(
            ndi.binary_closing, mask, ellipsoid_footprint(closing_um, voxel_um)
        )
    closed_voxels = int(mask.sum())

    # Cavity handling. "interior" is the default and the only mode that keeps the central
    # cavity filled across the whole series; see fill_interior. "enclosed" is plain
    # binary_fill_holes, which drops the cavity once it reaches the z=0 face; "none"
    # leaves it open, which makes depth get measured from the inner surface too.
    if fill == "interior":
        mask = fill_interior(mask)
    elif fill == "enclosed":
        mask = ndi.binary_fill_holes(mask)
    elif fill != "none":
        raise ValueError(f"unknown fill mode {fill!r}")
    filled_voxels = int(mask.sum())

    labels, n = ndi.label(mask)
    if n > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(counts.argmax())
    elif n == 0:
        mask = np.zeros_like(mask)

    # A second fill: removing other components can expose cavities that were previously
    # connected to the outside through them.
    if fill == "interior":
        mask = fill_interior(mask)
    elif fill == "enclosed":
        mask = ndi.binary_fill_holes(mask)

    # Surface conditioning, after the body is solid so it only shapes the outer surface.
    # Bright protrusions reach 17-42 um out of the embryo and are well above threshold, so
    # blurring will not remove them; an opening deletes anything thinner than its radius
    # and costs about half the volume that equivalent blurring does. The smoothing then
    # relaxes what roughness is left.
    presmooth_voxels = int(mask.sum())
    if surface_opening_um > 0:
        mask = _padded_morphology(
            ndi.binary_opening, mask, ellipsoid_footprint(surface_opening_um, voxel_um)
        )
    if surface_smoothing_um > 0:
        mask = smooth_surface(mask, voxel_um, surface_smoothing_um)

    if surface_opening_um > 0 or surface_smoothing_um > 0:
        # Both operations can shave off specks and punch pinholes through a thin shell, so
        # the component and cavity invariants have to be re-established afterwards --
        # otherwise conditioning silently reintroduces the unfilled-cavity bug.
        relabelled, n_after = ndi.label(mask)
        if n_after > 1:
            counts = np.bincount(relabelled.ravel())
            counts[0] = 0
            mask = relabelled == int(counts.argmax())
        if fill == "interior":
            mask = fill_interior(mask)
        elif fill == "enclosed":
            mask = ndi.binary_fill_holes(mask)

    voxel_volume = float(np.prod(voxel_um))
    body_voxels = int(mask.sum())

    # Does the body run into the edge of the imaged volume? Where it does, the embryo
    # surface was never imaged, so the distance transform will measure depth from a crop
    # plane rather than from real tissue boundary. Reported, not corrected -- the viewer
    # should treat those regions with suspicion.
    faces = [mask[0], mask[-1], mask[:, 0], mask[:, -1], mask[:, :, 0], mask[:, :, -1]]
    border_voxels = int(sum(int(f.sum()) for f in faces))

    # Largest background component that is not the exterior: any unfilled cavity left
    # inside the body. Must be 0 -- a nonzero value means the distance transform will
    # measure depth from an interior surface as well as from the embryo's outer surface.
    # Tracking this is what distinguishes "the cavity is filled" from "fill_holes ran".
    bg_labels, n_bg = ndi.label(~mask.astype(bool))
    residual_cavity = 0
    if n_bg > 1:
        counts = np.bincount(bg_labels.ravel())
        counts[0] = 0
        exterior = bg_labels[0, 0, 0]
        if exterior:
            counts[exterior] = 0
        residual_cavity = int(counts.max())

    stats = {
        "thresholds": thresholds,
        "threshold_used": threshold,
        "method": method,
        "raw_voxels": raw_voxels,
        "closed_voxels": closed_voxels,
        "filled_voxels": filled_voxels,
        "body_voxels": body_voxels,
        "body_volume_um3": body_voxels * voxel_volume,
        "foreground_fraction": body_voxels / mask.size,
        # Fraction of the final body that thresholding alone missed -- the gaps between
        # membranes that closing recovered. Small at level 4, where a 4.16 um voxel is
        # coarser than a cell and the membrane mesh already reads as solid tissue.
        "gap_fraction": (closed_voxels - raw_voxels) / closed_voxels if closed_voxels else 0.0,
        "enclosed_fill_fraction": (
            (filled_voxels - closed_voxels) / filled_voxels if filled_voxels else 0.0
        ),
        "discarded_fraction": (
            (filled_voxels - body_voxels) / filled_voxels if filled_voxels else 0.0
        ),
        "n_components": int(n),
        "border_voxels": border_voxels,
        "touches_border": border_voxels > 0,
        "residual_cavity_voxels": residual_cavity,
        "surface_conditioning_loss": (
            (presmooth_voxels - body_voxels) / presmooth_voxels if presmooth_voxels else 0.0
        ),
        "too_small": body_voxels * voxel_volume < min_volume_um3,
    }
    return mask.astype(np.uint8), stats


def _worker(args):
    t, store_path, level, kwargs = args
    ds = Dataset(store_path)
    vol = np.asarray(ds.image(level)[t, 0])
    mask, stats = segment_timepoint(vol, ds.levels[level].voxel_size_um, **kwargs)
    return t, mask, stats


def segment_all(
    store_path: Path,
    level: str = SEGMENT_LEVEL,
    workers: int = 8,
    **kwargs,
) -> tuple[np.ndarray, list[dict]]:
    """Segment every timepoint. Timepoints are fully independent."""
    ds = Dataset(store_path)
    lv = ds.levels[level]
    n_t = lv.n_timepoints
    masks = np.zeros((n_t, *lv.spatial_shape), dtype=np.uint8)
    stats: list[dict | None] = [None] * n_t

    jobs = [(t, store_path, level, kwargs) for t in range(n_t)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (t, mask, st) in enumerate(pool.map(_worker, jobs), start=1):
            masks[t] = mask
            stats[t] = st
            if done % 10 == 0 or done == n_t:
                rate = done / (time.time() - started)
                print(
                    f"  {done:3d}/{n_t} timepoints  ({rate:.1f}/s, "
                    f"{(n_t - done) / rate:.0f}s left)",
                    flush=True,
                )
    return masks, stats  # type: ignore[return-value]


def write_masks(ds: Dataset, masks: np.ndarray, level: str = SEGMENT_LEVEL) -> None:
    ds.create_label_group([level])
    arr = ds.label(level, mode="a")
    for t in range(masks.shape[0]):
        arr[t, 0] = masks[t]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=None)
    p.add_argument("--level", default=SEGMENT_LEVEL)
    p.add_argument("--method", default="otsu", choices=["otsu", "mad"])
    p.add_argument("--fill", default="interior", choices=["interior", "enclosed", "none"])
    p.add_argument("--closing-um", type=float, default=DEFAULT_CLOSING_UM)
    p.add_argument("--opening-um", type=float, default=DEFAULT_OPENING_UM)
    p.add_argument("--surface-opening-um", type=float, default=DEFAULT_SURFACE_OPENING_UM)
    p.add_argument("--surface-smoothing-um", type=float, default=DEFAULT_SURFACE_SMOOTHING_UM)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stats", type=Path, default=Path("segmentation_qc/stats.json"))
    p.add_argument("--dry-run", type=int, default=0, help="segment only the first N timepoints")
    args = p.parse_args()

    from .store import DEFAULT_STORE

    store = args.store or DEFAULT_STORE
    ds = Dataset(store)
    lv = ds.levels[args.level]
    print(f"segmenting {store.name} level {args.level} {lv.shape} voxel={lv.voxel_size_um} um")

    kwargs = dict(
        method=args.method,
        fill=args.fill,
        closing_um=args.closing_um,
        opening_um=args.opening_um,
        surface_opening_um=args.surface_opening_um,
        surface_smoothing_um=args.surface_smoothing_um,
    )
    if args.dry_run:
        masks = np.zeros((args.dry_run, *lv.spatial_shape), dtype=np.uint8)
        stats = []
        for t in range(args.dry_run):
            _, masks[t], st = _worker((t, store, args.level, kwargs))
            stats.append(st)
            print(f"  t={t} thr={st['threshold_used']:.0f} fg={st['foreground_fraction']:.4f} "
                  f"holes={st['hole_fraction']:.3f} comps={st['n_components']}")
    else:
        masks, stats = segment_all(store, args.level, workers=args.workers, **kwargs)
        write_masks(ds, masks, args.level)
        print(f"wrote labels/embryo/{args.level}")

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(
        json.dumps({"level": args.level, "params": kwargs, "stats": stats}, indent=1)
    )
    print(f"wrote {args.stats}")

    flagged = [t for t, s in enumerate(stats) if s["too_small"] or s["n_components"] == 0]
    if flagged:
        print(f"WARNING: {len(flagged)} timepoints look degenerate: {flagged}")


if __name__ == "__main__":
    main()
