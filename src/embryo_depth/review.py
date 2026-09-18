"""Visual review of the level-4 segmentation, before anything downstream is computed.

Opens napari with the level-4 image and the candidate masks overlaid, scrubbable over all
132 timepoints, and writes a QC figure of the per-timepoint series.

This is a gate, not a report. The distance field, the whole pyramid, and the viewer are
all derived from this mask and inherit its errors, and a wrong mask yields a depth field
that looks perfectly smooth and plausible while being wrong -- there is no later step at
which the mistake becomes obvious. Look at it before moving on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .segment import (
    DEFAULT_CLOSING_UM,
    DEFAULT_SURFACE_OPENING_UM,
    DEFAULT_SURFACE_SMOOTHING_UM,
    SEGMENT_LEVEL,
    _worker,
)
from .store import DEFAULT_STORE, Dataset

QC_DIR = Path("segmentation_qc")


def default_conditioning() -> dict:
    """Mask-shaping parameters, as used in the cache key and passed to the worker."""
    return {
        "closing_um": DEFAULT_CLOSING_UM,
        "surface_opening_um": DEFAULT_SURFACE_OPENING_UM,
        "surface_smoothing_um": DEFAULT_SURFACE_SMOOTHING_UM,
    }


def _tag(method: str, fill: str, level: str, params: dict | None = None) -> str:
    """Cache key covering every parameter that changes the mask.

    Every one of them has to appear here. Keying on only method and fill meant that adding
    surface conditioning silently reused pre-conditioning masks, so the review would have
    shown a mask the pipeline no longer produces -- the exact failure this gate exists to
    catch, arriving through the gate itself.
    """
    parts = [method, f"fill-{fill}"]
    for key in sorted(params or {}):
        parts.append(f"{key}-{params[key]:g}")
    return "_".join(parts) + f"_L{level}"


def cached_masks(
    store: Path,
    level: str,
    method: str,
    fill: str = "none",
    workers: int = 8,
    params: dict | None = None,
) -> np.ndarray:
    """Masks for one parameter set, computed once and cached as .npy."""
    params = params if params is not None else default_conditioning()
    tag = _tag(method, fill, level, params)
    cache = QC_DIR / f"masks_{tag}.npy"
    if cache.exists():
        return np.load(cache, mmap_mode="r")

    from concurrent.futures import ProcessPoolExecutor

    ds = Dataset(store)
    lv = ds.levels[level]
    masks = np.zeros((lv.n_timepoints, *lv.spatial_shape), dtype=np.uint8)
    stats = []
    jobs = [
        (t, store, level, {"method": method, "fill": fill, **params})
        for t in range(lv.n_timepoints)
    ]
    print(f"computing {tag} masks for {lv.n_timepoints} timepoints...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for t, mask, st in pool.map(_worker, jobs):
            masks[t] = mask
            stats.append((t, st))
    QC_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, masks)
    (QC_DIR / f"stats_{tag}.json").write_text(
        json.dumps([s for _, s in sorted(stats)], indent=1)
    )
    return masks


def qc_figure(methods: dict[str, list[dict]], out: Path) -> None:
    """Per-timepoint series that reveal instability a single frame would hide."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("threshold_used", "threshold (counts)"),
        ("body_volume_um3", "body volume (um^3)"),
        ("gap_fraction", "gaps recovered by closing"),
        ("enclosed_fill_fraction", "cavity filled"),
        ("residual_cavity_voxels", "UNFILLED cavity (must be 0)"),
        ("border_voxels", "voxels on volume face"),
        ("n_components", "connected components"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.0 * len(panels)), sharex=True)
    for ax, (key, label) in zip(axes, panels):
        for method, stats in methods.items():
            ax.plot([s[key] for s in stats], lw=1.2, label=method)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8, ncol=len(methods))
    axes[-1].set_xlabel("timepoint")
    fig.suptitle("Level-4 segmentation QC — look for steps, spikes, and drift", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--level", default=SEGMENT_LEVEL)
    p.add_argument("--methods", nargs="+", default=["otsu", "mad"])
    p.add_argument("--fill", default="interior", choices=["interior", "enclosed", "none"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--qc-only", action="store_true", help="write the figure, skip napari")
    args = p.parse_args()

    ds = Dataset(args.store)
    lv = ds.levels[args.level]
    voxel = lv.voxel_size_um

    params = default_conditioning()
    masks = {
        m: cached_masks(args.store, args.level, m, args.fill, args.workers, params)
        for m in args.methods
    }
    stats = {
        m: json.loads(
            (QC_DIR / f"stats_{_tag(m, args.fill, args.level, params)}.json").read_text()
        )
        for m in args.methods
    }
    qc_figure(stats, QC_DIR / f"qc_{_tag('both', args.fill, args.level, params)}.png")

    for m, st in stats.items():
        vol = np.array([s["body_volume_um3"] for s in st])
        jumps = np.abs(np.diff(vol)) / vol[:-1]
        worst = np.argsort(jumps)[::-1][:3]
        print(
            f"{m:5s}: volume {vol.min():.2e}..{vol.max():.2e} um^3, "
            f"median frame-to-frame change {np.median(jumps) * 100:.2f}%, "
            f"worst at t={[int(i) + 1 for i in worst]} "
            f"({[f'{jumps[i] * 100:.0f}%' for i in worst]})"
        )
        border = sum(1 for s in st if s["touches_border"])
        print(f"       {border}/{len(st)} timepoints touch the edge of the imaged volume")

    if args.qc_only:
        return

    import napari

    print("\nloading level-4 image into memory...")
    image = np.asarray(ds.image(args.level)[:, 0])

    viewer = napari.Viewer(title=f"segmentation review — level {args.level}")
    # scale so the anisotropic voxel (2.0 um in z, 4.16 um in xy) renders in proportion
    scale = (1.0, *voxel)
    viewer.add_image(
        image, name="image", scale=scale, contrast_limits=[32, 13462], colormap="gray"
    )
    for m in args.methods:
        viewer.add_labels(
            np.asarray(masks[m]), name=f"mask: {m}", scale=scale, opacity=0.35
        )
    if len(args.methods) == 2:
        a, b = (np.asarray(masks[m]).astype(bool) for m in args.methods)
        viewer.add_labels(
            (a ^ b).astype(np.uint8) * 3,
            name=f"disagreement ({args.methods[0]} XOR {args.methods[1]})",
            scale=scale,
            opacity=0.6,
            visible=False,
        )
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
