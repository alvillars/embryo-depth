"""Interactive depth-shell viewer.

Pick a depth, then scrub time. That asymmetry is the whole design: changing ``d`` is
occasional, changing ``t`` is constant and has to feel instant.

The naive shell is two full-volume passes per frame::

    shell = (depth >= d - w) & (depth <= d + w)
    view  = image * shell

which costs the *volume* (50M voxels at level 2) when the answer only costs the *shell*.
Since depth is already quantised to 256 uint8 values, counting-sorting the in-mask voxels
into 256 depth buckets turns a shell into a contiguous slice of a sorted array, so both
``d`` and ``t`` changes become array slicing over the ~1e5-1e6 voxels that actually appear.
At level 3 that is 0.06-0.10 ms a frame, so the shell itself is free.

**Render as a volume, not as points.** A thin shell is sparse, so a Points layer looks like
the natural fit and the shell query for it is 30x cheaper (0.06 ms against 2.2 ms). Measured
end to end it is the opposite: napari rebuilds a Points layer's internal state on every
assignment to ``data``, which costs ~200 ms for 67k points and caps time-scrubbing at 4.6 fps.
Dropping the colour mapping and updating only coordinates does not rescue it. Pushing a dense
volume to the GPU instead runs at 43 fps. Points mode is kept for comparison, but the
measurement, not the shape of the data, picks the default.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .depth import crop_bounded, touched_faces
from .depth_index import PersistedDepthIndex
from .store import DEFAULT_STORE, DEPTH_SCALE_UM, Dataset, Level

DEFAULT_LEVEL = "3"
DEFAULT_MODE = "volume"


@dataclass
class DepthIndex:
    """One timepoint's in-mask voxels, counting-sorted by quantised depth.

    ``offsets[d]`` .. ``offsets[d + 1]`` is the slice of voxels at depth unit ``d``, so the
    shell for a depth band is one contiguous range with no search and no full-volume scan.
    """

    coords: np.ndarray  # (n, 3) int16 zyx
    values: np.ndarray  # (n,) uint16 image intensity
    offsets: np.ndarray  # (257,) int64
    reliable: np.ndarray  # (n,) bool -- False where the crop plane bounds the depth
    shape: tuple[int, int, int]

    @property
    def nbytes(self) -> int:
        return self.coords.nbytes + self.values.nbytes + self.reliable.nbytes

    def band(self, lo: int, hi: int) -> slice:
        lo = int(np.clip(lo, 0, 255))
        hi = int(np.clip(hi, 0, 255))
        return slice(int(self.offsets[lo]), int(self.offsets[hi + 1]))


def build_index(
    image: np.ndarray, depth_units: np.ndarray, level: Level, mask: np.ndarray | None = None
) -> DepthIndex:
    """Counting-sort one timepoint's body voxels by depth."""
    body = depth_units > 0 if mask is None else mask.astype(bool)
    coords = np.argwhere(body).astype(np.int16)
    d = depth_units[body]
    order = np.argsort(d, kind="stable")  # counting sort over 256 values

    coords = coords[order]
    values = image[body][order].astype(np.uint16)
    counts = np.bincount(d, minlength=256)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    faces = touched_faces(body)
    if faces:
        flagged = crop_bounded(depth_units.astype(np.float32) * DEPTH_SCALE_UM, level, faces)
        reliable = ~flagged[body][order]
    else:
        reliable = np.ones(len(values), bool)
    return DepthIndex(coords, values, offsets, reliable, body.shape)


class ShellSource:
    """Serves depth shells for one level, either from a RAM-resident index built at launch
    or, when available, straight off the disk-backed ``depth_index/`` store (see
    ``depth_index.PersistedDepthIndex``) -- in which case ``.load()`` is near-instant instead
    of scanning and sorting every timepoint's full volume.
    """

    def __init__(self, dataset: Dataset, level: str, timepoints: range | None = None):
        self.ds = dataset
        self.level = level
        self.lv = dataset.levels[level]
        self.timepoints = list(timepoints or range(dataset.n_timepoints))
        self.index: dict[int, DepthIndex] = {}
        self._persisted: PersistedDepthIndex | None = None

    def load(self, verbose: bool = True) -> None:
        if self.ds.has_depth_index(self.level):
            if verbose:
                print(
                    f"  depth_index/{self.level} found -- reading from disk, "
                    "no RAM preload needed"
                )
            self._persisted = PersistedDepthIndex(self.ds, self.level)
            return

        image = self.ds.image(self.level)
        depth = self.ds.depth(self.level)
        started = time.time()
        for n, t in enumerate(self.timepoints, start=1):
            self.index[t] = build_index(
                np.asarray(image[t, 0]), np.asarray(depth[t, 0]), self.lv
            )
            if verbose and (n % 10 == 0 or n == len(self.timepoints)):
                mb = sum(i.nbytes for i in self.index.values()) / 2**20
                rate = n / (time.time() - started)
                print(
                    f"  indexed {n:3d}/{len(self.timepoints)}  {mb:7.0f} MB resident "
                    f"({rate:.1f}/s, {(len(self.timepoints) - n) / rate:.0f}s left)",
                    flush=True,
                )

    @property
    def max_depth_unit(self) -> int:
        """Deepest bucket that actually holds voxels, for the slider's upper bound.

        Must look at ``diff(offsets)``, not ``offsets``: offsets is a cumulative sum, so
        its last entry is nonzero whenever the body is non-empty and testing it directly
        would report 255 for every dataset.
        """
        if getattr(self, "_persisted", None) is not None:
            return self._persisted.max_depth_unit
        deepest = 0
        for idx in self.index.values():
            occupied = np.nonzero(np.diff(idx.offsets))[0]
            if len(occupied):
                deepest = max(deepest, int(occupied.max()))
        return deepest

    def shell(self, t: int, depth: int, half_width: int, reliable_only: bool = False):
        """Coordinates and intensities of the shell at ``depth`` +/- ``half_width``."""
        if getattr(self, "_persisted", None) is not None:
            return self._persisted.shell(t, depth, half_width, reliable_only)
        idx = self.index[t]
        sl = idx.band(depth - half_width, depth + half_width)
        coords, values = idx.coords[sl], idx.values[sl]
        if reliable_only:
            keep = idx.reliable[sl]
            coords, values = coords[keep], values[keep]
        return coords, values

    def shell_volume(self, t: int, depth: int, half_width: int, reliable_only: bool = False):
        """The same shell rendered as a dense 3D volume for napari's volume renderer."""
        if getattr(self, "_persisted", None) is not None:
            return self._persisted.shell_volume(t, depth, half_width, reliable_only)
        coords, values = self.shell(t, depth, half_width, reliable_only)
        out = np.zeros(self.index[t].shape, dtype=np.uint16)
        if len(coords):
            out[coords[:, 0], coords[:, 1], coords[:, 2]] = values
        return out


def naive_shell_volume(image, depth_units, depth: int, half_width: int) -> np.ndarray:
    """Reference implementation the index is checked against."""
    band = (depth_units >= depth - half_width) & (depth_units <= depth + half_width)
    band &= depth_units > 0
    return np.where(band, image, 0).astype(np.uint16)


def launch(
    store: Path, level: str, depth: int, half_width: int, mode: str, limit: int | None
) -> None:
    import napari

    ds = Dataset(store)
    if not ds.has_depth():
        raise SystemExit("no depth/ group yet -- run `python -m embryo_depth.depth` first")

    timepoints = range(min(limit, ds.n_timepoints)) if limit else range(ds.n_timepoints)
    src = ShellSource(ds, level, timepoints)
    lv = src.lv
    print(f"level {level} {lv.shape}, voxel {lv.voxel_size_um} um -- building depth index")
    src.load()

    scale = lv.voxel_size_um
    viewer = napari.Viewer(title=f"embryo depth shells — level {level}")
    state = {"t": 0, "d": depth, "w": half_width, "reliable_only": False}

    if mode == "volume":
        layer = viewer.add_image(
            src.shell_volume(0, depth, half_width),
            name="shell",
            scale=scale,
            colormap="gray",
            rendering="attenuated_mip",
            contrast_limits=[32, 6000],
        )
    else:
        coords, values = src.shell(0, depth, half_width)
        layer = viewer.add_points(
            coords * np.array(scale),
            name="shell",
            size=max(scale) * 1.5,
            features={"intensity": values},
            face_color="intensity",
            face_colormap="gray",
            border_width=0,
            shading="none",
        )

    def refresh():
        t, d, w = state["t"], state["d"], state["w"]
        if mode == "volume":
            layer.data = src.shell_volume(t, d, w, state["reliable_only"])
        else:
            coords, values = src.shell(t, d, w, state["reliable_only"])
            layer.data = coords * np.array(scale)
            layer.features = {"intensity": values}
            layer.face_color = "intensity"

    from magicgui import magicgui

    @magicgui(
        auto_call=True,
        timepoint={"widget_type": "Slider", "min": 0, "max": len(src.timepoints) - 1},
        depth_um={"widget_type": "Slider", "min": 0, "max": src.max_depth_unit},
        half_width_um={"widget_type": "Slider", "min": 1, "max": 20},
        hide_crop_bounded={"widget_type": "CheckBox"},
    )
    def controls(
        timepoint: int = 0,
        depth_um: int = depth,
        half_width_um: int = half_width,
        hide_crop_bounded: bool = False,
    ):
        state.update(
            t=src.timepoints[timepoint],
            d=depth_um,
            w=half_width_um,
            reliable_only=hide_crop_bounded,
        )
        refresh()

    viewer.window.add_dock_widget(controls, area="right", name="depth")
    viewer.dims.ndisplay = 3
    napari.run()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--level", default=DEFAULT_LEVEL, help="2 is sharper, 3 is faster")
    p.add_argument("--depth", type=int, default=10, help="starting depth in micrometres")
    p.add_argument("--half-width", type=int, default=3, help="shell half-thickness in um")
    p.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["volume", "points"],
        help="volume is ~10x faster to scrub; see the module docstring",
    )
    p.add_argument("--limit", type=int, default=None, help="only load the first N timepoints")
    args = p.parse_args()
    launch(args.store, args.level, args.depth, args.half_width, args.mode, args.limit)


if __name__ == "__main__":
    main()
