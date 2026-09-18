"""npe2 reader: exposes depth as a normal navigable napari axis.

This is the literal "drop the file in, scrub depth like z" path the user originally wanted:
handing napari an array-like object of shape ``(t, d, z, y, x)`` gets an automatic ``t``
slider and an automatic ``d`` slider for free, the same way napari already gives one for any
plain numpy/dask/zarr array -- no custom magicgui UI code needed here.

It relies on ``depth_index.PersistedDepthIndex`` for the actual reads, so a depth-band query
touches only the on-disk chunks overlapping that band (see ``depth_index.py`` and
``store.create_depth_index_group``) rather than scanning a full volume or requiring
everything to be preloaded into RAM first.

``viewer.py``/``plugin.py``'s hand-rolled sliders remain the benchmarked, fast default for a
real scrubbing session (see their module docstrings) -- this reader is an additional,
standards-shaped on-ramp, not a replacement. Requires ``depth_index/{level}`` to already
exist; this module only reads it, it does not build it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .depth_index import PersistedDepthIndex
from .store import Dataset
from .viewer import DEFAULT_LEVEL

#: Depth axis is fixed to the stored quantisation range (store.DEPTH_SATURATED + 1).
N_DEPTH_BUCKETS = 256

#: Band half-width is a construction parameter, not an axis -- turning it into one would mean
#: shipping ~20 near-duplicate volumes per timepoint. Matches viewer.py's own default.
DEFAULT_HALF_WIDTH_UM = 3


class DepthShellArray:
    """Array-like ``(t, d, z, y, x)`` view over a ``PersistedDepthIndex``.

    Implements just enough of the array protocol for napari's automatic dimension sliders:
    ``.shape``, ``.ndim``, ``.dtype`` and basic ``__getitem__``. Each ``__getitem__`` touches
    only the on-disk chunks overlapping the requested ``(t, d)`` -- see
    ``PersistedDepthIndex.band``.
    """

    def __init__(self, persisted: PersistedDepthIndex, n_timepoints: int, half_width_um: int):
        self._persisted = persisted
        self.half_width_um = half_width_um
        self.shape = (n_timepoints, N_DEPTH_BUCKETS, *persisted.shape)
        self.ndim = 5
        self.dtype = np.dtype("uint16")

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) < 2 or not all(isinstance(k, (int, np.integer)) for k in key[:2]):
            raise IndexError(
                "DepthShellArray only supports indexing a concrete (t, d) pair, "
                "e.g. arr[3, 40] or arr[3, 40, :, :, :]"
            )
        t, d = int(key[0]), int(key[1])
        vol = self._persisted.shell_volume(t, d, self.half_width_um)
        rest = key[2:]
        return vol[rest] if rest else vol


def napari_get_reader(path):
    """npe2 reader hook. Recognised only once ``depth_index/`` has actually been built --
    see ``depth_index.build_depth_index`` -- so this never claims a store it cannot serve.
    """
    if isinstance(path, list):
        return None
    p = Path(path)
    if not (p / "depth_index" / ".zgroup").exists():
        return None
    return _read_depth_axis


def _read_depth_axis(path):
    ds = Dataset(path)
    if ds.has_depth_index(DEFAULT_LEVEL):
        # Same fast default viewer.py picks for scrubbing (see its module docstring).
        level = DEFAULT_LEVEL
    else:
        try:
            level = next(lv for lv in sorted(ds.levels, key=int) if ds.has_depth_index(lv))
        except StopIteration:
            return []

    persisted = PersistedDepthIndex(ds, level)
    arr = DepthShellArray(persisted, ds.n_timepoints, DEFAULT_HALF_WIDTH_UM)
    meta = {
        "name": f"embryo depth shells (level {level})",
        "scale": (1.0, 1.0, *ds.levels[level].voxel_size_um),
        "colormap": "gray",
        "rendering": "attenuated_mip",
        "contrast_limits": [32, 6000],
    }
    return [(arr, meta, "image")]
