"""Access to the OME-Zarr acquisition store and the derived label / depth groups.

Everything here reads geometry from the store's own ``.zattrs`` rather than hardcoding
it, so the same code works on the other ``Position_*`` datasets, which differ in extent.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import zarr

#: Override with --store on any CLI entry point, or the EMBRYO_DEPTH_STORE env var.
DEFAULT_STORE = Path(os.environ.get("EMBRYO_DEPTH_STORE", "Position_10.ome.zarr"))

LABEL_NAME = "embryo"

#: Quantisation of the stored depth field. Depth is smooth, bounded, and derived from a
#: level-4 mask whose surface is only accurate to ~4 um, so a 1 um quantum is still ~4x
#: finer than the underlying segmentation and float32 would be storing noise.
#:
#: 1 um per unit gives a 0-255 um range. The headroom matters: with the crop plane treated
#: as "not a surface", depth in the deep core reaches 144 um, which a 0.5 um quantum would
#: silently saturate at 127.5. One unit also means one micrometre, so slider positions read
#: directly as depth.
DEPTH_SCALE_UM = 1.0
DEPTH_SATURATED = 255
DEPTH_BACKGROUND = 0

#: The depth field is stored for these levels only. Level 0 would be 195 GiB on its own
#: and adds no real accuracy over level 1, which is a 2x-per-axis interpolation away.
DEPTH_LEVELS = ("1", "2", "3", "4")

_TARGET_CHUNK_BYTES = 64 << 20

#: Row-axis chunk target for depth_index/, far smaller than _TARGET_CHUNK_BYTES on purpose:
#: the whole point of that store is that a narrow depth band's row range touches only a
#: handful of chunks, which one-chunk-per-timepoint (labels/depth's convention) would defeat.
_ROW_CHUNK_TARGET_BYTES = 1 << 18


@dataclass(frozen=True)
class Level:
    """One resolution level of a multiscale group."""

    path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype
    scale: tuple[float, ...]  # per-axis, order tczyx
    translation: tuple[float, ...]

    @property
    def voxel_size_um(self) -> tuple[float, float, float]:
        """Physical (z, y, x) voxel size, the ``sampling`` a distance transform needs."""
        return tuple(self.scale[-3:])  # type: ignore[return-value]

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """(z, y, x) extent of a single timepoint."""
        return tuple(self.shape[-3:])  # type: ignore[return-value]

    @property
    def n_timepoints(self) -> int:
        return self.shape[0]


def suggest_chunks(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """Chunk one timepoint into pieces of at most ``_TARGET_CHUNK_BYTES``.

    The acquisition store uses a fixed ``(1, 1, 100, 768, 768)`` at every level, which at
    level 4 is 20x larger than the whole array -- the padding compresses away but still
    has to be decompressed on every read. Splitting each axis into equal parts instead
    keeps coarse levels to a single small chunk and costs no padding.
    """
    chunk = list(shape[-3:])
    while math.prod(chunk) * itemsize > _TARGET_CHUNK_BYTES and max(chunk) > 1:
        axis = int(np.argmax(chunk))
        chunk[axis] = math.ceil(chunk[axis] / 2)
    return (1, 1, *chunk)


def suggest_row_chunks(
    max_rows: int, itemsize: int, target_bytes: int = _ROW_CHUNK_TARGET_BYTES
) -> int:
    """Row-axis chunk size for the ``depth_index/`` arrays.

    Unlike ``suggest_chunks`` (one small chunk covering a whole timepoint), this aims for
    many *small* chunks along the row axis, so that reading a narrow depth band -- a small
    contiguous row range -- only touches a handful of them instead of decompressing an
    entire timepoint's worth of body voxels.
    """
    rows = max(max_rows, 1)
    while rows * itemsize > target_bytes and rows > 1:
        rows = math.ceil(rows / 2)
    return max(rows, 1)


class Dataset:
    """The acquisition store plus the label and depth groups derived from it."""

    def __init__(self, path: Path | str = DEFAULT_STORE):
        self.path = Path(path)
        if not (self.path / ".zattrs").exists():
            raise FileNotFoundError(f"not an OME-Zarr store: {self.path}")
        self._attrs = json.loads((self.path / ".zattrs").read_text())
        self._multiscale = self._attrs["multiscales"][0]
        self.axes = self._multiscale["axes"]
        self.levels = {lv.path: lv for lv in self._read_levels()}

    def _read_levels(self) -> list[Level]:
        levels = []
        for ds in self._multiscale["datasets"]:
            meta = json.loads((self.path / ds["path"] / ".zarray").read_text())
            scale = translation = None
            for tf in ds["coordinateTransformations"]:
                if tf["type"] == "scale":
                    scale = tuple(tf["scale"])
                elif tf["type"] == "translation":
                    translation = tuple(tf["translation"])
            if scale is None:
                raise ValueError(f"level {ds['path']} has no scale transformation")
            levels.append(
                Level(
                    path=ds["path"],
                    shape=tuple(meta["shape"]),
                    chunks=tuple(meta["chunks"]),
                    dtype=np.dtype(meta["dtype"]),
                    scale=scale,
                    translation=translation or (0.0,) * len(scale),
                )
            )
        return levels

    @property
    def finest(self) -> Level:
        return self.levels[self._multiscale["datasets"][0]["path"]]

    @property
    def coarsest(self) -> Level:
        return self.levels[self._multiscale["datasets"][-1]["path"]]

    @property
    def n_timepoints(self) -> int:
        return self.finest.n_timepoints

    def coordinate_transformations(self, level: str) -> list[dict]:
        """The transformations for ``level``, copied verbatim from the image metadata.

        Derived multiscales must reuse these exactly or napari will misregister the
        overlay against the image.
        """
        for ds in self._multiscale["datasets"]:
            if ds["path"] == level:
                return json.loads(json.dumps(ds["coordinateTransformations"]))
        raise KeyError(f"no such level: {level}")

    # -- array access -----------------------------------------------------------------

    def image(self, level: str) -> zarr.Array:
        return zarr.open(str(self.path / level), mode="r")

    def label(self, level: str, mode: str = "r") -> zarr.Array:
        return zarr.open(str(self.path / "labels" / LABEL_NAME / level), mode=mode)

    def depth(self, level: str, mode: str = "r") -> zarr.Array:
        return zarr.open(str(self.path / "depth" / level), mode=mode)

    def depth_index(self, level: str, name: str, mode: str = "r") -> zarr.Array:
        """One of the four arrays (``coords``, ``values``, ``reliable``, ``offsets``) that
        make up ``depth_index/{level}`` -- see ``create_depth_index_group``."""
        return zarr.open(str(self.path / "depth_index" / level / name), mode=mode)

    def has_labels(self) -> bool:
        return (self.path / "labels" / LABEL_NAME / ".zattrs").exists()

    def has_depth(self) -> bool:
        return (self.path / "depth" / ".zattrs").exists()

    def has_depth_index(self, level: str) -> bool:
        return (self.path / "depth_index" / level / ".zattrs").exists()

    # -- creating the derived groups --------------------------------------------------

    def create_label_group(self, levels: list[str], overwrite: bool = False) -> None:
        """Create ``labels/embryo`` as an NGFF label multiscale covering ``levels``.

        Additive: no existing key in the acquisition store is touched.
        """
        root = zarr.open_group(str(self.path), mode="a")
        labels_group = root.require_group("labels")

        existing = list(labels_group.attrs.get("labels", []))
        if LABEL_NAME not in existing:
            labels_group.attrs["labels"] = existing + [LABEL_NAME]

        group = labels_group.require_group(LABEL_NAME)
        for level in levels:
            src = self.levels[level]
            group.require_dataset(
                level,
                shape=src.shape,
                chunks=suggest_chunks(src.shape, 1),
                dtype="u1",
                compressor=numcodecs.Zstd(level=5),
                fill_value=0,
                write_empty_chunks=False,
                # Match the acquisition arrays. Both separators are valid and zarr records
                # the choice in .zarray, but this store is shared, and "/" also keeps level
                # 0 as a nested tree rather than 4224 chunk files in one directory.
                dimension_separator="/",
                overwrite=overwrite,
                exact=True,
            )

        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": LABEL_NAME,
                "axes": self.axes,
                "datasets": [
                    {"path": lv, "coordinateTransformations": self.coordinate_transformations(lv)}
                    for lv in levels
                ],
            }
        ]
        group.attrs["image-label"] = {
            "version": "0.4",
            "colors": [{"label-value": 1, "rgba": [255, 128, 0, 128]}],
            "properties": [{"label-value": 1, "name": "embryo"}],
            "source": {"image": "../../"},
        }

    def create_depth_group(
        self, levels: list[str] = list(DEPTH_LEVELS), overwrite: bool = False
    ) -> None:
        """Create the ``depth/`` group holding the quantised distance field.

        This is a custom extension, not NGFF-standard: a distance field is continuous
        rather than a set of object IDs, so putting it under ``labels/`` would break
        ``image-label`` semantics. Readers look only at the root ``multiscales`` and
        ``labels/``, so a sibling group is inert to them. Level paths deliberately match
        the image pyramid indices so a level index never means two different things.
        """
        root = zarr.open_group(str(self.path), mode="a")
        group = root.require_group("depth")
        for level in levels:
            src = self.levels[level]
            group.require_dataset(
                level,
                shape=src.shape,
                chunks=suggest_chunks(src.shape, 1),
                dtype="u1",
                compressor=numcodecs.Zstd(level=5),
                fill_value=0,
                write_empty_chunks=False,
                # Match the acquisition arrays. Both separators are valid and zarr records
                # the choice in .zarray, but this store is shared, and "/" also keeps level
                # 0 as a nested tree rather than 4224 chunk files in one directory.
                dimension_separator="/",
                overwrite=overwrite,
                exact=True,
            )

        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "depth",
                "axes": self.axes,
                "datasets": [
                    {"path": lv, "coordinateTransformations": self.coordinate_transformations(lv)}
                    for lv in levels
                ],
            }
        ]
        group.attrs["depth_quantization"] = {
            "scale_um_per_unit": DEPTH_SCALE_UM,
            "saturation_value": DEPTH_SATURATED,
            "background_value": DEPTH_BACKGROUND,
            "description": (
                "Euclidean distance from the embryo surface, in micrometres, as "
                "round(depth_um / scale_um_per_unit). 0 means outside the mask; the "
                "labels/embryo array is authoritative for inside/outside."
            ),
        }

    def depth_quantization(self) -> dict:
        attrs = json.loads((self.path / "depth" / ".zattrs").read_text())
        return attrs["depth_quantization"]

    def create_depth_index_group(
        self,
        levels: list[str],
        max_body: dict[str, int],
        overwrite: bool = False,
        row_chunk_target_bytes: int = _ROW_CHUNK_TARGET_BYTES,
    ) -> None:
        """Create ``depth_index/{level}/{coords,values,reliable,offsets}``.

        A custom extension, like ``depth/``: this is the on-disk equivalent of the
        counting-sorted-by-depth index ``viewer.py``'s ``build_index`` builds in RAM at
        every session (see ``depth_index.py`` for the write pass that populates it).
        Persisting it, chunked in small row-blocks, turns depth into a queryable axis the
        same way z/y/x already are for the acquisition arrays: a depth-band query becomes a
        lookup in the small ``offsets`` table plus one small chunked read, not a full-volume
        scan. ``max_body`` is each level's widest per-timepoint body-voxel count (from
        ``labels/embryo/{level}``), which sizes the row axis before any data is written --
        rows beyond a given timepoint's real body-voxel count are left at ``fill_value`` and
        compress away.

        ``coords``: ``(n_t, max_body, 3)`` int16, body voxel (z,y,x), sorted by depth within
        each timepoint's row-span.
        ``values``: ``(n_t, max_body)`` uint16, image intensity at that voxel -- stored
        redundantly rather than re-touching the (differently chunked) base image array at
        query time.
        ``reliable``: ``(n_t, max_body)`` uint8 (bool), ``~crop_bounded(...)``, same order.
        ``offsets``: ``(n_t, 257)`` int64. ``offsets[t, d]`` is the row where depth-bucket
        ``d`` begins within timepoint ``t``'s span (``offsets[t, 256]`` is that timepoint's
        valid row count). Tiny (~270 KB total across all timepoints) -- meant to be read
        fully into memory and kept resident, the same role arithmetic chunk-boundary math
        plays for z, since these boundaries are irregular (the body grows over the series)
        and so cannot be computed from shape alone.
        """
        root = zarr.open_group(str(self.path), mode="a")
        depth_index_group = root.require_group("depth_index")
        for level in levels:
            src = self.levels[level]
            n_t = src.n_timepoints
            body = max_body[level]
            # sized off the widest of the three dtypes (coords is int16 but has 3 components
            # per row; values/reliable are narrower, so 2 bytes/row is the binding one)
            row_chunk = suggest_row_chunks(body, 2, row_chunk_target_bytes)
            group = depth_index_group.require_group(level)
            group.require_dataset(
                "coords",
                shape=(n_t, body, 3),
                chunks=(1, row_chunk, 3),
                dtype="i2",
                compressor=numcodecs.Zstd(level=5),
                fill_value=0,
                write_empty_chunks=False,
                dimension_separator="/",
                overwrite=overwrite,
                exact=True,
            )
            group.require_dataset(
                "values",
                shape=(n_t, body),
                chunks=(1, row_chunk),
                dtype="u2",
                compressor=numcodecs.Zstd(level=5),
                fill_value=0,
                write_empty_chunks=False,
                dimension_separator="/",
                overwrite=overwrite,
                exact=True,
            )
            group.require_dataset(
                "reliable",
                shape=(n_t, body),
                chunks=(1, row_chunk),
                dtype="u1",
                compressor=numcodecs.Zstd(level=5),
                fill_value=0,
                write_empty_chunks=False,
                dimension_separator="/",
                overwrite=overwrite,
                exact=True,
            )
            group.require_dataset(
                "offsets",
                shape=(n_t, 257),
                chunks=(n_t, 257),
                dtype="i8",
                compressor=numcodecs.Zstd(level=5),
                fill_value=0,
                write_empty_chunks=False,
                dimension_separator="/",
                overwrite=overwrite,
                exact=True,
            )
            group.attrs["max_body_voxels"] = body

        depth_index_group.attrs["depth_index"] = {
            "version": "custom-1",
            "description": (
                "Per-timepoint body voxels sorted by quantised depth, mirroring viewer.py's "
                "DepthIndex but persisted and chunked on disk. offsets[t, d]..offsets[t, d+1] "
                "is the row range for depth bucket d within timepoint t's span; rows beyond "
                "offsets[t, 256] are unwritten padding up to that level's max_body_voxels."
            ),
            "depth_scale_um_per_unit": DEPTH_SCALE_UM,
        }


def quantize_depth(edt_um: np.ndarray) -> np.ndarray:
    """Convert a distance transform in micrometres to the stored ``uint8`` encoding."""
    units = np.rint(edt_um / DEPTH_SCALE_UM)
    return np.clip(units, 0, DEPTH_SATURATED).astype(np.uint8)


def dequantize_depth(stored: np.ndarray) -> np.ndarray:
    """Convert the stored encoding back to micrometres."""
    return stored.astype(np.float32) * DEPTH_SCALE_UM


def um_to_units(depth_um: float) -> int:
    """Depth in micrometres to the stored integer unit, for slider bounds."""
    return int(np.clip(round(depth_um / DEPTH_SCALE_UM), 0, DEPTH_SATURATED))
