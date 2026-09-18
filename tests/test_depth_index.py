"""Tests for depth_index.py: the persisted, chunk-lazy on-disk equivalent of viewer.py's
in-RAM DepthIndex.

Unlike test_pipeline.py's upscaling/depth tests, which operate entirely on in-memory arrays,
these exercise the real thing on a tiny synthetic OME-Zarr store on disk -- there is no
lighter-weight way to test "does reading a narrow depth band actually touch a bounded number
of on-disk chunks", which is the whole point of this module.
"""

import numcodecs
import numpy as np
import pytest
import zarr
from zarr.storage import DirectoryStore

from embryo_depth.depth import surface_distance_um
from embryo_depth.depth_index import PersistedDepthIndex, _max_body_voxels, _worker
from embryo_depth.store import Dataset, quantize_depth
from embryo_depth.viewer import build_index

ANISOTROPIC = (2.0, 0.52, 0.52)
AXES = [
    {"name": "t", "type": "time"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def _sphere(shape, radius_um, voxel_um, centre=None):
    centre = centre or [s / 2 for s in shape]
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    r2 = sum(((g - c) * v) ** 2 for g, c, v in zip(grids, centre, voxel_um))
    return r2 <= radius_um**2


@pytest.fixture
def synth_store(tmp_path):
    """A tiny, real on-disk OME-Zarr store, built the same way segment.py/depth.py build
    the real one: one level, a few timepoints, labels/embryo and depth/ already populated --
    the state depth_index.py expects to run against.
    """
    n_t = 3
    shape = (24, 40, 40)
    path = tmp_path / "synth.ome.zarr"

    root = zarr.open_group(str(path), mode="a")
    arr = root.require_dataset(
        "1",
        shape=(n_t, 1, *shape),
        chunks=(1, 1, *shape),
        dtype="u2",
        compressor=numcodecs.Zstd(level=5),
        dimension_separator="/",
    )
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": AXES,
            "datasets": [
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, *ANISOTROPIC]},
                        {"type": "translation", "translation": [0.0, 0.0, 0.0, 0.0, 0.0]},
                    ],
                }
            ],
        }
    ]

    ds = Dataset(path)
    ds.create_label_group(["1"])
    ds.create_depth_group(["1"])

    rng = np.random.default_rng(0)
    for t in range(n_t):
        # Radius grows across timepoints (so body size varies, exercising the sizing pass
        # for real) and the last timepoint's sphere is centred on z=0 (so its cavity pokes
        # past the crop plane, exercising the reliable/crop_bounded path too).
        radius = 8.0 + 2.0 * t
        centre = [0, shape[1] / 2, shape[2] / 2] if t == n_t - 1 else None
        mask = _sphere(shape, radius, ANISOTROPIC, centre)
        depth_um = surface_distance_um(mask, ANISOTROPIC)
        quantized = quantize_depth(depth_um)
        image = rng.integers(100, 4000, shape).astype(np.uint16)
        image[~mask] = 0

        arr[t, 0] = image
        ds.label("1", "a")[t, 0] = mask.astype(np.uint8)
        ds.depth("1", "a")[t, 0] = quantized

    return path, n_t


def _populate_depth_index(store_path, level="1", row_chunk_target_bytes=None):
    ds = Dataset(store_path)
    max_body = {level: _max_body_voxels(ds, level)}
    kwargs = {} if row_chunk_target_bytes is None else {
        "row_chunk_target_bytes": row_chunk_target_bytes
    }
    ds.create_depth_index_group([level], max_body, **kwargs)
    for t in range(ds.n_timepoints):
        _worker((t, store_path, level))
    return ds, max_body[level]


def test_depth_index_offsets_partition_body_voxels(synth_store):
    store_path, n_t = synth_store
    ds, _ = _populate_depth_index(store_path)
    offsets = np.asarray(ds.depth_index("1", "offsets"))
    depth = ds.depth("1")
    for t in range(n_t):
        assert np.all(np.diff(offsets[t]) >= 0)
        assert offsets[t, 0] == 0
        # offsets[t, 256] is the count of voxels build_index_rows treats as "body"
        # (depth_units > 0), which is the depth field's own definition, not the label mask's
        # -- a thin rim of surface voxels can quantise to exactly 0 and be excluded from
        # both, matching viewer.build_index's existing behaviour.
        assert offsets[t, 256] == int(np.count_nonzero(np.asarray(depth[t, 0]) > 0))


def test_depth_index_band_matches_inram_build_index(synth_store):
    """The key regression test: the disk-backed path must reproduce the in-RAM path exactly."""
    store_path, n_t = synth_store
    ds, _ = _populate_depth_index(store_path)
    persisted = PersistedDepthIndex(ds, "1")
    lv = ds.levels["1"]

    for t in range(n_t):
        image = np.asarray(ds.image("1")[t, 0])
        depth_units = np.asarray(ds.depth("1")[t, 0])
        in_ram = build_index(image, depth_units, lv)

        for depth, half_width in [(0, 1), (3, 2), (6, 3), (9, 1), (2, 6)]:
            sl = in_ram.band(depth - half_width, depth + half_width)
            expected_coords = in_ram.coords[sl]
            expected_values = in_ram.values[sl]
            expected_reliable = in_ram.reliable[sl]

            got_coords, got_values, got_reliable = persisted.band(
                t, depth - half_width, depth + half_width
            )
            np.testing.assert_array_equal(got_coords, expected_coords)
            np.testing.assert_array_equal(got_values, expected_values)
            np.testing.assert_array_equal(got_reliable, expected_reliable)

            got_vol = persisted.shell_volume(t, depth, half_width)
            expected_vol = np.zeros(in_ram.shape, dtype=np.uint16)
            if len(expected_coords):
                expected_vol[
                    expected_coords[:, 0], expected_coords[:, 1], expected_coords[:, 2]
                ] = expected_values
            np.testing.assert_array_equal(got_vol, expected_vol)

    deepest = 0
    for t in range(n_t):
        idx = build_index(np.asarray(ds.image("1")[t, 0]), np.asarray(ds.depth("1")[t, 0]), lv)
        occupied = np.nonzero(np.diff(idx.offsets))[0]
        if len(occupied):
            deepest = max(deepest, int(occupied.max()))
    assert persisted.max_depth_unit == deepest


class _CountingStore:
    """Wraps a MutableMapping store to count __getitem__ calls, i.e. chunk reads."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __getitem__(self, key):
        self.calls += 1
        return self.inner[key]

    def __setitem__(self, key, value):
        self.inner[key] = value

    def __delitem__(self, key):
        del self.inner[key]

    def __contains__(self, key):
        return key in self.inner

    def __iter__(self):
        return iter(self.inner)

    def __len__(self):
        return len(self.inner)

    def keys(self):
        return self.inner.keys()

    def values(self):
        return self.inner.values()

    def get(self, key, default=None):
        return self.inner.get(key, default)


def test_depth_index_band_read_touches_bounded_chunks(synth_store):
    """The property this whole design is for: a narrow depth band must not require reading
    (let alone decompressing) a whole timepoint's worth of body voxels.

    Forces a tiny row-chunk target so the synthetic fixture's small body actually gets
    sub-chunked -- with the real dataset's default 256 KiB target this happens naturally,
    but a fast unit-test fixture is far smaller than that threshold.
    """
    store_path, n_t = synth_store
    ds, max_body = _populate_depth_index(store_path, row_chunk_target_bytes=64)
    assert max_body > 200  # otherwise this test isn't exercising real sub-chunking

    counting = _CountingStore(DirectoryStore(str(store_path)))
    coords = zarr.open_array(store=counting, path="depth_index/1/coords", mode="r")

    counting.calls = 0
    _ = coords[0, 0:5]
    touched_narrow = counting.calls
    assert touched_narrow > 0

    counting.calls = 0
    _ = coords[0, :]
    touched_wide = counting.calls

    assert touched_narrow < touched_wide
    assert touched_narrow <= 4


def test_create_depth_index_group_matches_depth_group_conventions(synth_store):
    store_path, _ = synth_store
    ds, _ = _populate_depth_index(store_path)

    depth_arr = ds.depth("1")
    coords_arr = ds.depth_index("1", "coords")
    assert coords_arr.compressor.__class__ == depth_arr.compressor.__class__
    assert coords_arr._meta["dimension_separator"] == "/"
    assert ds.has_depth_index("1")
    assert not ds.has_depth_index("2")


# --- napari reader -------------------------------------------------------------------


def test_reader_returns_none_without_depth_index(synth_store):
    from embryo_depth.reader import napari_get_reader

    store_path, _ = synth_store
    assert napari_get_reader(str(store_path)) is None


def test_reader_returns_callable_once_depth_index_exists(synth_store):
    from embryo_depth.reader import napari_get_reader

    store_path, _ = synth_store
    _populate_depth_index(store_path)
    reader = napari_get_reader(str(store_path))
    assert callable(reader)


def test_read_depth_axis_returns_expected_layer_data(synth_store):
    from embryo_depth.reader import DEFAULT_HALF_WIDTH_UM, N_DEPTH_BUCKETS, _read_depth_axis

    store_path, n_t = synth_store
    _populate_depth_index(store_path)

    layers = _read_depth_axis(str(store_path))
    assert len(layers) == 1
    data, meta, layer_type = layers[0]
    assert layer_type == "image"
    assert data.shape == (n_t, N_DEPTH_BUCKETS, 24, 40, 40)
    assert data.ndim == 5
    assert meta["scale"] == (1.0, 1.0, *ANISOTROPIC)

    # a concrete (t, d) index must match PersistedDepthIndex.shell_volume exactly
    ds = Dataset(store_path)
    persisted = PersistedDepthIndex(ds, "1")
    expected = persisted.shell_volume(1, 10, DEFAULT_HALF_WIDTH_UM)
    np.testing.assert_array_equal(np.asarray(data[1, 10]), expected)


def test_depth_shell_array_rejects_non_concrete_td_index(synth_store):
    from embryo_depth.reader import DepthShellArray

    store_path, n_t = synth_store
    ds, _ = _populate_depth_index(store_path)
    arr = DepthShellArray(PersistedDepthIndex(ds, "1"), n_t, 3)
    with pytest.raises(IndexError):
        arr[:, 10]
