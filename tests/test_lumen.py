"""Tests for lumen.py (lumen/tissue segmentation) and the store.py/upscale.py changes that
let a second and third label group (``lumen``, ``tissue``) live alongside ``labels/embryo``.
"""

import json

import numcodecs
import numpy as np
import pytest
import zarr

from embryo_depth.lumen import lumen_threshold, segment_lumen_timepoint
from embryo_depth.store import Dataset
from embryo_depth.upscale import build_pyramid

ANISOTROPIC = (2.0, 0.52, 0.52)
AXES = [
    {"name": "t", "type": "time"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def sphere(shape, radius_um, voxel_um, centre=None):
    centre = centre or [s / 2 for s in shape]
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    r2 = sum(((g - c) * v) ** 2 for g, c, v in zip(grids, centre, voxel_um))
    return r2 <= radius_um**2


# --- synthetic embryo with a lumen, a bright shell, and a small decoy dark blob ----
#
# lumen.py runs at LUMEN_LEVEL == segment.py's SEGMENT_LEVEL ("4"), whose voxel is coarse
# (2.0, 4.16, 4.16) um, not level 1's fine (2.0, 0.52, 0.52). Using the fine voxel size here
# would make the default 12 um closing/opening footprint ~45 voxels across instead of ~6,
# which is both unrealistic and pathologically slow.
#
# The cavity/shell/decoy radii are picked well above the default 12 um closing/opening scale
# (segment.py's own module docstring: smoothing alone shrinks a 30 um sphere by 12% at sigma
# = 6 um -- a structure sized close to the conditioning radius gets eroded away almost
# entirely, which would make this fixture test conditioning collapse, not component
# selection).
LEVEL4_VOXEL = (2.0, 4.16, 4.16)
SHAPE = (100, 50, 50)
CENTRE = [s / 2 for s in SHAPE]
DECOY_CENTRE = [50, 39, 25]  # well inside the embryo, ~58 um from centre in y


def _synthetic_volume(seed=0):
    embryo_mask = sphere(SHAPE, 80.0, LEVEL4_VOXEL, CENTRE)
    cavity_mask = sphere(SHAPE, 40.0, LEVEL4_VOXEL, CENTRE)
    decoy_mask = sphere(SHAPE, 8.0, LEVEL4_VOXEL, DECOY_CENTRE)
    assert not (cavity_mask & decoy_mask).any()  # decoy must not touch the true cavity
    assert np.all(embryo_mask[decoy_mask])  # decoy must sit inside the embryo
    assert cavity_mask.sum() > 4 * decoy_mask.sum()  # decoy must really be the smaller one

    # Real intensities are noisy (segment.py: background ~130, cavity ~500), not two exact
    # values -- a degenerate two-value histogram makes Otsu land exactly on the boundary.
    rng = np.random.default_rng(seed)
    vol = np.clip(rng.normal(130, 15, SHAPE), 0, None).astype(np.uint16)
    shell = embryo_mask & ~cavity_mask & ~decoy_mask
    vol[shell] = np.clip(rng.normal(1000, 60, shell.sum()), 0, None).astype(np.uint16)
    dark = cavity_mask | decoy_mask
    vol[dark] = np.clip(rng.normal(400, 30, dark.sum()), 0, None).astype(np.uint16)
    return vol, embryo_mask, cavity_mask, decoy_mask


def test_lumen_threshold_separates_dark_interior_from_bright_shell():
    vol, embryo_mask, cavity_mask, decoy_mask = _synthetic_volume()
    threshold = lumen_threshold(vol, embryo_mask)
    assert 300 < threshold < 700
    dark = embryo_mask & (vol < threshold)
    expected = cavity_mask | decoy_mask
    # noise can misclassify a handful of voxels near the tails; the bulk must still match
    mismatched = dark ^ expected
    assert mismatched.sum() < 0.02 * expected.sum()


def test_segment_lumen_keeps_largest_component_and_drops_a_decoy():
    vol, embryo_mask, cavity_mask, decoy_mask = _synthetic_volume()
    lumen_mask, tissue_mask, stats = segment_lumen_timepoint(
        vol, embryo_mask.astype(np.uint8), LEVEL4_VOXEL
    )
    lumen_bool = lumen_mask.astype(bool)
    # the true (larger) cavity survives...
    assert lumen_bool[tuple(int(c) for c in CENTRE)]
    # ...but the smaller, disconnected decoy does not
    assert not lumen_bool[tuple(int(c) for c in DECOY_CENTRE)]
    assert stats["n_components"] >= 1
    assert not stats["too_small"]


def test_segment_lumen_reports_lumen_found_for_a_true_central_cavity():
    vol, embryo_mask, cavity_mask, decoy_mask = _synthetic_volume()
    _, _, stats = segment_lumen_timepoint(vol, embryo_mask.astype(np.uint8), LEVEL4_VOXEL)
    assert stats["lumen_found"]


def test_select_lumen_component_returns_empty_when_no_candidate_is_central():
    """No dark region at the embryo's own deepest point -> empty mask, not a guess.

    This is the real failure mode hit on early Position_10 timepoints: before a lumen has
    opened up, nothing dark occupies the embryo's geometric core, and picking "the largest
    dark blob anyway" would silently return a peripheral cell instead of admitting there is
    no lumen yet.
    """
    from embryo_depth.lumen import _select_lumen_component

    embryo_mask = sphere(SHAPE, 80.0, LEVEL4_VOXEL, CENTRE)
    # an off-centre dark blob, far enough from the true centre that it cannot contain the
    # embryo's own deepest interior point (which sits at/near CENTRE by symmetry)
    off_centre = sphere(SHAPE, 10.0, LEVEL4_VOXEL, [50, 44, 25])
    assert not off_centre[tuple(int(c) for c in CENTRE)]

    mask, n, found = _select_lumen_component(off_centre, embryo_mask, LEVEL4_VOXEL)
    assert not found
    assert n >= 1  # there was a real candidate, it just wasn't central
    assert not mask.any()


def test_tissue_mask_is_embryo_minus_lumen():
    vol, embryo_mask, cavity_mask, decoy_mask = _synthetic_volume()
    lumen_mask, tissue_mask, _ = segment_lumen_timepoint(
        vol, embryo_mask.astype(np.uint8), LEVEL4_VOXEL
    )
    expected_tissue = embryo_mask & ~lumen_mask.astype(bool)
    assert np.array_equal(tissue_mask.astype(bool), expected_tissue)
    assert not (lumen_mask.astype(bool) & tissue_mask.astype(bool)).any()


def test_lumen_mask_stays_within_the_embryo():
    vol, embryo_mask, cavity_mask, decoy_mask = _synthetic_volume()
    lumen_mask, _, _ = segment_lumen_timepoint(vol, embryo_mask.astype(np.uint8), LEVEL4_VOXEL)
    assert np.all(embryo_mask[lumen_mask.astype(bool)])


# --- store.py: labels/{name} must coexist without clobbering each other ------------


@pytest.fixture
def synth_store(tmp_path):
    n_t = 2
    shape = (16, 24, 24)
    path = tmp_path / "synth.ome.zarr"
    root = zarr.open_group(str(path), mode="a")
    root.require_dataset(
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
    return path, n_t, shape


def test_label_groups_with_different_names_coexist(synth_store):
    path, n_t, shape = synth_store
    ds = Dataset(path)
    ds.create_label_group(["1"], name="embryo")
    ds.create_label_group(["1"], name="lumen")

    embryo_mask = np.ones(shape, dtype=np.uint8)
    lumen_mask = np.zeros(shape, dtype=np.uint8)
    lumen_mask[4:8, 8:16, 8:16] = 1
    ds.label("1", mode="a", name="embryo")[0, 0] = embryo_mask
    ds.label("1", mode="a", name="lumen")[0, 0] = lumen_mask

    assert ds.has_labels(name="embryo")
    assert ds.has_labels(name="lumen")
    assert not ds.has_labels(name="tissue")
    np.testing.assert_array_equal(np.asarray(ds.label("1", name="embryo")[0, 0]), embryo_mask)
    np.testing.assert_array_equal(np.asarray(ds.label("1", name="lumen")[0, 0]), lumen_mask)


def test_default_label_name_still_reads_embryo(synth_store):
    """Positional/default callers (segment.py, upscale.py, viewer.py, ...) must be unaffected."""
    path, n_t, shape = synth_store
    ds = Dataset(path)
    ds.create_label_group(["1"])  # no name given -> defaults to "embryo"
    ds.label("1", "a")[0, 0] = 1  # mode positional, as depth_index's fixture already relies on
    assert ds.has_labels()
    np.testing.assert_array_equal(np.asarray(ds.label("1")[0, 0]), np.ones(shape, dtype=np.uint8))


# --- upscale.py: --label threads through to store.py without touching other labels -


@pytest.fixture
def synth_pyramid_store(tmp_path):
    n_t = 1
    shape_src = (8, 8, 8)
    shape_tgt = (8, 16, 16)  # xy magnified 2x, z untouched -- mirrors level4->1 in miniature
    path = tmp_path / "pyr.ome.zarr"
    root = zarr.open_group(str(path), mode="a")
    root.require_dataset(
        "1",
        shape=(n_t, 1, *shape_src),
        chunks=(1, 1, *shape_src),
        dtype="u2",
        compressor=numcodecs.Zstd(level=5),
        dimension_separator="/",
    )
    root.require_dataset(
        "0",
        shape=(n_t, 1, *shape_tgt),
        chunks=(1, 1, *shape_tgt),
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
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 2.0, 0.5, 0.5]},
                        {"type": "translation", "translation": [0.0, 0.0, 0.0, 0.0, 0.0]},
                    ],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 2.0, 1.0, 1.0]},
                        {"type": "translation", "translation": [0.0, 0.0, 0.0, 0.0, 0.0]},
                    ],
                },
            ],
        }
    ]
    ds = Dataset(path)
    ds.create_label_group(["1"], name="lumen")
    mask = sphere((8, 8, 8), 3.0, (2.0, 1.0, 1.0))
    ds.label("1", mode="a", name="lumen")[0, 0] = mask.astype(np.uint8)
    return path, shape_tgt


def test_upscale_label_name_only_touches_its_own_label_group(synth_pyramid_store):
    path, shape_tgt = synth_pyramid_store
    build_pyramid(path, source_level="1", target_levels=["0"], workers=1, label_name="lumen")

    ds = Dataset(path)
    assert ds.has_labels(name="lumen")
    assert not ds.has_labels(name="embryo")
    out = np.asarray(ds.label("0", name="lumen")[0, 0])
    assert out.shape == shape_tgt
    assert out.sum() > 0


def test_upscale_multiclass_prediction_keeps_classes_and_image_label_metadata(synth_pyramid_store):
    path, shape_tgt = synth_pyramid_store
    ds = Dataset(path)
    ds.create_label_group(["1"], name="prediction")
    grp = zarr.open_group(str(path / "labels" / "prediction"), mode="a")
    custom = {
        "version": "0.4",
        "colors": [
            {"label-value": 1, "rgba": [255, 128, 0, 128]},
            {"label-value": 2, "rgba": [0, 160, 255, 128]},
        ],
        "properties": [
            {"label-value": 1, "name": "epiblast"},
            {"label-value": 2, "name": "lumen"},
        ],
        "source": {"image": "../../"},
    }
    grp.attrs["image-label"] = custom
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[sphere((8, 8, 8), 3.5, (2.0, 1.0, 1.0))] = 1
    labels[sphere((8, 8, 8), 1.5, (2.0, 1.0, 1.0))] = 2
    ds.label("1", mode="a", name="prediction")[0, 0] = labels

    build_pyramid(path, source_level="1", target_levels=["0"], workers=1, label_name="prediction")

    out = np.asarray(Dataset(path).label("0", name="prediction")[0, 0])
    assert set(np.unique(out)) == {0, 1, 2}
    attrs = json.loads((path / "labels" / "prediction" / ".zattrs").read_text())
    assert attrs["image-label"] == custom
    assert [d["path"] for d in attrs["multiscales"][0]["datasets"]] == ["0", "1"]


def test_label_group_has_singleton_channel_on_multichannel_image(tmp_path):
    """A label is one class map: c=1 even when the image has several channels."""
    shape = (4, 6, 6)
    path = tmp_path / "two_channel.ome.zarr"
    root = zarr.open_group(str(path), mode="a")
    root.require_dataset("1", shape=(2, 2, *shape), chunks=(1, 1, *shape), dtype="u2",
                         compressor=numcodecs.Zstd(level=5), dimension_separator="/")
    root.attrs["multiscales"] = [{
        "version": "0.4", "axes": AXES,
        "datasets": [{"path": "1", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0, *ANISOTROPIC]},
            {"type": "translation", "translation": [0.0] * 5}]}],
    }]
    ds = Dataset(path)
    ds.create_label_group(["1"], name="prediction")
    assert ds.label("1", name="prediction").shape == (2, 1, *shape)
    ds.create_label_group(["1"], name="prediction")  # idempotent: no shape-mismatch error
