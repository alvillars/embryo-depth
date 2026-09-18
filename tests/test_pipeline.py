import numpy as np
import pytest
from scipy import ndimage as ndi

from embryo_depth.depth import crop_bounded, surface_distance_um, touched_faces
from embryo_depth.segment import fill_interior, signed_distance, smooth_surface
from embryo_depth.store import DEPTH_SCALE_UM, Level, dequantize_depth, quantize_depth
from embryo_depth.upscale import (
    linear_axis_weights,
    nearest_indices,
    smooth_upsample_to,
    upsample_to,
)

ANISOTROPIC = (2.0, 0.52, 0.52)


def make_level(path, shape, scale, translation=None):
    return Level(
        path=path,
        shape=(1, 1, *shape),
        chunks=(1, 1, *shape),
        dtype=np.dtype("u1"),
        scale=(1.0, 1.0, *scale),
        translation=translation or (0.0, 0.0, 0.0, 0.0, 0.0),
    )


def sphere(shape, radius_um, voxel_um, centre=None):
    centre = centre or [s / 2 for s in shape]
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    r2 = sum(((g - c) * v) ** 2 for g, c, v in zip(grids, centre, voxel_um))
    return r2 <= radius_um**2


# --- distance transform ------------------------------------------------------------


def test_sphere_edt_recovers_radius_in_micrometres():
    """A sphere of known radius must have a maximum depth equal to that radius.

    This is the test that catches a missing or wrong `sampling` argument. Without it the
    transform silently returns depth in voxels, which on this anisotropic grid is wrong
    along z by ~4x -- and a depth field in the wrong units still looks perfectly smooth.
    """
    radius = 30.0
    mask = sphere((60, 200, 200), radius, ANISOTROPIC)
    depth = surface_distance_um(mask, ANISOTROPIC)
    assert depth.max() == pytest.approx(radius, abs=1.5)


def test_edt_without_sampling_is_wrong_along_z():
    """Guard the guard: confirm the sphere test would actually fail if sampling were lost.

    Dropping ``sampling`` reports depth in voxels. Because z is the coarse axis, the
    nearest surface is found ~2x too close, so a 30 um sphere reads as ~15 -- an error in
    the direction that makes tissue look shallower than it is.
    """
    mask = sphere((60, 200, 200), 30.0, ANISOTROPIC)
    naive = ndi.distance_transform_edt(mask)  # voxels, not micrometres
    assert naive.max() == pytest.approx(15.0, abs=1.0)
    assert abs(naive.max() - 30.0) > 10.0


def test_depth_is_zero_exactly_outside_the_mask():
    mask = sphere((40, 120, 120), 20.0, ANISOTROPIC)
    depth = surface_distance_um(mask, ANISOTROPIC)
    assert np.all(depth[~mask] == 0)
    assert np.all(depth[mask] > 0)


def test_boundary_modes_differ_only_where_the_body_is_cut():
    """A body running off the edge: 'surface' makes the crop plane shallow, 'ignore' does not."""
    mask = np.zeros((40, 80, 80), bool)
    mask[:, 20:60, 20:60] = True  # spans the full z range, so z faces are cut
    ignored = surface_distance_um(mask, ANISOTROPIC, boundary="ignore")
    as_surface = surface_distance_um(mask, ANISOTROPIC, boundary="surface")
    assert as_surface[0][mask[0]].max() < ignored[0][mask[0]].max()
    # away from the cut faces the two agree
    mid = mask.shape[0] // 2
    assert np.allclose(ignored[mid], as_surface[mid], atol=1e-4)


def test_crop_bounded_flags_tissue_near_a_cut_face():
    mask = np.zeros((40, 80, 80), bool)
    mask[:, 20:60, 20:60] = True
    level = make_level("1", mask.shape, ANISOTROPIC)
    depth = surface_distance_um(mask, ANISOTROPIC)
    faces = touched_faces(mask)
    assert set(faces) == {"z0", "z1"}
    flagged = crop_bounded(depth, level, faces)
    assert flagged[0].any()  # the cut plane itself is unreliable
    assert not flagged[:, 0].any()  # nothing flagged where there is no body


def test_no_faces_touched_means_nothing_is_crop_bounded():
    mask = sphere((60, 200, 200), 25.0, ANISOTROPIC)
    level = make_level("1", mask.shape, ANISOTROPIC)
    depth = surface_distance_um(mask, ANISOTROPIC)
    assert touched_faces(mask) == {}
    assert not crop_bounded(depth, level, {}).any()


# --- cavity filling ----------------------------------------------------------------


def test_fill_interior_fills_a_fully_enclosed_cavity():
    mask = sphere((60, 120, 120), 25.0, ANISOTROPIC)
    mask &= ~sphere((60, 120, 120), 10.0, ANISOTROPIC)
    assert not mask[30, 60, 60]
    assert fill_interior(mask)[30, 60, 60]


def test_fill_interior_fills_a_cavity_cut_open_by_the_volume_edge():
    """The case plain fill_holes gets wrong once the embryo outgrows the imaged volume.

    A cavity touching a face is reachable from the array border, so binary_fill_holes
    stops treating it as a hole and leaves it open -- 16-20% of the body on this dataset.
    """
    shape = (60, 120, 120)
    # Both spheres centred on the z=0 plane, so the body is a shell cut open by that face
    # and its cavity reaches the face -- the geometry this dataset develops from t~90.
    body = sphere(shape, 25.0, ANISOTROPIC, centre=[0, 60, 60])
    cavity = sphere(shape, 10.0, ANISOTROPIC, centre=[0, 60, 60])
    mask = body & ~cavity
    assert cavity[0, 60, 60] and mask[0].any()  # cavity really does reach the face

    assert not ndi.binary_fill_holes(mask)[cavity].all()  # plain fill leaves it open
    assert fill_interior(mask)[cavity].all()  # face-sealed fill closes it


def test_fill_interior_does_not_swallow_the_exterior():
    mask = sphere((60, 120, 120), 20.0, ANISOTROPIC)
    filled = fill_interior(mask)
    assert filled[0, 0, 0] == False  # noqa: E712
    assert filled.sum() == pytest.approx(mask.sum(), rel=1e-3)


def test_fill_interior_is_extensive():
    rng = np.random.default_rng(0)
    mask = sphere((40, 80, 80), 15.0, ANISOTROPIC)
    mask &= rng.random(mask.shape) > 0.1  # punch random holes
    assert np.all(fill_interior(mask)[mask])


# --- surface conditioning ----------------------------------------------------------


def test_signed_distance_is_positive_inside_negative_outside():
    shape = (60, 200, 200)
    radius = 30.0
    mask = sphere(shape, radius, ANISOTROPIC)
    signed = signed_distance(mask, ANISOTROPIC)
    assert np.all(signed[mask] > 0)
    assert np.all(signed[~mask] < 0)
    # the zero level set sits within one voxel of the true surface along z (the coarse axis
    # on this anisotropic sampling, so the loosest bound)
    y0, x0 = shape[1] // 2, shape[2] // 2
    z1 = shape[0] // 2
    while mask[z1, y0, x0]:
        z1 += 1
    assert abs(signed[z1, y0, x0]) <= ANISOTROPIC[0]


@pytest.mark.parametrize("radius", [30.0, 60.0, 120.0])
def test_smoothing_shrinks_by_the_curvature_flow_law(radius):
    """Smoothing the signed distance is curvature flow, so convex bodies shrink.

    The surface retreats by sigma^2 / R, i.e. the volume drops by 3 * sigma^2 / R^2.
    Pinning the law rather than a tolerance is what makes the bias safe to reason about:
    it is 12% on a 30 um sphere but ~0.4% on this embryo at sigma = 6 um.
    """
    sigma = 6.0
    shape = (int(4 * radius / ANISOTROPIC[0]), int(2.6 * radius / ANISOTROPIC[1]),) * 1 + (
        int(2.6 * radius / ANISOTROPIC[1]),
    )
    mask = sphere(shape, radius, ANISOTROPIC)
    smoothed = smooth_surface(mask, ANISOTROPIC, sigma)
    observed = (smoothed.sum() - mask.sum()) / mask.sum()
    predicted = -3 * sigma**2 / radius**2
    assert observed == pytest.approx(predicted, rel=0.15)


def test_smoothing_removes_a_thin_spike():
    shape = (60, 200, 200)
    radius = 30.0
    mask = sphere(shape, radius, ANISOTROPIC)
    # The sphere reaches y ~ 157 at this voxel size; put the spike beyond that so the test
    # is about the spike and not about a finger buried inside the body.
    spike = np.zeros(shape, bool)
    spike[30, 155:190, 99:102] = True
    assert not mask[30, 185, 100]
    smoothed = smooth_surface(mask | spike, ANISOTROPIC, 6.0)
    assert not smoothed[30, 185, 100]  # tip of the spike is gone
    assert smoothed[30, 100, 100]  # bulk of the body survives


# --- upscaling ---------------------------------------------------------------------


def test_upsample_hits_the_exact_target_shape():
    """Shapes came from floor(n/2), so 143 -> 287 -> 575 -> 1151 -> 2303, never exactly 2x."""
    source = make_level("4", (150, 143, 143), (2.0, 4.16, 4.16))
    for shape, scale in [
        ((150, 286, 287), (2.0, 2.08, 2.08)),
        ((150, 572, 575), (2.0, 1.04, 1.04)),
        ((150, 1145, 1151), (2.0, 0.52, 0.52)),
        ((300, 2291, 2303), (1.0, 0.26, 0.26)),
    ]:
        target = make_level("t", shape, scale)
        out = upsample_to(np.zeros((150, 143, 143), np.uint8), source, target)
        assert out.shape == shape


def test_upsample_indices_are_monotone_and_cover_the_source():
    source = make_level("4", (150, 143, 143), (2.0, 4.16, 4.16))
    target = make_level("0", (300, 2291, 2303), (1.0, 0.26, 0.26))
    for axis in range(3):
        idx = nearest_indices(source, target, axis)
        assert np.all(np.diff(idx) >= 0)
        assert idx.min() == 0
        assert idx.max() == source.shape[axis + 2] - 1


def test_upsample_preserves_values_and_rough_volume_fraction():
    source = make_level("4", (20, 30, 30), (2.0, 4.16, 4.16))
    target = make_level("2", (20, 120, 121), (2.0, 1.04, 1.04))
    rng = np.random.default_rng(1)
    vol = (rng.random((20, 30, 30)) > 0.7).astype(np.uint8)
    out = upsample_to(vol, source, target)
    assert set(np.unique(out)) <= {0, 1}
    assert out.mean() == pytest.approx(vol.mean(), abs=0.03)


def test_z_is_only_doubled_between_level_1_and_0():
    """Levels 1-4 all share z=150; only level 0 has 300. A fixed factor would break this."""
    source = make_level("4", (150, 143, 143), (2.0, 4.16, 4.16))
    for shape, scale, expect_identity in [
        ((150, 572, 575), (2.0, 1.04, 1.04), True),
        ((300, 2291, 2303), (1.0, 0.26, 0.26), False),
    ]:
        idx = nearest_indices(source, make_level("t", shape, scale), 0)
        assert np.array_equal(idx, np.arange(150)) == expect_identity


def test_linear_axis_weights_reduce_to_identity_when_scale_matches():
    """z between level 4 and level 1 shares scale+translation, so it must cost nothing extra."""
    source = make_level("4", (150, 143, 143), (2.0, 4.16, 4.16))
    target = make_level("1", (150, 1145, 1151), (2.0, 0.52, 0.52))
    lo, hi, weight_hi = linear_axis_weights(source, target, 0)
    assert np.array_equal(lo, np.arange(150))
    # hi is still lo+1 (clipped) even at zero weight -- weight_hi==0 everywhere is what
    # makes it irrelevant to _blend_axis, which is the actual invariant that matters
    assert np.all(weight_hi == 0)


def test_linear_axis_weights_interpolate_the_magnified_xy_axes():
    source = make_level("4", (150, 143, 143), (2.0, 4.16, 4.16))
    target = make_level("1", (150, 1145, 1151), (2.0, 0.52, 0.52))
    lo, hi, weight_hi = linear_axis_weights(source, target, 1)
    assert np.all(hi - lo <= 1)
    assert np.all((weight_hi >= 0) & (weight_hi <= 1))
    # not every target voxel can land exactly on a source voxel at an 8x magnification
    assert np.any(weight_hi > 0)


def test_smooth_upsample_hits_the_exact_target_shape():
    source = make_level("4", (150, 143, 143), (2.0, 4.16, 4.16))
    mask = np.zeros((150, 143, 143), np.uint8)
    for shape, scale in [
        ((150, 286, 287), (2.0, 2.08, 2.08)),
        ((150, 572, 575), (2.0, 1.04, 1.04)),
        ((150, 1145, 1151), (2.0, 0.52, 0.52)),
        ((300, 2291, 2303), (1.0, 0.26, 0.26)),
    ]:
        target = make_level("t", shape, scale)
        out = smooth_upsample_to(mask, source, target)
        assert out.shape == shape
        assert set(np.unique(out)) <= {0, 1}


def test_smooth_upsample_preserves_values_and_rough_volume_fraction():
    source = make_level("4", (20, 30, 30), (2.0, 4.16, 4.16))
    target = make_level("2", (20, 120, 121), (2.0, 1.04, 1.04))
    mask = sphere((20, 30, 30), 12.0, (2.0, 4.16, 4.16))
    out = smooth_upsample_to(mask.astype(np.uint8), source, target)
    assert set(np.unique(out)) <= {0, 1}
    assert out.mean() == pytest.approx(mask.mean(), abs=0.05)


def test_smooth_upsample_reduces_surface_roughness_vs_nearest_neighbour():
    """The whole point of issue 1: an 8x-in-xy upsample should not look like a staircase.

    Roughness proxy: boundary-voxel count normalised by volume^(2/3) (a discretised
    surface-area-to-volume ratio). Nearest-neighbour block-replication inflates this by
    replicating the source's already-blocky surface into much bigger axis-aligned steps;
    interpolating the signed distance field should not.
    """
    source = make_level("4", (36, 24, 24), (2.0, 4.16, 4.16))
    target = make_level("1", (36, 192, 192), (2.0, 0.52, 0.52))  # 8x in xy, matching level4->1
    mask = sphere((36, 24, 24), 9.0, (2.0, 4.16, 4.16)).astype(np.uint8)

    def roughness(m):
        boundary = m.astype(bool) ^ ndi.binary_erosion(m.astype(bool))
        return boundary.sum() / max(m.sum(), 1) ** (2 / 3)

    nn = upsample_to(mask, source, target)
    sm = smooth_upsample_to(mask, source, target)
    assert roughness(sm) < roughness(nn)


# --- quantisation ------------------------------------------------------------------


def test_quantisation_roundtrip_is_within_half_a_step():
    values = np.linspace(0, 120, 500).astype(np.float32)
    back = dequantize_depth(quantize_depth(values))
    assert np.max(np.abs(back - values)) <= DEPTH_SCALE_UM / 2 + 1e-6


def test_quantisation_saturates_rather_than_wrapping():
    assert quantize_depth(np.array([1e6])).item() == 255
    assert quantize_depth(np.array([-5.0])).item() == 0


def test_quantised_depth_is_zero_exactly_where_the_mask_is_zero():
    mask = sphere((40, 120, 120), 18.0, ANISOTROPIC)
    stored = quantize_depth(surface_distance_um(mask, ANISOTROPIC))
    assert np.all(stored[~mask] == 0)


# --- depth-bucketed index ----------------------------------------------------------


def _fake_timepoint(seed=0):
    from embryo_depth.store import quantize_depth

    rng = np.random.default_rng(seed)
    mask = sphere((30, 60, 60), 12.0, ANISOTROPIC)
    depth = quantize_depth(surface_distance_um(mask, ANISOTROPIC))
    image = (rng.integers(100, 4000, mask.shape)).astype(np.uint16)
    image[~mask] = 0
    return image, depth, make_level("2", mask.shape, ANISOTROPIC)


@pytest.mark.parametrize("depth_um,half_width", [(0, 1), (3, 2), (6, 3), (9, 1), (2, 6)])
def test_index_shell_matches_the_naive_full_volume_computation(depth_um, half_width):
    """The index is the one optimisation that could silently drop voxels.

    Slicing a counting-sorted array must give exactly what masking the whole volume gives,
    for every band -- including bands that run off either end of the depth range.
    """
    from embryo_depth.viewer import ShellSource, build_index, naive_shell_volume

    image, depth, level = _fake_timepoint()
    index = build_index(image, depth, level)

    src = ShellSource.__new__(ShellSource)
    src.index = {0: index}
    got = src.shell_volume(0, depth_um, half_width)
    expected = naive_shell_volume(image, depth, depth_um, half_width)
    np.testing.assert_array_equal(got, expected)


def test_index_covers_every_body_voxel_exactly_once():
    from embryo_depth.viewer import build_index

    image, depth, level = _fake_timepoint(seed=3)
    index = build_index(image, depth, level)
    assert len(index.coords) == int((depth > 0).sum())
    assert index.offsets[0] == 0
    assert index.offsets[-1] == len(index.coords)
    # every voxel lands in the bucket matching its own depth
    for unit in range(1, 256):
        sl = index.band(unit, unit)
        if sl.stop > sl.start:
            c = index.coords[sl]
            assert np.all(depth[c[:, 0], c[:, 1], c[:, 2]] == unit)


def test_index_band_is_clamped_not_wrapped():
    from embryo_depth.viewer import build_index

    image, depth, level = _fake_timepoint(seed=5)
    index = build_index(image, depth, level)
    assert index.band(-10, 2).start == 0
    assert index.band(250, 999).stop == len(index.coords)


def test_max_depth_unit_reports_the_deepest_occupied_bucket():
    """offsets is a cumsum, so testing it directly reports 255 for any non-empty body."""
    from embryo_depth.viewer import ShellSource, build_index

    image, depth, level = _fake_timepoint(seed=7)
    src = ShellSource.__new__(ShellSource)
    src.index = {0: build_index(image, depth, level)}
    assert src.max_depth_unit == int(depth.max())
    assert src.max_depth_unit < 255


# --- napari plugin -----------------------------------------------------------------


def test_plugin_widget_constructs(qt_app):
    """magicgui resolves annotations at construction time.

    Adding `from __future__ import annotations` to plugin.py turns them into strings that
    magicgui cannot resolve ('No module named Path'), and the widget fails to build only
    when a user opens it from the plugin menu -- never during normal imports or the rest of
    this suite. Constructing it here is the cheapest way to keep that from regressing.
    """
    pytest.importorskip("napari")
    from embryo_depth.plugin import depth_shell_widget

    widget = depth_shell_widget()
    assert widget._loader is not None
    assert widget._shell is not None
    # the sliders the user drives must exist with sane bounds
    assert widget._shell.depth_um.min == 0
    assert widget._shell.half_width_um.min >= 1
    # the loader's annotations must have resolved to real widgets, not forward refs
    assert widget._loader.store.value is not None
    assert set(widget._loader.level.choices) == {"1", "2", "3", "4"}
