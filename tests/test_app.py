"""Offline checks of the converter's RGB decision, merges, placement and layer authoring.

No backend and no LIF file: liffile's array is a stand-in xarray, and the Mikro client
is a recorder, so the assertions land on the calls that would reach the server rather
than on a property that predicts them.

Everything here is about one rule:

    ONE LAYER PER CHANNEL. The only exception is a packed RGB(A) pixel.
"""

import numpy as np
import pytest
import xarray as xr

import app as M
from leica_metadata import DisplayWindow, LeicaChannel, LeicaDimension


class _RecordingMikro:
    """Stands in for the Mikro client: records every mutation the layer author makes."""

    def __init__(self, calls: list) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        return lambda **kw: self._calls.append((name, kw))


def test_the_app_declares_its_action():
    """One action, taking a mikro file and handed a mikro client."""
    assert M.app.services == ["mikro"]
    declaration = M.app.registry.to_implement_agent_input().model_dump(
        mode="json", exclude_none=True
    )
    [implementation] = declaration["implementations"]
    assert implementation["interface"] == "convert_lif_file_to_datasets"
    args = {arg["key"]: arg for arg in implementation["definition"]["args"]}
    assert args["file"]["identifier"] == "@mikro/file"
    assert args["levels"]["default"] == 0
    # The client and the task are injected, not ports.
    assert "mikro" not in args and "task" not in args


# --- liffile's array, one tile at a time --------------------------------------------


def whole(dims, shape, fill=None):
    data = np.zeros(shape, dtype=np.uint8) if fill is None else fill
    return xr.DataArray(data, dims=dims)


@pytest.mark.parametrize(
    ("dims", "shape", "expected"),
    [
        # A LAS X snapshot: liffile's sample axis.
        (("Y", "X", "S"), (8, 8, 3), (3, True)),
        (("Y", "X", "S"), (8, 8, 4), (4, True)),
        # Any other extent behind an "S" axis is some other packing, not a picture.
        (("Y", "X", "S"), (8, 8, 2), (2, False)),
        # Three dyes, stored channel-planar.
        (("C", "Y", "X"), (3, 8, 8), (3, False)),
        (("Z", "C", "Y", "X"), (5, 4, 8, 8), (4, False)),
        # One channel, squeezed away by liffile: c comes back at extent 1.
        (("Y", "X"), (8, 8), (1, False)),
    ],
)
def test_the_sample_axis_decides_packing(dims, shape, expected):
    array, _, packed = M.tile_array(whole(dims, shape), None)
    assert (int(array.sizes["c"]), packed) == expected


def test_samples_alongside_channels_are_refused():
    with pytest.raises(ValueError):
        M.tile_array(whole(("C", "Y", "X", "S"), (2, 8, 8, 3)), None)


def test_each_tile_is_its_own_pixels():
    tiles = np.arange(3, dtype=np.uint8)[:, None, None, None] * np.ones(
        (3, 2, 4, 4), dtype=np.uint8
    )
    scan = whole(("M", "C", "Y", "X"), tiles.shape, tiles)
    for m in range(3):
        array, extra, _ = M.tile_array(scan, m)
        assert "m" not in array.dims and not extra
        assert int(array.max()) == m


def test_a_leftover_axis_is_declared_index_not_space():
    _, extra, _ = M.tile_array(whole(("λ", "C", "Y", "X"), (5, 2, 4, 4)), None)
    assert extra == {"λ": "INDEX"}


@pytest.mark.parametrize(
    ("sizes", "levels"),
    [
        ({"y": 512, "x": 512}, 2),
        ({"y": 1024, "x": 1024}, 2),
        ({"y": 11003, "x": 11022}, 6),
        ({"y": 64, "x": 64}, 2),
        ({"y": 3741, "x": 3680}, 4),
    ],
)
def test_auto_levels(sizes, levels):
    assert M.auto_levels(sizes) == levels


# --- placement -----------------------------------------------------------------------


def dim(label, count, length_m, origin=0.0, unit="m"):
    return LeicaDimension(label, count, origin, length_m, unit)


def test_voxel_scale():
    dims = {
        "X": dim("X", 1024, 1.845238e-04),
        "Y": dim("Y", 1024, 1.845238e-04),
        "Z": dim("Z", 36, 1.17194e-05, origin=1.51853e-03),
    }
    scale, note = M.voxel_scale(dims, ["c", "z", "y", "x"])
    assert scale == pytest.approx({"z": 0.33484, "y": 0.180375171, "x": 0.180375171})
    assert note is None


def test_a_snapshot_is_not_registered():
    dims = {"X": dim("X", 1097, 1098.0, unit="Pixel"), "Y": dim("Y", 1097, 1098.0, unit="Pixel")}
    scale, note = M.voxel_scale(dims, ["c", "y", "x"])
    assert scale is None and "not registered" in note


def test_stage_offsets_put_the_field_centre_at_the_stage_centre():
    data = xr.DataArray(np.zeros((1, 3, 101, 201)), dims=("c", "z", "y", "x"))
    centre = M.StageCentre(1000.0, 2000.0, "TileScanInfo", False, False, False)
    z = dim("Z", 3, 2e-06, origin=1.5e-3)
    offsets = M.stage_offsets(centre, {"x": 0.5, "y": 0.5, "z": 1.0}, data, {"Z": z})
    assert offsets == pytest.approx({"x": 950.0, "y": 1975.0, "z": 1500.0})


class FakeTileScan:
    def __init__(self, rows, flip_x=False, flip_y=False, swap_xy=False):
        self.tiles = np.array(
            [(0, 0, 0.0, y, x) for x, y in rows],
            dtype=[
                ("field_y", np.int32),
                ("field_x", np.int32),
                ("pos_z", np.float64),
                ("pos_y", np.float64),
                ("pos_x", np.float64),
            ],
        )
        self.flip_x, self.flip_y, self.swap_xy = flip_x, flip_y, swap_xy

    def __len__(self):
        return len(self.tiles)


class FakeLifImage:
    def __init__(self, tilescan):
        self.tilescan = tilescan


class StampedImage:
    """What `acquisition_window` reads of a LIF image: its dimensions and frame stamps."""

    def __init__(self, sizes, stamps):
        self.dims = tuple(sizes)
        self.sizes = dict(sizes)
        self.timestamps = None if stamps is None else np.array(stamps, dtype="datetime64[ms]")


def stamp(second):
    return np.datetime64("2026-03-19T17:00:00") + np.timedelta64(second, "s")


def utc(second):
    import datetime

    return datetime.datetime(2026, 3, 19, 17, 0, second, tzinfo=datetime.UTC)


def test_an_images_window_runs_from_its_first_frame_to_its_last():
    window = M.acquisition_window(StampedImage({"C": 3, "Y": 8, "X": 8}, [stamp(5), stamp(1), stamp(3)]), None)
    assert (window.start, window.end) == (utc(1), utc(5))
    assert window.start.utcoffset() is not None, "a FILETIME is UTC, and says so"


def test_an_unstamped_image_states_no_window():
    assert M.acquisition_window(StampedImage({"Y": 8, "X": 8}, None), None) is None
    assert M.acquisition_window(StampedImage({"Y": 8, "X": 8}, []), None) is None


def test_a_tile_states_its_own_frames():
    # M outermost, two channels per tile: frames in the image's dimension order.
    image = StampedImage({"M": 3, "C": 2, "Y": 8, "X": 8}, [stamp(s) for s in (0, 1, 10, 11, 20, 21)])
    window = M.acquisition_window(image, 1)
    assert (window.start, window.end) == (utc(10), utc(11))


def test_a_tile_states_nothing_when_its_frames_cannot_be_told_apart():
    image = StampedImage({"M": 3, "C": 2, "Y": 8, "X": 8}, [stamp(0), stamp(1)])
    assert M.acquisition_window(image, 1) is None


def windows(per_t):
    return [(index, (w.start, w.end)) for index, w in per_t]


def test_each_timepoint_spans_its_own_frames():
    image = StampedImage({"T": 3, "C": 2, "Y": 8, "X": 8}, [stamp(s) for s in (0, 1, 10, 12, 20, 21)])
    whole, per_t = M.frame_windows(image, None)
    assert (whole.start, whole.end) == (utc(0), utc(21))
    assert windows(per_t) == [(0, (utc(0), utc(1))), (1, (utc(10), utc(12))), (2, (utc(20), utc(21)))]


def test_a_tiles_timepoints_are_its_own():
    image = StampedImage({"M": 2, "T": 2, "Y": 8, "X": 8}, [stamp(s) for s in (0, 1, 10, 11)])
    whole, per_t = M.frame_windows(image, 1)
    assert (whole.start, whole.end) == (utc(10), utc(11))
    assert windows(per_t) == [(0, (utc(10), utc(10))), (1, (utc(11), utc(11)))]


def test_no_time_axis_states_no_timepoints():
    assert M.frame_windows(StampedImage({"C": 2, "Y": 8, "X": 8}, [stamp(0), stamp(1)]), None)[1] == []
    assert M.frame_windows(StampedImage({"T": 1, "C": 2, "Y": 8, "X": 8}, [stamp(0), stamp(1)]), None)[1] == []


def test_frames_that_cannot_be_told_apart_state_no_timepoints():
    whole, per_t = M.frame_windows(StampedImage({"T": 3, "C": 2, "Y": 8, "X": 8}, [stamp(0), stamp(5)]), None)
    assert (whole.start, whole.end) == (utc(0), utc(5)), "the whole is still every stamp there is"
    assert per_t == []
    assert M.frame_windows(StampedImage({"M": 3, "T": 2, "Y": 8, "X": 8}, [stamp(0)]), 1) == (None, [])


def test_stage_centre_reads_tile_scan_info_in_micrometers():
    image = FakeLifImage(FakeTileScan([(0.08, 0.04), (0.081, 0.04)]))
    centre = M.stage_centre(image, 1, None)
    assert (centre.x, centre.y) == pytest.approx((81000.0, 40000.0))
    assert centre.source == "TileScanInfo"
    assert M.stage_centre(image, 5, None) is None


def test_tile_scan_flips_apply():
    image = FakeLifImage(FakeTileScan([(0.08, 0.04)], flip_x=True, swap_xy=True))
    centre = M.stage_centre(image, 0, None)
    assert (centre.x, centre.y) == pytest.approx((40000.0, -80000.0))


# --- which images get a scene --------------------------------------------------------
#
# Geometry as LAS X writes it for the files checked: tile centres on a grid, and a
# merged image whose one centre is the grid's mean.


def scan_print(path, centre=(1000.0, 1000.0), extent=(350.6, 350.6), tiles=4, channels=4):
    return M.Footprint(path, path.rsplit("/", 1)[-1], tiles, channels, centre, extent)


def merged_print(path, centre=(1000.0, 1000.0), extent=(348.9, 346.5), channels=4):
    return M.Footprint(path, path.rsplit("/", 1)[-1], 1, channels, centre, extent)


def test_a_sibling_merge_is_found():
    prints = [scan_print("R1"), merged_print("R1_Merged")]
    assert M.merged_partners(prints) == {0: 1}


def test_a_merge_named_otherwise_and_elsewhere_is_found_by_geometry():
    prints = [
        scan_print("pig overview/Series", centre=(84319.0, 39593.7), extent=(8486.2, 8486.2), tiles=64),
        merged_print("pig_overview_merged", centre=(84319.0, 39593.7), extent=(8372.0, 8510.6)),
    ]
    assert M.merged_partners(prints) == {0: 1}


def test_two_scans_of_one_name_get_their_own_merges():
    prints = [
        scan_print("TileScan 2/R1", centre=(84811.3, 38286.5)),
        scan_print("TileScan 3/R1", centre=(84810.4, 39489.0)),
        merged_print("TileScan 3/R1_Merged", centre=(84810.4, 39489.0)),
        merged_print("TileScan 2/R1_Merged", centre=(84811.3, 38286.5)),
    ]
    assert M.merged_partners(prints) == {0: 3, 1: 2}


def test_a_field_at_the_scan_centre_is_not_its_merge():
    # A zoomed-in single image taken at the middle of the tile grid.
    prints = [scan_print("R1"), merged_print("Zoom", extent=(90.0, 90.0))]
    assert M.merged_partners(prints) == {}


def test_a_scan_without_a_merge_has_no_partner():
    prints = [scan_print("R1"), merged_print("Elsewhere", centre=(5000.0, 5000.0))]
    assert M.merged_partners(prints) == {}


def test_one_merge_answers_for_one_scan():
    # Two scans over the same grid (a re-acquisition), one merged image.
    prints = [scan_print("a/Series"), scan_print("b/Series"), merged_print("R1_Merged")]
    assert M.merged_partners(prints) == {0: 2}


def test_channels_have_to_agree():
    prints = [scan_print("R1"), merged_print("R1_Merged", channels=1)]
    assert M.merged_partners(prints) == {}


def test_without_geometry_the_name_decides():
    prints = [
        scan_print("TileScan 1/R 1", centre=None, extent=None),
        merged_print("TileScan 1/R 1_Merged", centre=None, extent=None),
        merged_print("R 1_Merged", centre=None, extent=None),
    ]
    assert M.merged_partners(prints) == {0: 1}


def test_unreadable_images_are_skipped():
    assert M.merged_partners([None, scan_print("R1"), None]) == {}


# --- the decision `convert_tile` makes ----------------------------------------------


@pytest.mark.parametrize(
    ("packed", "channels", "dtype", "expected"),
    [
        (True, 3, np.uint8, True),
        (True, 4, np.uint8, True),
        (False, 3, np.uint8, False),
        (True, 6, np.uint8, False),
        # An RGB layer is drawn unstretched over 0..255: a deeper sample axis is a colour
        # camera's acquisition and gets a layer and a window per sample instead.
        (True, 3, np.uint16, False),
        (True, 3, np.float32, False),
    ],
)
def test_packs_one_pixel(packed, channels, dtype, expected):
    assert M.packs_one_pixel(packed, channels, dtype) is expected


# --- the window a layer opens with ---------------------------------------------------
#
# Sparse fluorescence: 0.5 % of every z-column is signal, at a different depth per column.
# By voxel count that is background, so the stack's p99 is the background level — while
# the projection the volume layer draws is signal in every pixel.


@pytest.fixture
def sparse_stack():
    stack = np.full((200, 32, 32), 2, dtype=np.uint8)
    depth = np.arange(32 * 32).reshape(32, 32) % 200
    np.put_along_axis(
        stack,
        depth[None],
        np.linspace(100, 200, 32 * 32).reshape(1, 32, 32).astype(np.uint8),
        axis=0,
    )
    assert float(np.percentile(stack, 99)) == 2.0
    return stack


def leica_channel(window=None, bits=8, lut="Green"):
    return LeicaChannel(0, lut, bits, "ALEXA 488", None, None, (), None, None, window)


def window_spec(array, dims, levels=1, leica=()):
    """The spec `channel_specs` gives channel 0 of `array`, through the real function."""
    data = xr.DataArray(array[None], dims=("c", *dims))
    pyramid = [data] + [data.coarsen(y=2**n, x=2**n).max() for n in range(1, levels)]
    return M.channel_specs(data, pyramid, list(leica), False, "z" in dims)[0]


def test_a_volume_window_is_taken_from_its_projection(sparse_stack):
    spec = window_spec(sparse_stack, ("z", "y", "x"))
    assert spec.clim[0] > 99 and spec.clim[1] > 190
    # About 1 % of the projection saturated, not all of it.
    assert float((sparse_stack.max(axis=0) >= spec.clim[1]).mean()) < 0.02
    # While the anchor's histogram stays the voxels' own.
    assert spec.histogram.p99 == 2.0


def test_an_over_budget_stack_still_gets_the_exact_projection_window(
    sparse_stack, monkeypatch
):
    exact = window_spec(sparse_stack, ("z", "y", "x"))
    monkeypatch.setattr(M, "HISTOGRAM_BUDGET_BYTES", sparse_stack.nbytes // 2)
    pooled = window_spec(sparse_stack, ("z", "y", "x"), levels=3)
    assert pooled.clim == exact.clim
    # And its histogram from the finest level that fits, not the coarsest.
    assert sum(pooled.histogram.histogram) == sparse_stack.size / 4


def test_the_las_x_window_wins_where_the_file_has_one(sparse_stack):
    spec = window_spec(
        sparse_stack,
        ("z", "y", "x"),
        leica=[leica_channel(DisplayWindow(0.0, 0.5, 1.0))],
    )
    assert spec.clim == (0.0, 127.5)
    assert spec.label == "ALEXA 488"
    assert spec.color == (0, 255, 0, 255)


def test_the_las_x_window_is_a_fraction_of_the_recorded_bits():
    channel = leica_channel(DisplayWindow(0.0, 1.0 / 3, 1.0), bits=12)
    assert M.leica_window(channel, np.uint16) == pytest.approx((0.0, 1365.0))
    assert M.leica_window(channel, np.float32) is None
    assert M.leica_window(leica_channel(None), np.uint8) is None


def test_display_window():
    assert M.display_window(np.arange(101.0).reshape(1, 101)) == (1.0, 99.0)
    # Over 99 % one value: the percentiles coincide, and a window of no width draws
    # nothing.
    dots = np.zeros((64, 64), dtype=np.uint16)
    dots[0, :8] = 500
    assert M.display_window(dots) == (0.0, 500.0)
    assert M.display_window(np.full((4, 4), 7.0)) == (7.0, 8.0)


def test_colours_follow_the_lut_then_the_band_then_the_palette():
    assert M.channel_color(leica_channel(lut="Magenta"), 0, False, 4) == (255, 0, 255, 255)
    from leica_metadata import LeicaDetector

    banded = LeicaChannel(
        0,
        "custom LUT",
        8,
        None,
        None,
        LeicaDetector("HyD S 1", None, None, None, None, None, (650.0, 750.0)),
        (),
        None,
        None,
        None,
    )
    assert M.channel_color(banded, 0, False, 4) == M.color_for_wavelength(
        700.0, M.COLOR_BY_EMISSION, M.FAR_RED
    )
    assert M.channel_color(None, 1, False, 4) == M.fallback_color(1, 4)


# --- the layers themselves, not a property that predicts them -----------------------


class FakeId:
    id = "fake"
    name = "fake"


class FakeLens:
    def __init__(self, id):
        self.id = id


class FakeDataset:
    """A dataset as far as the layer author reads one: its axes, and its lenses.

    A lens' id spells out what it pins — ``"lens"`` for the whole lens, ``"lens:c=2"``
    for channel 2 — so a recorded call can be asked which channel it draws.
    """

    axis_names = ("c", "z", "y", "x")

    def lens(self, **selections):
        for axis in selections:
            if axis not in self.axis_names:
                raise ValueError(f"Invalid axis {axis!r} for axes {list(self.axis_names)}")
        if not selections:
            return FakeLens("lens")
        return FakeLens("lens:" + ",".join(f"{a}={v}" for a, v in selections.items()))


def pinned_channel(lens_id):
    """The channel a fake lens id pins, or None for the whole lens and for no lens."""
    if not isinstance(lens_id, str) or not lens_id.startswith("lens:"):
        return None
    pins = dict(pin.split("=") for pin in lens_id[len("lens:") :].split(","))
    return int(pins["c"]) if "c" in pins else None


def fake_specs(n):
    return [
        M.ChannelSpec(
            index=i,
            label=f"ch{i}",
            color=(i * 60 % 256, 0, 0, 255),
            histogram=M.ValueHistogramInput(
                histogram=[1.0], bins=[0.0, 1.0], min=0.0, max=1.0, p1=0.0, p99=1.0
            ),
            # Deliberately not the histogram's p1/p99, so a check can tell which of the
            # two a layer was given.
            clim=(10.0 + i, 20.0 + i),
            leica=None,
        )
        for i in range(n)
    ]


def layers_for(specs, implicit_rgb, is_volume=True):
    calls = []
    made = M.author_channel_layers(
        _RecordingMikro(calls), FakeDataset(), FakeId(), specs, is_volume, implicit_rgb
    )
    assert made == len(calls), "layer count must match the calls made"
    return calls


def selected_channels(call):
    """Every channel index this layer covers, whichever kind of layer it is.

    An unrecognised call is an error rather than an empty list: otherwise the converter
    moving to a mutation this walker does not know would make every check below pass
    vacuously against nothing at all.
    """
    name, kw = call
    if name in ("create_intensity_layer", "create_volume_layer"):
        index = pinned_channel(kw.get("lens"))
        assert index is not None, f"{name} call's lens pins no channel: {kw!r}"
        return [index]
    if name == "create_rgb_layer":
        found = [pinned_channel(kw.get(k)) for k in ("red_lens", "green_lens", "blue_lens")]
        assert None not in found, f"{name} call is missing a component lens: {kw!r}"
        return found
    if name == "create_layer":
        found = []

        def walk(node):
            if node is None:
                return
            if node.get("kind") == "channel":
                index = pinned_channel(node.get("lens"))
                assert index is not None, f"channel node's lens pins no channel: {node!r}"
                found.append(index)
            for child in node.get("children") or ():
                walk(child)

        graph = kw.get("render_graph")
        assert graph is not None, f"create_layer call carries no render_graph: {kw!r}"
        walk(graph.model_dump()["root"])
        assert found, f"create_layer graph has no channel node: {kw!r}"
        return found
    raise AssertionError(f"unrecognised layer mutation {name!r}; teach the walker about it")


def test_a_z_stack_is_one_volume_layer_per_channel():
    four = layers_for(fake_specs(4), False)
    assert len(four) == 4
    assert {n for n, _ in four} == {"create_volume_layer"}
    assert sorted(i for c in four for i in selected_channels(c)) == [0, 1, 2, 3]
    assert sorted(tuple(kw["color"]) for _, kw in four) == sorted(
        tuple(s.color) for s in fake_specs(4)
    )
    assert sorted((kw["clim_min"], kw["clim_max"]) for _, kw in four) == [
        (10.0 + i, 20.0 + i) for i in range(4)
    ]


def test_a_packed_rgb_z_stack_is_one_render_graph_layer():
    rgb = layers_for(fake_specs(3), True)
    assert len(rgb) == 1
    assert rgb[0][0] == "create_layer"
    assert sorted(selected_channels(rgb[0])) == [0, 1, 2]
    assert len(layers_for(fake_specs(4), True)) == 1


@pytest.mark.parametrize("n", [2, 5])
def test_implicit_rgb_over_a_non_pixel_channel_count_is_refused(n):
    assert len(layers_for(fake_specs(n), True)) == n


def test_flat_channels_are_intensity_layers():
    flat = layers_for(fake_specs(4), False, is_volume=False)
    assert {n for n, _ in flat} == {"create_intensity_layer"}
    assert sorted(i for c in flat for i in selected_channels(c)) == [0, 1, 2, 3]
    assert [kw.get("mode") for _, kw in flat] == [None] * 4


def test_a_flat_packed_rgb_pixel_is_one_rgb_layer():
    flat_rgb = layers_for(fake_specs(3), True, is_volume=False)
    assert [n for n, _ in flat_rgb] == ["create_rgb_layer"]
    assert selected_channels(flat_rgb[0]) == [0, 1, 2]
    kw = flat_rgb[0][1]
    assert (kw["clim_min"], kw["clim_max"]) == (0.0, 255.0)
    assert "blending" not in kw


def test_rgba_labels_and_colours():
    assert [M.channel_label(None, i, True) for i in range(4)] == [
        "Red",
        "Green",
        "Blue",
        "Alpha",
    ]
    assert [M.channel_color(None, i, True, 4) for i in range(4)] == [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 255, 255),
    ]


def test_a_persisted_level_one_changes_no_value(tmp_path):
    import dask.array as da

    base = xr.DataArray(
        da.from_array(np.random.default_rng(0).integers(0, 255, (2, 4, 64, 64), dtype=np.uint8), chunks=(1, 1, 64, 64)),
        dims=("c", "z", "y", "x"),
    )
    lazy = M.build_pyramid(base, levels=4, method="max", materialize=False)
    persisted = M.image_pyramid(base, 4, "max", {}, str(tmp_path))
    assert len(persisted) == 4
    for a, b in zip(lazy, persisted):
        assert a.dims == b.dims and a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Level 1 is read back from the scratch directory, not re-derived from the base.
    assert any(tmp_path.iterdir())


# --- the scene snapshot ---------------------------------------------------------------


def test_a_preview_is_the_maximum_projection_in_each_channels_window():
    data = np.zeros((2, 3, 4, 4), dtype=np.uint16)
    data[0, 1, 0, 0] = 20  # channel 0, bright at one plane only
    data[1, 2, 3, 3] = 15  # channel 1, halfway into its window
    level = xr.DataArray(data, dims=("c", "z", "y", "x"))
    specs = [
        M.ChannelSpec(0, "a", (255, 0, 0, 255), None, (0.0, 20.0), None),
        M.ChannelSpec(1, "b", (0, 255, 0, 255), None, (10.0, 20.0), None),
    ]

    rgb = M.render_preview(level, specs, implicit_rgb=False, side=4)

    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (255, 0, 0)
    assert tuple(rgb[3, 3]) == (0, 127, 0)
    assert rgb[1:3, 1:3].max() == 0


def test_a_preview_fits_its_side():
    level = xr.DataArray(np.zeros((1, 100, 50), dtype=np.uint8), dims=("c", "y", "x"))
    specs = [M.ChannelSpec(0, "a", (255, 255, 255, 255), None, (0.0, 1.0), None)]
    assert M.render_preview(level, specs, False, side=32).shape == (25, 13, 3)


def test_an_rgba_previews_alpha_is_not_drawn():
    level = xr.DataArray(np.full((4, 2, 2), 255, dtype=np.uint8), dims=("c", "y", "x"))
    specs = [
        M.ChannelSpec(i, "", M.channel_color(None, i, True, 4), None, (0.0, 255.0), None)
        for i in range(4)
    ]
    assert tuple(M.render_preview(level, specs, True, side=2)[0, 0]) == (255, 255, 255)


def test_tiles_are_laid_out_at_their_stage_positions():
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[..., 0] = 200
    green = np.zeros((2, 2, 3), dtype=np.uint8)
    green[..., 1] = 100
    canvas = M.mosaic(
        [
            M.Preview(red, (0.0, 0.0), (1.0, 1.0)),
            M.Preview(green, (0.0, 2.0), (1.0, 1.0)),
        ]
    )
    assert canvas.shape == (2, 4, 3)
    assert tuple(canvas[0, 0]) == (200, 0, 0)
    assert tuple(canvas[1, 3]) == (0, 100, 0)


def test_the_snapshot_is_a_png():
    import zlib

    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    png = M.encode_png(rgb)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    start = png.index(b"IDAT") + 4
    length = int.from_bytes(png[start - 8 : start - 4], "big")
    rows = np.frombuffer(zlib.decompress(png[start : start + length]), np.uint8)
    rows = rows.reshape(2, 1 + 3 * 3)
    assert (rows[:, 0] == 0).all()
    assert (rows[:, 1:].reshape(2, 3, 3) == rgb).all()


def test_a_scene_gets_its_snapshot(tmp_path):
    calls = []
    scene = M.Scene.model_construct(id="s1", name="scene")
    entry = M.Converted(
        0, "x", None, [], True, False, False,
        M.Preview(np.zeros((2, 2, 3), np.uint8), None, None),
    )
    M.snapshot_scene(_RecordingMikro(calls), scene, [entry], str(tmp_path))
    [(name, kw)] = calls
    assert name == "create_scene_snapshot"
    assert kw["scene"] == "s1"
    assert kw["file"].name.endswith(".png")


def test_a_mosaic_never_exceeds_the_snapshot_side():
    tile = np.full((4, 4, 3), 255, dtype=np.uint8)
    far = 10 * M.SNAPSHOT_PIXELS
    canvas = M.mosaic(
        [M.Preview(tile, (0.0, 0.0), (1.0, 1.0)), M.Preview(tile, (0.0, far), (1.0, 1.0))]
    )
    assert max(canvas.shape[:2]) <= M.SNAPSHOT_PIXELS
    assert canvas[0, 0].any() and canvas[0, -1].any()


def test_a_composed_scene_is_each_datasets_default():
    calls = []

    class Mikro(_RecordingMikro):
        def create_scene(self, **kw):
            calls.append(("create_scene", kw))
            return M.Scene.model_construct(
                id="s1", name=kw["name"], world_coordinate_system=FakeId()
            )

    entries = [
        M.Converted(tile, f"t{tile}", FakeDataset(), fake_specs(1), True, False, False, None)
        for tile in range(2)
    ]
    M.compose_scene(Mikro(calls), "mosaic", FakeId(), entries)
    nominations = [kw for name, kw in calls if name == "set_lens_default_scene"]
    assert nominations == [{"lens": "lens", "scene": "s1"}] * 2
