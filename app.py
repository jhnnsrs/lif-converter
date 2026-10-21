"""Convert a Leica LIF file into one ``ArrayDataset`` per image — per tile, for a tile scan.

The same contract as the OME converter, read natively: no Bio-Formats, no JVM, and no
OME in between. ``liffile`` memory-maps each image's block straight out of the file, so
the pixels are never decoded into RAM or copied to a scratch store before upload, and the
metadata comes from the image's own XML — LAS X's record of the acquisition, which keeps
what an OME translation drops: which sequential setting and detector recorded each
channel, its laser lines, the pinhole in Airy units, the display window LAS X showed it
in, and where the stage put every tile. See `leica_metadata`.

Everything the file says about an image is pinned to a *coordinate*: facts true of the
whole image (the Leica XML record, the stand, the objective, the scanner, the stage) hang
off the origin anchor; facts true of one channel (its histogram, dye, detector, laser
lines, light path) hang off that channel's anchor. Physical scale is an edge, not a
property — every calibrated image registers into one micrometer world, at its stage
position, and each gets a scene of its own over it.

A tile scan is one LIF image with an ``M`` axis. Each tile is uploaded as a dataset of
its own and registered at its own stage position. If the file also holds the merged image
those tiles were stitched into, only the merged image is staged; otherwise the tiles share
one mosaic scene — see `merged_partners`.

Nothing about a scene is bootstrapped: every lens and layer is authored explicitly,
**one layer per channel**. The only exception is a packed 8-bit RGB(A) snapshot — see
`packs_one_pixel`.

Absent facts stay absent. A snapshot measured in pixels is **not** registered into the
world: ``scale={x: 1.0}`` is not a neutral default but the claim that one pixel is one
micrometer. Such a dataset is staged in its own pixel grid instead.
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from arkitekt import App, Task, run
from mikro import Mikro, build_pyramid, canonical, mikro_service, scales_from, space_3d
from mikro.api.schema import (
    ArrayDataset,
    AxisAnchorInput,
    AxisType,
    Blending,
    ColorMap,
    CoordinateAnchorInput,
    File,
    LabelInput,
    OmeMetadataInput,
    OptikitStateInput,
    PreferredView,
    ProjectionMode,
    Scene,
    SourceFileInput,
    StageStateInput,
    ValueHistogramInput,
    ValueRelation,
)
from mikro.inputs.render import (
    RGB_PRIMARIES,
    color_for_wavelength,
    composite_graph,
    fallback_color,
    rgba,
)
from mikro.pyramid import axes_for

from converter_metadata import device, devices, setting, text
from leica_instruments import GENERIC, LeicaAcquisition, read_acquisition, read_channels
from leica_metadata import (
    LUT_RGB,
    LeicaChannel,
    LeicaDimension,
    LeicaScan,
    dimensions,
    image_record,
)
from logos import declare_logos, logo_of

logger = logging.getLogger(__name__)

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.0.1"

app = App(
    "lif-converter",
    __version__,
    description=(
        "Converts Leica LIF files into mikro datasets natively, one per image (one per "
        "tile for a tile scan), with Leica's own metadata: sequential settings, "
        "detectors, laser lines, pinhole, display windows and stage positions."
    ),
    author="jhnnsrs",
    scopes=["read", "write"],
)
app.service(mikro_service)
declare_logos(app)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The axes the world below is built over. Every scale and offset is derived by
# intersecting an image's own dims with this set: `register` raises if a scale or an
# offset names an axis the two spaces do not share.
WORLD_AXES = frozenset({"z", "y", "x"})

# The axis type for each name this converter can emit. Stated rather than inferred:
# the library's fallback calls every unrecognised name SPACE, which is right for
# z/y/x and silently wrong for anything else — and it matters twice over, because
# the pyramid coarsens exactly the SPACE axes.
AXIS_TYPES: dict[str, AxisType] = {
    "t": AxisType.TIME,
    "c": AxisType.CHANNEL,
    "z": AxisType.SPACE,
    "y": AxisType.SPACE,
    "x": AxisType.SPACE,
}

AXIS_LONG_NAMES = {
    "t": "time",
    "c": "channel",
    "z": "depth",
    "y": "height",
    "x": "width",
}

# `ValueHistogramInput.from_array` calls `np.asarray`, so it is the one place a lazy
# image is forced into memory. Above this many bytes per channel slice the histogram
# is computed from the finest pyramid level that fits instead.
HISTOGRAM_BUDGET_BYTES = 256 * 1024 * 1024

# Past this many layers a scene stops being useful. Only a mosaic scene can realistically
# reach it — a 364-tile preview scan is 364 layers. The world still holds every
# registration and outlives the run, so a subset can be staged later.
MAX_SCENE_LAYERS = 48

# How Leica names a merged image: the scan's own name with "merged" appended — "R1_Merged",
# "Series 3 Merged", but also "…_4merged" with no separator at all. Only a fallback: a
# merged image is just as often named something else entirely ("pig_overview_merged"
# for the tiles of "pig overview/Series"), so `merged_partners` matches on geometry first.
MERGED_SUFFIX = re.compile(r"[\s_-]*merged$", re.IGNORECASE)

# How far a merged image's stage centre may sit from the mean of its tiles' centres and
# still be their merge. LAS X writes the one as exactly the other; a micrometer is
# rounding, while two scans on one slide are hundreds apart.
MERGE_CENTRE_TOLERANCE_UM = 1.0

# How closely a merged image's extent has to match its tile grid's. Stitching trims a
# little of the overlap (0.99 on the files checked); a different acquisition at the same
# centre — a zoomed-in field, an overview — is off by a factor.
MERGE_EXTENT_RATIO = (0.9, 1.1)

# An automatic pyramid halves until its coarsest level fits a tile of this many pixels
# on its longer side: enough to open a 12k stitched image in one fetch, and no more
# levels than a 512-pixel field needs.
PYRAMID_TOP_PIXELS = 512

# What this converter reads. XLEF/XLIF/XLCF are folders of XML pointing at sidecar files,
# and only the one file the action was given is ever downloaded.
READABLE_SUFFIXES = (".lif", ".lof")

# What a channel is drawn in when LAS X did not name a LUT colour: its detection band's
# middle, mapped to roughly the colour the eye would have seen down the eyepiece.
COLOR_BY_EMISSION = (
    (400, (150, 0, 255)),
    (450, (75, 0, 255)),
    (490, (0, 130, 255)),
    (520, (0, 255, 200)),
    (560, (0, 255, 0)),
    (590, (255, 255, 0)),
    (625, (255, 150, 0)),
    (700, (255, 0, 0)),
)

FAR_RED = (255, 70, 70)


# ---------------------------------------------------------------------------
# The array
# ---------------------------------------------------------------------------


def image_array(image: Any) -> xr.DataArray:
    """The whole of one LIF image, lazily, over a memory map of its block in the file.

    ``out="memmap"`` maps the bytes where they lie: nothing is decoded into RAM and
    nothing is copied to a scratch file, so a 40 GB stitched image costs page cache, not
    memory. One plane per chunk, because a LIF block is stored plane after plane — a
    chunk boundary anywhere else would read every plane once per chunk across it.

    Dims are liffile's, in stored order (``M C Z Y X``, ``Z C Y X``, ``Y X S`` …), with
    singleton dimensions already squeezed away.
    """
    import dask.array as da

    stored = image.asarray(out="memmap")
    dims = tuple(str(d) for d in image.dims)
    chunks = tuple(
        size if d in ("Y", "X", "S") else 1 for d, size in zip(dims, stored.shape)
    )
    return xr.DataArray(
        da.from_array(stored, chunks=chunks, name=f"lif-{image.uuid or image.path}"),
        dims=dims,
    )


def tile_array(
    whole: xr.DataArray, tile: int | None
) -> tuple[xr.DataArray, dict[str, str], bool]:
    """One tile (or the whole image) as a named array, its extra axis types, is-it-RGB.

    The same reduction the OME converter makes: dims lower-cased, ``c`` kept even at
    extent 1 (the per-channel anchors key off it), any other singleton dropped, and
    anything left that is not t/c/z/y/x declared INDEX so the pyramid does not halve it.

    The last flag is whether ``c`` is a *sample* axis — liffile's ``S``, three or four
    samples of one pixel, i.e. a snapshot. It is decided here, where ``S`` is still
    visible; once renamed to ``c`` it is indistinguishable from three dyes.
    """
    array = whole.isel(M=tile) if tile is not None else whole

    packed_samples = False
    if "S" in array.dims:
        if array.sizes.get("C", 1) > 1:
            # Samples *and* channels: a layout liffile names but this converter has no
            # honest single axis for. Refused rather than reduced to sample 0.
            raise ValueError(
                f"Image has {array.sizes['C']} channels of {array.sizes['S']} samples "
                "each; not flattening them into one channel axis."
            )
        if "C" in array.dims:
            array = array.squeeze("C")
        packed_samples = array.sizes["S"] in (3, 4)
        array = array.rename({"S": "c"})

    array = array.rename({d: str(d).lower() for d in array.dims if str(d) in "TCZYX"})

    if "c" not in array.dims:
        array = array.expand_dims("c")

    dropped = [
        d for d in array.dims if str(d) not in ("c", "y", "x") and array.sizes[d] == 1
    ]
    if dropped:
        array = array.squeeze(dropped, drop=True)

    extra_types = {str(d): "INDEX" for d in array.dims if str(d) not in AXIS_TYPES}

    return (
        xr.DataArray(array.data, dims=[str(d) for d in array.dims]),
        extra_types,
        packed_samples,
    )


def auto_levels(sizes: Mapping[str, int]) -> int:
    """How many pyramid levels an image needs: halvings until it fits `PYRAMID_TOP_PIXELS`.

    Including the base, so a 1024-pixel field is two levels and an 11k stitched image is
    six. A fixed count is wrong in both directions — levels below one tile are wasted
    uploads, and a stitched image stopped at a fixed depth still opens at megapixels.

    Never fewer than two: a one-level pyramid uploads with an empty ``scales`` list, which
    no other converter here sends — one coarser level costs a quarter of a small field.
    """
    side = max(int(sizes.get("y", 1)), int(sizes.get("x", 1)))
    levels = 1
    while side > PYRAMID_TOP_PIXELS or levels < 2:
        side = -(-side // 2)
        levels += 1
    return levels


def image_pyramid(
    data: xr.DataArray,
    levels: int,
    method: str,
    types: Mapping[str, str],
    scratch: str,
) -> list[xr.DataArray]:
    """The pyramid of a memory-mapped base: level 0 left mapped, level 1 persisted.

    mikro uploads each level as a separate computation, and a lazy level is chained all
    the way back to level 0 — so a pyramid left wholly lazy reads the base once per
    level, six passes over a 40 GB stitched image. mikro's own remedy (``materialize``)
    copies the whole base to a scratch Zarr first, in ``TMPDIR`` whatever
    ``MIKRO_SCRATCH_DIR`` says, which for a base that is already a random-access store
    is a full copy for nothing.

    The middle path: level 1 — a quarter of the base, an eighth for a z-stack — is
    written once into `scratch` (a directory the action owns, beside the download, and
    removes once this dataset is uploaded), and every coarser level is built from it. Two passes over the base in
    all: one for level 1, one for the upload of level 0 itself.
    """
    import dask.array as da

    pyramid = build_pyramid(
        data, levels=levels, method=method, types=types, materialize=False
    )
    if len(pyramid) <= 2:
        return pyramid

    path = tempfile.mkdtemp(prefix="level1-", dir=scratch)
    pyramid[1].data.to_zarr(path, overwrite=True)
    level1 = xr.DataArray(da.from_zarr(path), dims=pyramid[1].dims)
    return [
        pyramid[0],
        *build_pyramid(
            level1, levels=len(pyramid) - 1, method=method, types=types, materialize=False
        ),
    ]


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def voxel_scale(
    leica_dims: Mapping[str, LeicaDimension], dims: Sequence[str]
) -> tuple[dict[str, float] | None, str | None]:
    """The per-axis factor from this dataset's indices into micrometers.

    Returns ``(None, reason)`` when the image has no lateral calibration — a snapshot in
    ``Pixel`` units. Registering anyway would positively claim one pixel is one
    micrometer, at the wrong size next to genuinely calibrated images.

    A missing *axial* spacing is different: omitting ``z`` and passing ``z=1.0`` are the
    same claim, so there is no way to abstain. The assumption is recorded instead.
    """
    shared = [d for d in dims if d in WORLD_AXES]
    sizes = {
        axis: leica_dims[label].spacing_um if label in leica_dims else None
        for axis, label in (("x", "X"), ("y", "Y"), ("z", "Z"))
    }
    if sizes["x"] is None or sizes["y"] is None:
        return (
            None,
            "no lateral pixel size in the file — not registered into a physical space",
        )

    note = None
    if "z" in shared and sizes["z"] is None:
        note = "z spacing unknown, registered as 1 um per plane"
    elif "z" in shared and leica_dims["Z"].length < 0:
        note = "z-stack recorded top-down; planes registered in ascending z"

    return {axis: sizes[axis] or 1.0 for axis in shared}, note


@dataclass(frozen=True)
class StageCentre:
    """Where the stage put the centre of one field, in world micrometers."""

    x: float
    y: float
    source: str
    flip_x: bool
    flip_y: bool
    swap_xy: bool


def stage_centre(image: Any, tile: int | None, scan: LeicaScan | None) -> StageCentre | None:
    """The centre of `tile` of `image` (or of the image), from what LAS X recorded.

    ``TileScanInfo`` first: it names one centre per tile, and a merged image carries one
    for itself — exactly the mean of its tiles', which is what lets `merged_partners`
    find it. The scan-wide ``StagePosX/Y`` otherwise.

    Leica's flips apply to the TileScanInfo positions, in which they are recorded. The
    scan-wide flags describe the scanner rather than the stage, so they are kept as
    settings on the anchor and not applied here.
    """
    info = image.tilescan
    if info is not None and len(info):
        index = tile if tile is not None else 0
        if index >= len(info):
            return None
        row = info.tiles[index]
        x, y = float(row["pos_x"]) * 1e6, float(row["pos_y"]) * 1e6
        flips = (bool(info.flip_x), bool(info.flip_y), bool(info.swap_xy))
        if flips[0]:
            x = -x
        if flips[1]:
            y = -y
        if flips[2]:
            x, y = y, x
        return StageCentre(x, y, "TileScanInfo", *flips)
    if scan is not None and scan.stage_x_m is not None and scan.stage_y_m is not None:
        return StageCentre(
            scan.stage_x_m * 1e6, scan.stage_y_m * 1e6, "StagePos", False, False, False
        )
    return None


def stage_offsets(
    centre: StageCentre | None,
    scale: Mapping[str, float],
    data: xr.DataArray,
    leica_dims: Mapping[str, LeicaDimension],
) -> dict[str, float]:
    """Where pixel (0, 0, 0) sits in the world: the stage centre less half the field.

    Pixel centres, so a field of N pixels spans N - 1 spacings between its first and last;
    the same convention as the spacing itself (``Length / (N - 1)``).
    """
    offsets: dict[str, float] = {}
    if centre is not None:
        for axis, value in (("x", centre.x), ("y", centre.y)):
            if axis in data.dims and axis in scale:
                offsets[axis] = value - (data.sizes[axis] - 1) * scale[axis] / 2
    z = leica_dims.get("Z")
    if "z" in data.dims and z is not None and z.unit == "m":
        offsets["z"] = min(z.origin, z.origin + z.length) * 1e6
    return offsets


# ---------------------------------------------------------------------------
# Scenes
#
# Which images are staged, and with what. Every image is uploaded and registered; this
# only decides what is drawn together. A merged image is the picture its tiles make, so
# drawing both draws the same pixels twice.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """What `merged_partners` needs to know about one image, in world micrometers."""

    path: str
    name: str
    tiles: int
    channels: int
    centre: tuple[float, float] | None
    extent: tuple[float, float] | None


def footprint(image: Any, scan: LeicaScan | None) -> Footprint:
    """An image's place on the slide: the mean of its field centres, and the extent of
    the grid those fields cover (one field, for an image that is not a tile scan)."""
    sizes = dict(image.sizes)
    tiles = int(sizes.get("M", 1))
    leica_dims = dimensions(image.xml_element)
    spacing = {
        axis: leica_dims[axis].spacing_um if axis in leica_dims else None
        for axis in ("X", "Y")
    }
    centres = [
        centre
        for tile in (range(tiles) if tiles > 1 else [None])
        if (centre := stage_centre(image, tile, scan)) is not None
    ]

    centre = extent = None
    if centres and len(centres) == (tiles if tiles > 1 else 1):
        xs, ys = [c.x for c in centres], [c.y for c in centres]
        centre = (sum(xs) / len(xs), sum(ys) / len(ys))
        if spacing["X"] is not None and spacing["Y"] is not None:
            field = (
                (sizes.get("X", 1) - 1) * spacing["X"],
                (sizes.get("Y", 1) - 1) * spacing["Y"],
            )
            if centres[0].swap_xy:
                field = (field[1], field[0])
            extent = (
                max(xs) - min(xs) + field[0],
                max(ys) - min(ys) + field[1],
            )

    channels = int(sizes.get("C", 1))
    return Footprint(image.path, image.name, tiles, channels, centre, extent)


def _is_merge_of(merged: Footprint, scan: Footprint) -> bool:
    if merged.centre is None or scan.centre is None:
        return False
    if merged.extent is None or scan.extent is None:
        return False
    distance = np.hypot(
        merged.centre[0] - scan.centre[0], merged.centre[1] - scan.centre[1]
    )
    if distance > MERGE_CENTRE_TOLERANCE_UM:
        return False
    low, high = MERGE_EXTENT_RATIO
    return all(
        low <= merged_side / scan_side <= high
        for merged_side, scan_side in zip(merged.extent, scan.extent)
        if scan_side > 0
    )


def _is_named_merge_of(merged: Footprint, scan: Footprint) -> bool:
    return (
        posixpath.dirname(merged.path) == posixpath.dirname(scan.path)
        and MERGED_SUFFIX.search(merged.name) is not None
        and MERGED_SUFFIX.sub("", merged.name) == scan.name
    )


def merged_partners(footprints: Sequence[Footprint | None]) -> dict[int, int]:
    """Which tile scans have their merged image in the file: ``{scan: merged}``.

    Decided on geometry, not names. LAS X writes a merged image's stage centre as exactly
    the mean of its tiles' centres, and its extent as the tile grid's less a little
    overlap — while the names are whatever the user typed: ``R1`` and ``R1_Merged`` side
    by side, but also ``pig overview/Series`` and ``pig_overview_merged`` a folder apart,
    and two ``R1`` scans in two folders of one file. The name rule (a sibling called the
    scan's name plus "merged") stands in only where the geometry is not recorded.

    Matched on the image **path**, never the name; one merged image answers for one scan
    at most, and both have to have the same channels.
    """
    partners: dict[int, int] = {}
    taken: set[int] = set()
    for i, scan in enumerate(footprints):
        if scan is None or scan.tiles < 2:
            continue
        candidates = [
            j
            for j, other in enumerate(footprints)
            if other is not None
            and other.tiles == 1
            and other.channels == scan.channels
            and j not in taken
        ]
        by_geometry = [j for j in candidates if _is_merge_of(footprints[j], scan)]
        by_name = [
            j
            for j in candidates
            if (footprints[j].centre is None or scan.centre is None)
            and _is_named_merge_of(footprints[j], scan)
        ]
        found = (by_geometry or by_name or [None])[0]
        if found is not None:
            partners[i] = found
            taken.add(found)
    return partners


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def series_optikit(
    file: File,
    image: Any,
    tile: int | None,
    acquisition: LeicaAcquisition,
    leica_dims: Mapping[str, LeicaDimension],
    centre: StageCentre | None,
    scale_note: str | None,
) -> OptikitStateInput | None:
    """The hardware state of the whole image (or tile).

    The instrument's own devices come from its model (`leica_instruments`): the system
    it is, its objective and scanner — or camera, on a widefield system.
    """
    scan = acquisition.scan
    system = acquisition.system
    z = leica_dims.get("Z")
    stage = None
    if centre is not None:
        stage = StageStateInput(
            x=f"{centre.x:g} um",
            y=f"{centre.y:g} um",
            z=f"{z.origin * 1e6:g} um" if z is not None and z.unit == "m" else None,
        )

    info = image.tilescan
    field = info.tiles[tile] if info is not None and tile is not None and tile < len(info) else None
    timestamps = image.timestamps
    t = leica_dims.get("T")

    state_devices = devices(
        *acquisition.devices(),
        # The run itself. The only home the source file's identity has, so `source-file`
        # and `source-file-id` are load-bearing rather than decorative.
        device(
            "acquisition",
            "software",
            setting("software", string=(system.software if system else None) or "LAS X"),
            setting("source-file", string=file.name),
            setting("source-file-id", string=str(file.id)),
            setting("image-path", string=text(image.path)),
            setting("image-name", string=text(image.name)),
            setting("unique-id", string=text(image.uuid)),
            setting("tile-index", number=float(tile) if tile is not None else None),
            setting(
                "tile-count",
                number=float(image.sizes["M"]) if "M" in image.sizes else None,
            ),
            setting(
                "field-x", number=float(field["field_x"]) if field is not None else None
            ),
            setting(
                "field-y", number=float(field["field_y"]) if field is not None else None
            ),
            setting(
                "acquisition-start",
                string=str(timestamps[0]) if timestamps is not None and len(timestamps) else None,
            ),
            setting("pixel-type", string=str(image.dtype)),
            *(
                setting(f"size-{label.lower()}", number=float(dim.count))
                for label, dim in leica_dims.items()
            ),
            *(
                setting(f"spacing-{label.lower()}", quantity=f"{dim.spacing_um:g} um")
                for label, dim in leica_dims.items()
                if label in "XYZ" and dim.spacing_um is not None
            ),
            setting(
                "time-increment",
                quantity=f"{abs(t.length) / (t.count - 1):g} s"
                if t is not None and t.unit == "s" and t.count > 1 and t.length
                else None,
            ),
            # Says out loud what the registration had to assume. There is no way to
            # express "unknown" through a BY_DIMENSION edge.
            setting("scale-note", string=scale_note),
        ),
        device(
            "stage",
            "stage",
            setting("position-source", string=centre.source if centre else None),
            setting(
                "flip-x",
                flag=centre.flip_x if centre else (scan.flip_x if scan else None),
            ),
            setting(
                "flip-y",
                flag=centre.flip_y if centre else (scan.flip_y if scan else None),
            ),
            setting(
                "swap-xy",
                flag=centre.swap_xy if centre else (scan.swap_xy if scan else None),
            ),
            setting("scanner-flip-x", flag=scan.flip_x if scan else None),
            setting("scanner-flip-y", flag=scan.flip_y if scan else None),
            setting("scanner-swap-xy", flag=scan.swap_xy if scan else None),
        ),
    )

    if stage is None and not state_devices:
        return None
    return OptikitStateInput(stage=stage, devices=state_devices)


def channel_label(channel: LeicaChannel | None, index: int, implicit_rgb: bool) -> str:
    if implicit_rgb:
        return (
            ("Red", "Green", "Blue", "Alpha")[index] if index < 4 else f"Sample {index}"
        )
    if channel is not None and channel.label:
        return channel.label
    return f"Channel {index}"


def channel_color(
    channel: LeicaChannel | None, index: int, implicit_rgb: bool, total: int
) -> tuple[int, int, int, int]:
    """What to draw this channel in.

    The LUT LAS X displayed it in first — the one of these that is a recorded fact rather
    than a guess. Then the middle of its detection band, which at least lands in the
    right part of the spectrum. Then a palette, so that two channels never silently
    collide.
    """
    if implicit_rgb:
        # Index 3 of an RGBA pixel is alpha — a mask, not a colour. White labels it
        # without painting it the same red as channel 0.
        if index >= 3:
            return rgba((255, 255, 255))
        return rgba(RGB_PRIMARIES[index])

    if channel is not None:
        if channel.lut in LUT_RGB:
            return rgba(LUT_RGB[channel.lut])
        if channel.detector is not None and channel.detector.band_nm is not None:
            middle = sum(channel.detector.band_nm) / 2
            return color_for_wavelength(middle, COLOR_BY_EMISSION, FAR_RED)

    return fallback_color(index, total)


def leica_window(
    channel: LeicaChannel | None, dtype: Any
) -> tuple[float, float] | None:
    """The contrast window LAS X showed this channel in, in its own values, or None.

    Leica keeps the window as fractions of the channel's full range, and the range is the
    ``Resolution`` it was recorded at — 12-bit data in a 16-bit array tops out at 4095,
    not 65535. Integer data only: a float channel has no full range to be a fraction of.
    """
    if channel is None or channel.window is None:
        return None
    dtype = np.dtype(dtype)
    if dtype.kind not in "ui":
        return None
    full = (
        float(2**channel.bits - 1)
        if channel.bits and channel.bits <= dtype.itemsize * 8
        else float(np.iinfo(dtype).max)
    )
    low, high = channel.window.black * full, channel.window.white * full
    if high <= low:
        return None
    return low, high


@dataclass(frozen=True)
class ChannelSpec:
    """One channel, as both the anchors and the layers need it.

    Computed once because the histogram is the expensive part — it is the render
    metadata the anchor carries, and computing it twice would double the only forced
    materialisation in the converter.
    """

    index: int
    label: str
    color: tuple[int, int, int, int]
    histogram: ValueHistogramInput
    clim: tuple[float, float]
    leica: LeicaChannel | None


def display_window(shown: np.ndarray) -> tuple[float, float]:
    """The contrast window a channel's layer opens with: p1/p99 of what it *shows*.

    `shown` is the picture, which is not the voxels wherever the layer projects — see
    `channel_specs`. p1/p99 rather than min/max, because a single hot pixel should not
    decide the display range of the whole channel — except where the percentiles coincide
    (a channel that is over 99 % one value), which would be a window of no width.
    """
    shown = shown[np.isfinite(shown)]
    low, high = (float(v) for v in np.percentile(shown, [1, 99]))
    if high <= low:
        low, high = float(shown.min()), float(shown.max())
    if high <= low:
        high = low + 1.0
    return low, high


def channel_specs(
    data: xr.DataArray,
    pyramid: Sequence[xr.DataArray],
    leica: Sequence[LeicaChannel],
    implicit_rgb: bool,
    is_volume: bool,
) -> list[ChannelSpec]:
    """One spec per channel of the array.

    A contrast window is per channel by nature, which is why each histogram is computed
    from that channel's slice alone rather than from the stack. The window itself is
    LAS X's where the file recorded one (`leica_window`), and only otherwise derived from
    the picture — which also saves the extra read a projection costs.
    """
    specs: list[ChannelSpec] = []
    total = data.sizes["c"]

    for index in range(total):
        # Packed samples are three array channels behind one LIF channel description.
        channel = leica[index] if index < len(leica) and not implicit_rgb else None

        base = data.isel(c=index)
        slice_ = base
        if base.nbytes > HISTOGRAM_BUDGET_BYTES:
            # `from_array` materialises whatever it is handed, so a multi-gigabyte
            # channel is summarised from the finest pyramid level that fits. Max
            # downsampling keeps only block maxima, so every level shifts the
            # distribution up — hence the finest, not the coarsest.
            slice_ = next(
                (
                    level.isel(c=index)
                    for level in pyramid[1:]
                    if level.isel(c=index).nbytes <= HISTOGRAM_BUDGET_BYTES
                ),
                pyramid[-1].isel(c=index),
            )

        values = np.asarray(slice_)

        clim = leica_window(channel, data.dtype)
        if clim is None:
            # What the layer draws. A z-stack is drawn as a maximum projection, and the
            # maxima along z are a different population from the voxels: in sparse
            # fluorescence the stack's p99 sits barely above background while nearly
            # every projected pixel holds something brighter.
            shown = values
            if is_volume:
                projection = base.max("z")
                if slice_ is base:
                    shown = values.max(axis=slice_.dims.index("z"))
                elif projection.nbytes <= HISTOGRAM_BUDGET_BYTES:
                    shown = np.asarray(projection)
                else:
                    shown = values.max(axis=slice_.dims.index("z"))
            clim = display_window(shown)

        specs.append(
            ChannelSpec(
                index=index,
                label=channel_label(channel, index, implicit_rgb),
                color=channel_color(channel, index, implicit_rgb, total),
                histogram=ValueHistogramInput.from_array(values),
                clim=clim,
                leica=channel,
            )
        )

    return specs


def channel_anchors(
    specs: Sequence[ChannelSpec],
    acquisition: LeicaAcquisition,
    sample_label: str,
    leica_dims: Mapping[str, LeicaDimension] | None = None,
) -> list[CoordinateAnchorInput]:
    """One anchor per channel: its contrast window, its identity, its optics — laid out
    the way the instrument that recorded it is built."""
    return [
        CoordinateAnchorInput(
            axis_anchors=(AxisAnchorInput(axis="c", value=spec.index),),
            value_histogram=spec.histogram,
            label=LabelInput(label=spec.label),
            microscope=(
                acquisition.channel_devices(spec.leica) if spec.leica is not None else None
            ),
            light_graph=(
                acquisition.channel_lightpath(spec.leica, sample_label, leica_dims)
                if spec.leica is not None
                else None
            ),
        )
        for spec in specs
    ]


def channel_lens(dataset: Any, index: int) -> Any:
    """The lens that pins channel `index`: what a layer over one channel draws.

    A layer draws its lens and nothing narrower, so a channel is a lens of its own.
    A dataset with no channel axis is one signal already, and its whole lens is it.
    """
    if "c" not in dataset.axis_names:
        return dataset.lens()
    return dataset.lens(c=index)


def author_channel_layers(
    mikro: Mikro,
    dataset: ArrayDataset,
    scene: Any,
    specs: Sequence[ChannelSpec],
    is_volume: bool,
    implicit_rgb: bool,
    order_offset: int = 0,
) -> int:
    """One layer per channel of `dataset`, composited into `scene`.

    Per channel, because one layer over a multi-channel dataset means one recipe standing
    in for every channel: no per-channel colour, no per-channel contrast, nothing to
    toggle. It is also the only place LAS X's own LUT colour can land.

    The one exception is a packed RGB(A) series, one layer over its samples: those
    "channels" are one photograph, and separating them gives toggles that are only ever
    right in one combination. `implicit_rgb` is trusted only as far as it can be checked —
    this function refuses it for any channel count that is not 3 or 4. That refusal is the
    backstop for the whole rule, kept here rather than left to the server because it fires
    first and demotes instead of failing the upload.

    One lens per channel, beside the dataset's whole one: a layer draws its lens and
    nothing narrower, so the channel a layer shows is the one its lens pins.

    **Which kind of layer** — each recipe has a server kind that states it as fields:

    * one channel, flat → ``create_intensity_layer``
    * one channel, through a z-stack → ``create_volume_layer``
    * a packed RGB(A) pixel, flat → ``create_rgb_layer``

    ``create_layer`` and a render graph are left for the one shape none of them can say,
    a packed RGB(A) **z-stack**: an RGB layer carries no projection mode.

    Returns the number of layers created — not ``len(specs)`` in the RGB case.
    """
    lens = dataset.lens()

    if implicit_rgb and len(specs) not in (3, 4):
        logger.warning(
            "Refusing to composite %s channel(s) into one RGB layer; a render graph "
            "covers exactly one channel unless the channels are the 3 or 4 samples of "
            "one pixel. Authoring one layer per channel instead.",
            len(specs),
        )
        implicit_rgb = False

    if implicit_rgb:
        if is_volume:
            # The one shape the flat kinds cannot say. Three primaries even for RGBA:
            # the fourth sample is alpha, a mask rather than a colour. `rgb_graph` says
            # nearly this, but its bare triples are what the server rejects.
            mikro.create_layer(
                lens=lens.id,
                scene=scene.id,
                render_graph=composite_graph(
                    [
                        {
                            "lens": channel_lens(dataset, index).id,
                            "color": rgba(primary),
                            "colormap": ColorMap.INTENSITY,
                            # Already display-referred; stretching it would change the
                            # colours the image was authored with.
                            "clim_min": 0.0,
                            "clim_max": 255.0,
                            "mode": ProjectionMode.MIP,
                        }
                        for index, primary in enumerate(RGB_PRIMARIES)
                    ]
                ),
                blending=Blending.ADDITIVE,
                visible=True,
                order=order_offset,
            )
            return 1

        # Flat: the photograph is exactly what an RGB layer is for, and it says so as
        # three component lenses, one per sample, rather than as a tree a reader has to
        # recognise. One contrast window across all three — they are components of one
        # picture, not three signals with their own scaling — left at the full 8-bit
        # range for the reason above.
        #
        # No `blending` argument here, and none accepted: the server forces NORMAL,
        # because a photograph sits *over* what is beneath it rather than summing with
        # it. That is a change from the ADDITIVE this path used to pass.
        mikro.create_rgb_layer(
            lens=lens.id,
            scene=scene.id,
            red_lens=channel_lens(dataset, 0).id,
            green_lens=channel_lens(dataset, 1).id,
            blue_lens=channel_lens(dataset, 2).id,
            clim_min=0.0,
            clim_max=255.0,
            visible=True,
            order=order_offset,
        )
        return 1

    for spec in specs:
        # `color`, not a colormap: what this converter knows about a channel is an RGBA
        # quad — LAS X's LUT colour, or the one its detection band implies —
        # and no named map means it. That fact used to force the whole layer onto a
        # render graph; an intensity layer carries it as a field now.
        settings = dict(
            lens=channel_lens(dataset, spec.index).id,
            scene=scene.id,
            color=spec.color,
            # Of what the layer shows, not of the voxels — see `channel_specs`.
            clim_min=spec.clim[0],
            clim_max=spec.clim[1],
            blending=Blending.ADDITIVE,
            visible=True,
            order=order_offset + spec.index,
        )

        if is_volume:
            # A z-stack has no single plane to show, so each channel projects through the
            # volume rather than picking one slice. Same kind of layer, one field more.
            mikro.create_volume_layer(mode=ProjectionMode.MIP, **settings)
        else:
            mikro.create_intensity_layer(**settings)

    return len(specs)


def compose_scene(
    mikro: Mikro, name: str, coordinate_system: Any, entries: Sequence[Converted]
) -> Scene:
    """A scene over `coordinate_system`, with every layer authored explicitly.

    ``create_scene`` rather than ``stage`` / ``create_scene_from_coordinate_system``:
    the bootstrap decides on your behalf how many registered sources become layers,
    which recipe each one gets, and — for anything multi-channel — collapses a whole
    dataset into a single layer. This converter knows exactly what it uploaded and what
    each channel is, so none of those are decisions worth delegating. It adopts the
    world system, then authors one lens per dataset and one layer per channel, each
    with its own colour, contrast window and projection.

    `entries` is one series, or the tiles of one mosaic — never the whole file; which
    series share a scene is the action's decision (see `merged_partners`), not this
    function's.

    The scene carries only what is genuinely scene-wide. ``blending`` is ADDITIVE
    because that is how fluorescence channels combine — and how three RGB primaries
    recompose a photograph. ``preferred_view`` is a viewer hint and nothing more: the
    projection that actually renders a volume lives on each layer's channel node, so
    this only says which way a viewer should open, and only claims a preference when
    something in the scene has depth to look into.
    """
    scene = mikro.create_scene(
        name=name,
        coordinate_system=coordinate_system.id,
        blending=Blending.ADDITIVE,
        preferred_view=(
            PreferredView.THREE_D
            if any(entry.is_volume for entry in entries)
            else PreferredView.TWO_D
        ),
    )

    # Ordered so a series' channels stay together in the layer stack rather than
    # interleaving with the next series'.
    order = 0
    for entry in entries:
        order += author_channel_layers(
            mikro,
            entry.dataset,
            scene,
            entry.specs,
            entry.is_volume,
            entry.implicit_rgb,
            order,
        )

    logger.info(
        "Scene %s over %s: %s layer(s) from %s dataset(s)",
        scene.id,
        scene.world_coordinate_system.name,
        order,
        len(entries),
    )
    return scene


def stage_group(
    mikro: Mikro, title: str, world: Any, entries: Sequence[Converted]
) -> None:
    """Compose the scene(s) for one group out of `scene_groups`.

    The registered series in it share one scene over the world. An uncalibrated series is
    not in the world, so it is staged in its own pixel grid instead — the honest frame for
    it, and one it shares with nothing.
    """
    registered = [entry for entry in entries if entry.registered]
    if registered:
        layer_count = sum(entry.layer_count for entry in registered)
        if layer_count > MAX_SCENE_LAYERS:
            logger.warning(
                "%s would compose %s layers from %s series; not building a scene over "
                "more than %s. The world holds every registration and outlives this run, "
                "so a subset can be composed later.",
                title,
                layer_count,
                len(registered),
                MAX_SCENE_LAYERS,
            )
        else:
            compose_scene(mikro, title, world, registered)

    for entry in entries:
        if entry.registered:
            continue
        if entry.dataset.intrinsic_system is None:
            logger.warning(
                "Series %s has no intrinsic coordinate system to compose over; the "
                "dataset is uploaded but has no scene.",
                entry.series,
            )
            continue
        compose_scene(mikro, entry.name, entry.dataset.intrinsic_system, [entry])


# ---------------------------------------------------------------------------
# One image, or one tile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Converted:
    """What the action needs to know about a series after it has landed."""

    series: int
    name: str
    dataset: ArrayDataset
    specs: list[ChannelSpec]
    registered: bool
    is_volume: bool
    implicit_rgb: bool

    @property
    def layer_count(self) -> int:
        """How many layers this series composes into a scene.

        Not ``len(specs)``: a packed RGB(A) series is one layer over its 3 or 4
        samples. Kept here rather than recomputed at each call site so the budget check
        in `stage_group` and what `author_channel_layers` actually creates cannot drift.

        For every other series this is ``len(specs)`` — one layer per channel — and that
        is the invariant, not an implementation detail.
        """
        return 1 if self.implicit_rgb else len(self.specs)


def packs_one_pixel(packed_samples: bool, channels: int, dtype: Any) -> bool:
    """Whether this image's channels are the samples of one pixel — the ONLY exception.

    ONE LAYER PER CHANNEL. Exactly one thing may say yes: liffile's sample axis
    (``packed_samples`` out of `tile_array`), which it exposes only where the channel
    descriptions are the red, green and blue of one pixel — a LAS X snapshot. Nothing
    that *counts* channels may: collapsing three dyes destroys per-channel colour,
    contrast and toggling.

    And the samples have to be 8-bit. An RGB layer is drawn unstretched over 0..255; a
    deeper sample axis is a colour camera's acquisition, and gets a layer and a window
    per sample instead.
    """
    return packed_samples and channels in (3, 4) and np.dtype(dtype) == np.uint8


def convert_tile(
    mikro: Mikro,
    file: File,
    image: Any,
    whole: xr.DataArray,
    tile: int | None,
    acquisition: LeicaAcquisition,
    leica: Sequence[LeicaChannel],
    leica_dims: Mapping[str, LeicaDimension],
    world: Any,
    levels: int,
    method: str,
    scratch: str,
) -> Converted:
    """Upload one tile of an image (or the whole image), and place it if the file says
    where. Never stages it: that depends on the images around it — see `merged_partners`.
    """
    array, extra_types, packed_samples = tile_array(whole, tile)
    scan = acquisition.scan

    # Canonical order — time, then channel, then space — is load-bearing: the render
    # axes are derived from the *position* of the spatial axes.
    data = canonical(array, types=extra_types)
    dims = [str(d) for d in data.dims]

    implicit_rgb = packs_one_pixel(packed_samples, data.sizes["c"], data.dtype)
    if packed_samples and not implicit_rgb:
        logger.warning(
            "%s has a packed sample axis but %s %s sample(s) are no 8-bit RGB(A) "
            "picture; treating them as channels, one layer each.",
            image.path,
            data.sizes["c"],
            data.dtype,
        )
    is_volume = data.sizes.get("z", 1) > 1

    pyramid = image_pyramid(
        data, levels or auto_levels(data.sizes), method, extra_types, scratch
    )
    scales = scales_from(pyramid, method)

    scale, scale_note = voxel_scale(leica_dims, dims)
    centre = stage_centre(image, tile, scan)
    specs = channel_specs(data, pyramid, leica, packed_samples, is_volume)

    label = text(image.name) or image.path
    if tile is not None:
        label = f"{label} (tile {tile})"
    name = f"{file.name} - {image.path}" + (f" (tile {tile})" if tile is not None else "")

    dataset = mikro.create_array_dataset(
        data=pyramid[0],
        scales=scales,
        name=name,
        axes=axes_for(
            data, types=dict.fromkeys(extra_types, "INDEX"), long_names=AXIS_LONG_NAMES
        ),
        anchors=[
            CoordinateAnchorInput(
                axis_anchors=tuple(
                    AxisAnchorInput(axis=str(d), value=0) for d in data.dims
                ),
                label=LabelInput(label=label),
                ome_metadata=OmeMetadataInput(
                    metadata_string=image_record(
                        file.name,
                        str(file.id),
                        image.path,
                        image.uuid,
                        tile,
                        image.xml_element,
                    )
                ),
                microscope=series_optikit(
                    file, image, tile, acquisition, leica_dims, centre, scale_note
                ),
            ),
            *channel_anchors(specs, acquisition, text(image.name) or "Sample", leica_dims),
        ],
        # The link back to the bytes these arrays were read out of. The identifier is
        # Leica's own UniqueID of the image, plus the tile — what survives a reader that
        # lists images in another order.
        source_files=[
            SourceFileInput(
                file=file.id,
                series_identifier=(image.uuid or image.path)
                + (f":{tile}" if tile is not None else ""),
                # liffile hands back the stored values and this converter only renames
                # and reorders axes: the histograms are true of the file too.
                value_relation=ValueRelation.IDENTICAL,
            )
        ],
    )

    if scale is None:
        logger.info("%s: %s", name, scale_note)
        return Converted(tile or 0, name, dataset, specs, False, is_volume, implicit_rgb)

    world.register(dataset, scale=scale, **stage_offsets(centre, scale, data, leica_dims))
    return Converted(tile or 0, name, dataset, specs, True, is_volume, implicit_rgb)


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


@app.action(
    name="Convert Leica LIF file to Datasets",
    description=(
        "Read a Leica LIF file natively (no Bio-Formats) and upload every image as a "
        "pyramided dataset — every tile, for a tile scan — with Leica's own metadata: "
        "sequential settings, detectors, laser lines, pinhole and display windows, "
        "placed in a shared micrometer space at its stage position and staged in a scene "
        "of its own. Tiles whose merged image is in the file are uploaded but not staged."
    ),
    collections=["conversion"],
    logo=logo_of("convert_lif_file_to_datasets"),
)
def convert_lif_file_to_datasets(
    file: File,
    mikro: Mikro,
    task: Task,
    levels: int = 0,
    method: str = "max",
    stage_scene: bool = True,
    world_name: str = "",
) -> list[ArrayDataset]:
    """Convert every image of a LIF file into a dataset, one per tile of a tile scan.

    `levels` of 0 picks the pyramid depth per image (`auto_levels`). Returns a list
    rather than yielding, for the reason the OME converter gives: a mosaic scene is
    composed after its last tile, and an abandoned generator would leave it unstaged.
    """
    assert file.store, "No file provided"
    if not file.name.lower().endswith(READABLE_SUFFIXES):
        raise ValueError(
            f"{file.name} is not a LIF or LOF file. XLEF/XLIF experiments reference "
            "sidecar files that are not part of the upload; export them as LIF."
        )

    task.progress(0, "Downloading file")
    # Under its own name, into a scratch dir of its own — never the working directory,
    # which the `finally` below would then delete from. MIKRO_SCRATCH_DIR when set,
    # TMPDIR otherwise; a tmpfs /tmp puts the download in RAM.
    workdir = tempfile.mkdtemp(
        prefix="lif-converter-", dir=os.environ.get("MIKRO_SCRATCH_DIR") or None
    )

    try:
        path = file.store.download(os.path.join(workdir, os.path.basename(file.name)))
        task.progress(5, "Reading the LIF header")

        from liffile import LifFile

        with LifFile(path) as lif:
            images = []
            for image in lif.images:
                if image.is_flim:
                    logger.warning(
                        "%s is a FLIM/TCSPC image; this converter does not read those.",
                        image.path,
                    )
                    continue
                images.append(image)

            # One per invocation: a physical space belongs to nobody and outlives every
            # dataset put into it. Two invocations mint two disjoint worlds.
            world = space_3d(
                mikro, world_name or f"{file.name} (micrometers)", unit="micrometer"
            )

            acquisitions: list[LeicaAcquisition] = []
            footprints: list[Footprint | None] = []
            for image in images:
                acquisition = LeicaAcquisition(GENERIC, None, None, None)
                place = None
                try:
                    acquisition = read_acquisition(image.xml_element)
                    logger.info(
                        "%s was recorded on %s (model %s)",
                        image.path,
                        (acquisition.system.name if acquisition.system else None)
                        or "an unnamed system",
                        acquisition.instrument.key,
                    )
                    place = footprint(image, acquisition.scan)
                except Exception:
                    # Only costs the merge decision and the instrument model: the image
                    # itself still converts.
                    logger.exception("Could not read where %s sits", image.path)
                acquisitions.append(acquisition)
                footprints.append(place)

            # Decided up front, from the header alone: a merged image anywhere in the
            # file decides whether its tiles are staged, and the tiles may come first.
            merged = merged_partners(footprints)
            for scan_index, merged_index in merged.items():
                logger.info(
                    "%s is the merge of the %s tiles of %s; the tiles are uploaded and "
                    "registered, but only the merged image is staged.",
                    images[merged_index].path,
                    footprints[scan_index].tiles,
                    images[scan_index].path,
                )

            units = sum(f.tiles if f else 1 for f in footprints) or 1
            done = 0
            datasets: list[ArrayDataset] = []
            failures: list[tuple[str, Exception]] = []

            for index, image in enumerate(images):
                tiles = int(image.sizes.get("M", 1))
                tile_indices: list[int | None] = list(range(tiles)) if "M" in image.sizes else [None]
                try:
                    whole = image_array(image)
                    leica = read_channels(image.xml_element)
                    leica_dims = dimensions(image.xml_element)
                except Exception as exc:
                    logger.exception("%s could not be read", image.path)
                    failures.append((image.path, exc))
                    done += len(tile_indices)
                    continue

                entries: list[Converted] = []
                for tile in tile_indices:
                    task.progress(
                        10 + int(85 * done / units),
                        f"Converting {image.path}"
                        + (f" tile {tile + 1}/{tiles}" if tile is not None else ""),
                    )
                    done += 1
                    # Per tile, and gone once it is uploaded: 98 tiles' persisted levels
                    # would otherwise all wait for the end of the file.
                    scratch = tempfile.mkdtemp(prefix="tile-", dir=workdir)
                    try:
                        entries.append(
                            convert_tile(
                                mikro,
                                file,
                                image,
                                whole,
                                tile,
                                acquisitions[index],
                                leica,
                                leica_dims,
                                world,
                                levels,
                                method,
                                scratch,
                            )
                        )
                    except Exception as exc:  # one bad tile must not cost the other 97
                        logger.exception("%s tile %s failed", image.path, tile)
                        failures.append((f"{image.path} [{tile}]", exc))
                    finally:
                        shutil.rmtree(scratch, ignore_errors=True)

                datasets.extend(entry.dataset for entry in entries)

                if not stage_scene or not entries or index in merged:
                    continue
                title = (
                    entries[0].name
                    if len(entries) == 1
                    else f"{file.name} - {image.path} ({len(entries)} tiles)"
                )
                try:
                    stage_group(mikro, title, world, entries)
                except Exception:  # the datasets are uploaded; a scene is not worth them
                    logger.exception("Could not stage %s", title)

        if failures and not datasets:
            raise RuntimeError(
                f"All {len(failures)} image(s) failed; first: {failures[0][1]}"
            ) from failures[0][1]
        if failures:
            logger.warning(
                "%s image(s) or tile(s) failed: %s",
                len(failures),
                [where for where, _ in failures],
            )

        task.progress(100, f"Converted {len(datasets)} dataset(s)")
        return datasets

    finally:
        # The whole scratch directory, and only ever that directory, which this
        # converter made and owns.
        shutil.rmtree(workdir, ignore_errors=True)
        if os.path.exists(workdir):
            # Cleanup must never mask whatever actually went wrong above.
            logger.warning("Could not remove the scratch directory %s", workdir)


if __name__ == "__main__":
    run(app)
