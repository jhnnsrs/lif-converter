"""Leica metadata: reading a LIF image's own XML into mikro inputs.

A LIF file is a UTF-16 XML header followed by raw memory blocks, and the header is
LAS X's own record of the acquisition — not a translation of it. Everything here walks
one image's ``Element`` (``LifImage.xml_element``) with plain ElementTree, so it can be
tested on a fragment of XML without a file, and so nothing passes through OME on the way.

Where the facts live, as checked against real files from an SP8 / STELLARIS:

* ``Data/Image/ImageDescription/Dimensions/DimensionDescription`` — one per axis
  (``DimID`` 1=X, 2=Y, 3=Z, 4=T, 10=M for the tiles of a tile scan). Lengths are in
  ``Unit`` (``m`` for a calibrated axis, ``Pixel`` for a snapshot), ``Length`` runs from
  the first element to the last, so the spacing is ``Length / (N - 1)``, and ``Origin``
  is the absolute position of the first element — the focus position for Z.
* ``…/Channels/ChannelDescription`` — one per channel: ``LUTName``, ``Resolution`` (bits),
  and ``ChannelProperty`` pairs naming the ``DetectorName``, the ``DyeName`` and the
  ``SequentialSettingIndex`` that recorded it. That last pair is the link from a channel
  to the hardware: see `leica_channels`.
* ``Attachment[@Name="HardwareSetting"]`` — the scan-wide ``ATL…SettingDefinition``
  (objective, pinhole, zoom, scan speed, averaging, stage), and under
  ``LDM_Block_Sequential/LDM_Block_Sequential_List`` one more per sequential setting,
  each with its active detectors, AOTF laser lines and spectral detection bands.
* ``Attachment[@Name="ViewerScaling"]`` — the display window LAS X last showed each
  channel in, as fractions of the full range.
* ``Attachment[@Name="TileScanInfo"]`` — the stage centre of every tile (read through
  ``LifImage.tilescan``, which parses the same element).

This module reads; it does not decide what kind of microscope it is reading. That, and
the light path laid out the way that microscope is built, is `leica_instruments`; the
`channel_lightpath` here is the generic one it falls back to.

Absent facts stay absent, as in the OME converter: every reader here returns ``None``
for a missing or empty attribute, and a channel whose hardware link does not check out
carries only what its own ``ChannelDescription`` says.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from mikro.api.schema import (
    BeamSplitterElementInput,
    BeamStateInput,
    ChannelKind,
    DetectorElementInput,
    LaserElementInput,
    LightEdgeInput,
    LightpathGraphInput,
    ObjectiveElementInput,
    ObjectiveImmersion,
    OpticalElementInput,
    OptikitStateInput,
    PinholeElementInput,
    PortRole,
    SampleElementInput,
    SpectrumInput,
)
from mikro.inputs.lightpath import in_out as _in_out
from mikro.inputs.lightpath import port as _port

from converter_metadata import device, devices, setting, text

logger = logging.getLogger(__name__)

# Leica's dimension ids, as far as this converter places them. Anything else (lambda,
# rotation, …) still reaches the array as an INDEX axis; it just has no place in space.
DIMENSION_LABEL = {1: "X", 2: "Y", 3: "Z", 4: "T", 10: "M"}

# The LUT names LAS X ships, as the colour each draws a channel in. A user-defined LUT
# carries any other name, and has no colour this converter can claim — it falls through
# to the detection band, then to the palette.
LUT_RGB: dict[str, tuple[int, int, int]] = {
    "Gray": (255, 255, 255),
    "Grey": (255, 255, 255),
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (0, 0, 255),
    "Cyan": (0, 255, 255),
    "Magenta": (255, 0, 255),
    "Yellow": (255, 255, 0),
}

# How Leica spells an immersion, in the objective's `Immersion` attribute and in its name.
LEICA_IMMERSION = {
    "OIL": ObjectiveImmersion.OIL,
    "WATER": ObjectiveImmersion.WATER,
    "W": ObjectiveImmersion.WATER,
    "GLYC": ObjectiveImmersion.GLYCEROL,
    "GLYCEROL": ObjectiveImmersion.GLYCEROL,
    "DRY": ObjectiveImmersion.AIR,
    "AIR": ObjectiveImmersion.AIR,
    "IMM": ObjectiveImmersion.MULTI,
}

# An AOTF line at this intensity or below is off. LAS X writes a closed line as zero, or
# as floating-point dust like 1.4e-16 — which is not a laser that fired.
LINE_OFF_PERCENT = 1e-6


# ---------------------------------------------------------------------------
# Attribute readers
# ---------------------------------------------------------------------------


def attr_text(element: ET.Element | None, name: str) -> str | None:
    if element is None:
        return None
    return text(element.get(name))


def attr_float(element: ET.Element | None, name: str) -> float | None:
    value = attr_text(element, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def attr_flag(element: ET.Element | None, name: str) -> bool | None:
    value = attr_text(element, name)
    if value is None:
        return None
    return value not in ("0", "false", "False")


def um(meters: float | None) -> str | None:
    """A length Leica wrote in metres, as a Length scalar string in micrometers."""
    if meters is None:
        return None
    return f"{meters * 1e6:g} um"


def nm(nanometers: float | None) -> str | None:
    """A wavelength Leica wrote in plain nanometres, as a Length scalar string."""
    if nanometers is None:
        return None
    return f"{nanometers:g} nm"


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeicaDimension:
    label: str
    count: int
    origin: float
    length: float
    unit: str

    @property
    def spacing_um(self) -> float | None:
        """Micrometers from one element to the next, or None when that is not a length.

        ``None`` for an axis in pixels (a snapshot), for a single element (there is no
        next one), and for a zero length — which is what Leica writes for an axis it did
        not calibrate rather than a claim that the elements coincide. The magnitude
        only: a z-stack recorded top-down has a negative ``Length``.
        """
        if self.unit != "m" or self.count < 2 or self.length == 0:
            return None
        return abs(self.length) / (self.count - 1) * 1e6


def dimensions(image_el: ET.Element) -> dict[str, LeicaDimension]:
    found: dict[str, LeicaDimension] = {}
    for dim in image_el.findall(
        "./Data/Image/ImageDescription/Dimensions/DimensionDescription"
    ):
        label = DIMENSION_LABEL.get(int(dim.get("DimID", "0")))
        if label is None or label in found:
            continue
        found[label] = LeicaDimension(
            label=label,
            count=int(dim.get("NumberOfElements", "1")),
            origin=attr_float(dim, "Origin") or 0.0,
            length=attr_float(dim, "Length") or 0.0,
            unit=dim.get("Unit", ""),
        )
    return found


# ---------------------------------------------------------------------------
# The scan
# ---------------------------------------------------------------------------


def hardware_setting(image_el: ET.Element) -> ET.Element | None:
    return image_el.find('./Data/Image/Attachment[@Name="HardwareSetting"]')


@dataclass(frozen=True)
class LeicaSystem:
    """Which instrument recorded the image, as the HardwareSetting attachment names it.

    ``name`` is the system (``STELLARIS 5``, ``TCS SP8 X``), not the stand: the stand is
    the scan's ``MicroscopeModel`` (``DMI8-CS``). ``camera`` says the setting definition
    is a camera one, which is what a widefield system writes whatever it is called.
    """

    system_type: int | None
    name: str | None
    data_source: str | None
    software: str | None
    hardware_server_version: str | None
    camera: bool = False


def leica_system(image_el: ET.Element) -> LeicaSystem | None:
    hardware = hardware_setting(image_el)
    if hardware is None:
        return None
    code = attr_float(hardware, "SystemType")
    top = scan_definition(image_el)
    return LeicaSystem(
        system_type=int(code) if code is not None and code.is_integer() else None,
        name=attr_text(hardware, "SystemTypeName"),
        data_source=attr_text(hardware, "DataSourceTypeName"),
        software=attr_text(hardware, "Software"),
        hardware_server_version=attr_text(hardware, "HardwareServerVersion"),
        camera=top is not None and top.tag == "ATLCameraSettingDefinition",
    )


def _is_setting_definition(element: ET.Element) -> bool:
    # ATLConfocalSettingDefinition on a point scanner, ATLCameraSettingDefinition on a
    # widefield stand: the same role under two names.
    return element.tag.startswith("ATL") and element.tag.endswith("SettingDefinition")


def scan_definition(image_el: ET.Element) -> ET.Element | None:
    """The scan-wide setting: what was true of the whole acquisition."""
    hardware = hardware_setting(image_el)
    if hardware is None:
        return None
    return next((el for el in hardware if _is_setting_definition(el)), None)


def sequential_definitions(image_el: ET.Element) -> list[ET.Element]:
    """One setting per sequential step, in the order `SequentialSettingIndex` counts."""
    hardware = hardware_setting(image_el)
    if hardware is None:
        return []
    listing = hardware.find("./LDM_Block_Sequential/LDM_Block_Sequential_List")
    if listing is None:
        return []
    return [el for el in listing if _is_setting_definition(el)]


@dataclass(frozen=True)
class LeicaScan:
    """The scan-wide state, from the top setting definition."""

    microscope_model: str | None
    system_serial_number: str | None
    objective_name: str | None
    objective_number: str | None
    magnification: float | None
    numerical_aperture: float | None
    immersion: str | None
    refraction_index: float | None
    pinhole_m: float | None
    pinhole_airy: float | None
    zoom: float | None
    scan_speed: float | None
    scan_direction: str | None
    scan_mode: str | None
    pixel_dwell_time_s: float | None
    line_average: float | None
    frame_average: float | None
    line_accumulation: float | None
    frame_accumulation: float | None
    bit_size: float | None
    rotator_angle: float | None
    z_mode: str | None
    stage_x_m: float | None
    stage_y_m: float | None
    flip_x: bool | None
    flip_y: bool | None
    swap_xy: bool | None
    is_inverted: bool | None = None
    can_do_sted: bool | None = None
    sted_active: bool | None = None
    resonant: bool | None = None
    base_zoom: float | None = None

    @property
    def immersion_kind(self) -> ObjectiveImmersion | None:
        if self.immersion:
            return LEICA_IMMERSION.get(self.immersion.upper())
        return None


def leica_scan(image_el: ET.Element) -> LeicaScan | None:
    top = scan_definition(image_el)
    if top is None:
        return None
    return LeicaScan(
        microscope_model=attr_text(top, "MicroscopeModel"),
        system_serial_number=attr_text(top, "SystemSerialNumber"),
        objective_name=attr_text(top, "ObjectiveName"),
        objective_number=attr_text(top, "ObjectiveNumber"),
        magnification=attr_float(top, "Magnification"),
        numerical_aperture=attr_float(top, "NumericalAperture"),
        immersion=attr_text(top, "Immersion"),
        refraction_index=attr_float(top, "RefractionIndex"),
        pinhole_m=attr_float(top, "Pinhole"),
        pinhole_airy=attr_float(top, "PinholeAiry"),
        zoom=attr_float(top, "Zoom"),
        scan_speed=attr_float(top, "ScanSpeed"),
        scan_direction=attr_text(top, "ScanDirectionXName"),
        scan_mode=attr_text(top, "ScanMode"),
        pixel_dwell_time_s=attr_float(top, "PixelDwellTime"),
        line_average=attr_float(top, "LineAverage"),
        frame_average=attr_float(top, "FrameAverage"),
        # Sic: LAS X writes the line one with an underscore.
        line_accumulation=attr_float(top, "Line_Accumulation"),
        frame_accumulation=attr_float(top, "FrameAccumulation"),
        bit_size=attr_float(top, "BitSize"),
        rotator_angle=attr_float(top, "RotatorAngle"),
        z_mode=attr_text(top, "ZUseModeName"),
        stage_x_m=attr_float(top, "StagePosX"),
        stage_y_m=attr_float(top, "StagePosY"),
        flip_x=attr_flag(top, "FlipX"),
        flip_y=attr_flag(top, "FlipY"),
        swap_xy=attr_flag(top, "SwapXY"),
        is_inverted=attr_flag(top, "IsInverseMicroscopeModel"),
        can_do_sted=attr_flag(top, "CanDoSTED"),
        sted_active=attr_flag(top, "IsSTEDActive"),
        resonant=attr_flag(top, "IsResonantScanner"),
        base_zoom=attr_float(top, "BaseZoom"),
    )


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaserLine:
    wavelength_nm: float
    intensity_percent: float
    source: str | None
    # The AOBS setting for this line, where the AOBS is set for it at all. LAS X writes
    # -1 for a line the AOBS does not handle — on a STELLARIS, the 405 diode, which fires
    # with -1 while the white-light lines beside it carry 100. Those are the AOBS notches.
    aobs_intensity: float | None = None
    # The laser behind the line, from `LaserArray` ("WLL", "Diode 405").
    laser_name: str | None = None


@dataclass(frozen=True)
class StedLine:
    wavelength_nm: float
    pulsed: bool | None


@dataclass(frozen=True)
class LeicaDetector:
    name: str
    type: str | None
    gain: float | None
    offset: float | None
    mode: str | None
    dye: str | None
    band_nm: tuple[float, float] | None
    # Where the detector sits: "Internal" behind the pinhole (descanned), "TLD" behind
    # the sample (transmitted), "RLD"/"External" beside the objective (non-descanned).
    scan_type: str | None = None
    channel_number: int | None = None
    vendor_serial: str | None = None
    is_sted_detector: bool | None = None

    @property
    def placement(self) -> str:
        """``descanned``, ``transmitted`` or ``non-descanned``."""
        kind = (self.scan_type or "").strip().lower()
        if kind == "tld":
            return "transmitted"
        if kind in ("rld", "external", "ndd", "nondescanned", "non-descanned"):
            return "non-descanned"
        return "descanned"


@dataclass(frozen=True)
class DisplayWindow:
    """A LAS X display window, as fractions of the channel's full range."""

    black: float
    white: float
    gamma: float | None

    @property
    def is_default(self) -> bool:
        # 0..1 at gamma 1 is what a window says when nobody set it.
        return self.black == 0.0 and self.white == 1.0 and self.gamma in (None, 1.0)


@dataclass(frozen=True)
class LeicaChannel:
    """One channel, joined to the hardware that recorded it where the file says which."""

    index: int
    lut: str | None
    bits: int | None
    dye: str | None
    setting_name: str | None
    detector: LeicaDetector | None
    lines: tuple[LaserLine, ...]
    pinhole_m: float | None
    pinhole_airy: float | None
    window: DisplayWindow | None
    sted_lines: tuple[StedLine, ...] = ()
    # The main beam splitter where a filter wheel names one (TD 488/561/633, RSP 500):
    # what an SP5 or SP8 without an AOBS separates excitation from emission with.
    beam_splitter: str | None = None
    # A camera channel's own settings (leica_widefield.WidefieldChannel), or None.
    widefield: object | None = None

    @property
    def label(self) -> str | None:
        return self.dye or (self.detector.dye if self.detector else None)


def _strip_vendor(dye: str | None) -> str | None:
    # Leica's dye database names are "Leica/ALEXA 488"; the prefix is a namespace.
    return dye.removeprefix("Leica/") if dye else None


def _properties(channel_el: ET.Element) -> dict[str, str]:
    return {
        key: value
        for prop in channel_el.findall("./ChannelProperty")
        if (key := prop.findtext("Key")) and (value := prop.findtext("Value"))
    }


def _window(
    element: ET.Element | None, black: str, white: str, gamma: str
) -> DisplayWindow | None:
    if element is None:
        return None
    low, high = attr_float(element, black), attr_float(element, white)
    if low is None or high is None or high <= low:
        return None
    return DisplayWindow(low, high, attr_float(element, gamma))


def _active_detectors(setting_el: ET.Element) -> list[ET.Element]:
    return [
        detector
        for detector in setting_el.findall("./DetectorList/Detector")
        if detector.get("IsActive") == "1"
    ]


def _detector(setting_el: ET.Element, detector_el: ET.Element) -> LeicaDetector:
    band = None
    # The spectral band is not on the detector but on the Spectro block, keyed by the
    # detector's `Channel` number — DetectionRangeBegin/End are something else (320/1
    # on every HyD checked).
    for multiband in setting_el.findall("./Spectro/MultiBand"):
        if multiband.get("Channel") == detector_el.get("Channel"):
            left = attr_float(multiband, "LeftWorld")
            right = attr_float(multiband, "RightWorld")
            if left is not None and right is not None and right > left:
                band = (left, right)
            break
    channel = attr_float(detector_el, "Channel")
    return LeicaDetector(
        name=detector_el.get("Name", ""),
        type=attr_text(detector_el, "Type"),
        gain=attr_float(detector_el, "Gain"),
        offset=attr_float(detector_el, "Offset"),
        mode=attr_text(detector_el, "AcquisitionModeName"),
        dye=_strip_vendor(attr_text(detector_el, "DyeName")),
        band_nm=band,
        scan_type=attr_text(detector_el, "ScanType"),
        channel_number=int(channel) if channel is not None and channel.is_integer() else None,
        vendor_serial=attr_text(detector_el, "VendorSerial"),
        is_sted_detector=attr_flag(detector_el, "IsSTEDDetector"),
    )


def _laser_name(setting_el: ET.Element, source: str | None, wavelength: float) -> str | None:
    # A source with one laser is that laser whatever its wavelength (the WLL says 0); a
    # source with several is the one at this line, or nobody.
    lasers = [
        laser
        for laser in setting_el.findall("./LaserArray/Laser")
        if source is not None and attr_text(laser, "LightSourceName") == source
    ]
    if len(lasers) == 1:
        return attr_text(lasers[0], "LaserName")
    for laser in lasers:
        if attr_float(laser, "Wavelength") == wavelength:
            return attr_text(laser, "LaserName")
    return None


def _lines(setting_el: ET.Element) -> tuple[LaserLine, ...]:
    found = []
    for aotf in setting_el.findall("./AotfList/Aotf"):
        source = attr_text(aotf, "LightSourceName")
        for line in aotf.findall("./LaserLineSetting"):
            wavelength = attr_float(line, "LaserLine")
            intensity = attr_float(line, "IntensityDev")
            if wavelength and intensity is not None and intensity > LINE_OFF_PERCENT:
                aobs = attr_float(line, "AOBSIntensityDev")
                found.append(
                    LaserLine(
                        wavelength,
                        intensity,
                        source,
                        aobs_intensity=aobs if aobs is not None and aobs >= 0 else None,
                        laser_name=_laser_name(setting_el, source, wavelength),
                    )
                )
    return tuple(found)


def _sted_lines(setting_el: ET.Element) -> tuple[StedLine, ...]:
    # Only on a setting that has STED on: a STELLARIS writes an empty depletion line (0)
    # into every setting whether it can deplete or not.
    if not attr_flag(setting_el, "IsSTEDActive"):
        return ()
    return tuple(
        StedLine(wavelength, attr_flag(line, "IsPulsedLaser"))
        for line in setting_el.findall("./STED_DepletionLine")
        if (wavelength := attr_float(line, "Wavelength"))
    )


# A wheel position is a beam splitter when the wheel or the filter says so. LAS AF/X
# name the main one "TD 488/561/633", "DD 488/561", "RSP 500", or the wheel
# "Beam Splitter"/"Excitation Beam Splitter".
BEAM_SPLITTER_WORDS = ("beam splitter", "beamsplitter", "dichroic")
# Not "SP": a short-pass filter is spelled that way too.
BEAM_SPLITTER_PREFIXES = ("TD ", "DD ", "RSP ")


def _beam_splitter(setting_el: ET.Element) -> str | None:
    for wheel in setting_el.findall("./FilterWheel/Wheel"):
        wheel_name = (attr_text(wheel, "FilterWheelName") or "").lower()
        filter_name = attr_text(wheel, "FilterName")
        if filter_name is None:
            continue
        if any(word in wheel_name for word in BEAM_SPLITTER_WORDS) or filter_name.startswith(
            BEAM_SPLITTER_PREFIXES
        ):
            return filter_name
    return None


def leica_channels(image_el: ET.Element) -> list[LeicaChannel]:
    """One `LeicaChannel` per ``ChannelDescription``, in file order.

    The link to the hardware is the channel's own ``SequentialSettingIndex`` (which
    sequential setting recorded it — the scan-wide one when there are none) and
    ``DetectorName`` (which of that setting's active detectors). It is trusted only as
    far as it checks out: the detector has to exist, be active, and agree on the dye
    where both name one. A channel whose link fails carries no detector and no laser
    lines — a dataset described as recorded by the wrong detector is worse than one
    that does not say.

    Files without ``ChannelProperty`` links (older LAS AF) fall back to Leica's own
    convention, the i-th active detector across the sequential settings in order, and
    only when that count matches the channels exactly.
    """
    channel_els = image_el.findall(
        "./Data/Image/ImageDescription/Channels/ChannelDescription"
    )
    scan = scan_definition(image_el)
    steps = sequential_definitions(image_el) or ([scan] if scan is not None else [])

    scaling = image_el.findall(
        './Data/Image/Attachment[@Name="ViewerScaling"]/ChannelScalingInfo'
    )
    # Index-aligned only when there is one per channel: a single-channel image with four
    # scaling entries does not say which of them is its own.
    viewer = scaling if len(scaling) == len(channel_els) else []

    ordered = [(step, d) for step in steps for d in _active_detectors(step)]

    channels = []
    for index, channel_el in enumerate(channel_els):
        props = _properties(channel_el)
        dye = _strip_vendor(props.get("DyeName"))
        step = detector_el = None

        if "DetectorName" in props:
            try:
                step = steps[int(props.get("SequentialSettingIndex", "0"))]
            except (ValueError, IndexError):
                step = None
            if step is not None:
                detector_el = next(
                    (
                        d
                        for d in _active_detectors(step)
                        if d.get("Name") == props["DetectorName"]
                    ),
                    None,
                )
        elif len(ordered) == len(channel_els):
            step, detector_el = ordered[index]

        detector = _detector(step, detector_el) if detector_el is not None else None
        if detector is not None and dye and detector.dye and detector.dye != dye:
            logger.info(
                "Channel %s names dye %r but detector %s recorded %r; not attaching "
                "the detector.",
                index,
                dye,
                detector.name,
                detector.dye,
            )
            detector = None
        if detector is None:
            step = None

        window = _window(
            viewer[index] if viewer else None, "BlackValue", "WhiteValue", "GammaValue"
        )
        if window is None or window.is_default:
            window = _window(
                detector_el.find("./LutInfo") if detector is not None else None,
                "LutBlackValue",
                "LutWhiteValue",
                "LutGammaValue",
            )
        if window is not None and window.is_default:
            window = None

        resolution = attr_text(channel_el, "Resolution")
        channels.append(
            LeicaChannel(
                index=index,
                lut=attr_text(channel_el, "LUTName"),
                bits=int(resolution) if resolution and resolution.isdigit() else None,
                dye=dye,
                setting_name=attr_text(step, "UserSettingName"),
                detector=detector,
                lines=_lines(step) if step is not None else (),
                pinhole_m=attr_float(step, "Pinhole"),
                pinhole_airy=attr_float(step, "PinholeAiry"),
                window=window,
                sted_lines=_sted_lines(step) if step is not None else (),
                beam_splitter=_beam_splitter(step) if step is not None else None,
            )
        )
    return channels


# ---------------------------------------------------------------------------
# Device state
# ---------------------------------------------------------------------------


def scan_devices(scan: LeicaScan | None) -> list:
    """The scan-wide devices: stand, objective and scanner."""
    if scan is None:
        return []
    return devices(
        device(
            "microscope",
            "microscope",
            setting("manufacturer", string="Leica Microsystems"),
            setting("model", string=scan.microscope_model),
            setting("serial-number", string=scan.system_serial_number),
        ),
        *optics_devices(scan),
    )


def optics_devices(scan: LeicaScan | None) -> list:
    """The objective and the scanner: what `scan_devices` says besides the stand."""
    if scan is None:
        return []
    return devices(
        device(
            "objective",
            "objective",
            setting("model", string=scan.objective_name),
            setting("order-number", string=scan.objective_number),
            setting("magnification", number=scan.magnification),
            setting("numerical-aperture", number=scan.numerical_aperture),
            setting("immersion", string=scan.immersion),
            setting("refractive-index", number=scan.refraction_index),
        ),
        device(
            "scanner",
            "scanner",
            setting(
                "type",
                string=None
                if scan.resonant is None
                else ("resonant" if scan.resonant else "galvo"),
            ),
            setting("scan-mode", string=scan.scan_mode),
            setting("zoom", number=scan.zoom),
            setting("scan-speed-hz", number=scan.scan_speed),
            setting("scan-direction", string=scan.scan_direction),
            setting(
                "pixel-dwell-time",
                quantity=f"{scan.pixel_dwell_time_s:g} s"
                if scan.pixel_dwell_time_s is not None
                else None,
            ),
            setting("line-average", number=scan.line_average),
            setting("frame-average", number=scan.frame_average),
            setting("line-accumulation", number=scan.line_accumulation),
            setting("frame-accumulation", number=scan.frame_accumulation),
            setting("bit-size", number=scan.bit_size),
            setting("rotator-angle", number=scan.rotator_angle),
            setting("z-mode", string=scan.z_mode),
            setting("pinhole", quantity=um(scan.pinhole_m)),
            setting("pinhole-airy", number=scan.pinhole_airy),
        ),
    )


def channel_optikit(channel: LeicaChannel) -> OptikitStateInput | None:
    """The hardware state of one channel: its detector, its lines and its pinhole."""
    detector = channel.detector
    index = channel.index
    state_devices = devices(
        device(
            f"channel-{index}",
            "channel",
            setting("lut", string=channel.lut),
            setting("bits", number=float(channel.bits) if channel.bits else None),
            setting("dye", string=channel.dye),
            setting("sequential-setting", string=channel.setting_name),
            setting(
                "display-black",
                number=channel.window.black if channel.window else None,
            ),
            setting(
                "display-white",
                number=channel.window.white if channel.window else None,
            ),
            setting(
                "display-gamma",
                number=channel.window.gamma if channel.window else None,
            ),
        ),
        device(
            f"detector-{index}",
            "detector",
            setting("name", string=detector.name if detector else None),
            setting("detector-type", string=detector.type if detector else None),
            setting("gain", number=detector.gain if detector else None),
            setting("offset", number=detector.offset if detector else None),
            setting("acquisition-mode", string=detector.mode if detector else None),
            setting("scan-type", string=detector.scan_type if detector else None),
            setting("serial-number", string=detector.vendor_serial if detector else None),
            setting(
                "detection-begin",
                quantity=nm(detector.band_nm[0])
                if detector and detector.band_nm
                else None,
            ),
            setting(
                "detection-end",
                quantity=nm(detector.band_nm[1])
                if detector and detector.band_nm
                else None,
            ),
        ),
        *(
            device(
                f"laser-{index}-{line_index}",
                "light-source",
                setting("source", string=line.source),
                setting("laser", string=line.laser_name),
                setting("wavelength", quantity=nm(line.wavelength_nm)),
                setting("aotf-intensity-percent", number=line.intensity_percent),
                setting("aobs-intensity-percent", number=line.aobs_intensity),
            )
            for line_index, line in enumerate(channel.lines)
        ),
        *(
            device(
                f"sted-{index}-{line_index}",
                "light-source",
                setting("role", string="depletion"),
                setting("wavelength", quantity=nm(line.wavelength_nm)),
                setting("pulsed", flag=line.pulsed),
            )
            for line_index, line in enumerate(channel.sted_lines)
        ),
        device(
            f"beam-splitter-{index}",
            "beam-splitter",
            setting("model", string=channel.beam_splitter),
        ),
        device(
            f"pinhole-{index}",
            "pinhole",
            setting("diameter", quantity=um(channel.pinhole_m)),
            setting("airy-units", number=channel.pinhole_airy),
        ),
    )
    if not state_devices:
        return None
    return OptikitStateInput(devices=state_devices)


# ---------------------------------------------------------------------------
# The light path
# ---------------------------------------------------------------------------


def channel_lightpath(
    channel: LeicaChannel,
    scan: LeicaScan | None,
    sample_label: str,
) -> LightpathGraphInput | None:
    """The light path for one channel, as far as LAS X recorded it.

    Lines → (combiner) → objective → sample → objective → pinhole → detector, each hop
    only where the file names the element. Built from one ``hops`` list, as in the OME
    converter, so that an edge cannot name a port its element does not declare.
    """
    detector = channel.detector
    band = (
        SpectrumInput(min=nm(detector.band_nm[0]), max=nm(detector.band_nm[1]))
        if detector and detector.band_nm
        else None
    )
    excitation_beam = BeamStateInput(
        wavelength=nm(channel.lines[0].wavelength_nm)
        if len(channel.lines) == 1
        else None,
    )
    emission_beam = BeamStateInput(
        wavelength=nm(sum(detector.band_nm) / 2) if detector and detector.band_nm else None,
        mode_hint=detector.mode if detector else None,
    )
    immersion = scan.immersion_kind if scan else None
    medium = immersion.value if immersion is not None else "AIR"

    elements: list[OpticalElementInput] = []
    hops: list[tuple[str, str, str, str, BeamStateInput | None, str]] = []
    tail: tuple[str, str] | None = None

    for index, line in enumerate(channel.lines):
        elements.append(
            LaserElementInput(
                id=f"line-{index}",
                label=f"{line.source or 'Laser'} {line.wavelength_nm:g} nm",
                nominal_wavelength=nm(line.wavelength_nm),
                channel=ChannelKind.FREE_SPACE,
                ports=[_port("out", "output", PortRole.OUTPUT)],
            )
        )
    if len(channel.lines) == 1:
        tail = ("line-0", "out")
    elif len(channel.lines) > 1:
        ports = [
            _port(f"in-{i}", f"line {i}", PortRole.INPUT)
            for i in range(len(channel.lines))
        ]
        ports.append(_port("out", "combined", PortRole.OUTPUT))
        # The AOTF is what actually merges the lines on a Leica point scanner.
        elements.append(BeamSplitterElementInput(id="aotf", label="AOTF", ports=ports))
        for i in range(len(channel.lines)):
            hops.append((f"line-{i}", "out", "aotf", f"in-{i}", None, "AIR"))
        tail = ("aotf", "out")

    has_objective = scan is not None and (
        scan.objective_name or scan.numerical_aperture or scan.magnification
    )
    if has_objective:
        elements.append(
            ObjectiveElementInput(
                id="objective",
                label=scan.objective_name or "Objective",
                brand="Leica",
                model=scan.objective_name,
                serial_number=scan.objective_number,
                magnification=scan.magnification,
                numerical_aperture=scan.numerical_aperture,
                immersion_medium=immersion,
                ports=[
                    _port("illum-in", "excitation in", PortRole.INPUT),
                    _port("illum-out", "excitation to sample", PortRole.OUTPUT),
                    _port("det-in", "fluorescence in", PortRole.INPUT),
                    _port("det-out", "fluorescence out", PortRole.OUTPUT),
                ],
            )
        )
        if tail is not None:
            hops.append((*tail, "objective", "illum-in", excitation_beam, "AIR"))
        tail = ("objective", "illum-out")

    elements.append(
        SampleElementInput(
            id="sample",
            label=sample_label,
            description=f"stained for {channel.label}" if channel.label else None,
            ports=[
                _port("illum-in", "illuminated volume", PortRole.INPUT),
                _port("em-out", "fluorescence", PortRole.OUTPUT, band),
            ],
        )
    )
    if tail is not None:
        hops.append((*tail, "sample", "illum-in", excitation_beam, medium))
    tail = ("sample", "em-out")

    if has_objective:
        hops.append((*tail, "objective", "det-in", emission_beam, medium))
        tail = ("objective", "det-out")

    if channel.pinhole_m is not None:
        elements.append(
            PinholeElementInput(
                id="pinhole",
                label="Confocal pinhole"
                + (
                    f" ({channel.pinhole_airy:.2f} AU)"
                    if channel.pinhole_airy is not None
                    else ""
                ),
                diameter=um(channel.pinhole_m),
                ports=_in_out(),
            )
        )
        hops.append((*tail, "pinhole", "in", emission_beam, "AIR"))
        tail = ("pinhole", "out")

    if detector is not None:
        elements.append(
            DetectorElementInput(
                id="detector",
                label=detector.name or "Detector",
                manufacturer="Leica",
                model=detector.type,
                gain=detector.gain,
                ports=[_port("in", "sensor", PortRole.INPUT, band)],
            )
        )
        hops.append((*tail, "detector", "in", emission_beam, "AIR"))

    if len(elements) < 2 or not hops:
        return None

    edges = [
        LightEdgeInput(
            id=f"{src}:{src_port}->{dst}:{dst_port}",
            source_element_id=src,
            source_port_id=src_port,
            target_element_id=dst,
            target_port_id=dst_port,
            beam=beam,
            medium=medium_,
        )
        for src, src_port, dst, dst_port, beam, medium_ in hops
    ]
    return LightpathGraphInput(elements=elements, edges=edges)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def own_element(image_el: ET.Element) -> ET.Element:
    """The image's element without its ``Children``: what is true of this image alone.

    A folder element nests every image under it, so serialising one whole would copy its
    descendants' metadata into it — and a LIF's root folder holds the entire file.
    """
    copy = ET.Element(image_el.tag, image_el.attrib)
    copy.extend(child for child in image_el if child.tag != "Children")
    return copy


def image_record(
    file_name: str,
    file_id: str,
    path: str,
    unique_id: str | None,
    tile: int | None,
    image_el: ET.Element,
) -> str:
    """The JSON the origin anchor's metadata blob takes, holding Leica's own XML.

    ``OmeMetadataInput`` is the only document slot an anchor has, and it takes any JSON
    string. What goes in it here is not OME: it says so under ``format``, and carries
    this image's XML element verbatim, so nothing LAS X recorded is lost to a schema
    this converter happens to model.
    """
    return json.dumps(
        {
            "format": "leica-lif",
            "source_file": file_name,
            "source_file_id": file_id,
            "image_path": path,
            "unique_id": unique_id,
            "tile_index": tile,
            "xml": ET.tostring(own_element(image_el), encoding="unicode"),
        }
    )
