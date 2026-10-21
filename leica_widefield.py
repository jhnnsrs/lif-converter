"""Leica metadata for camera systems: THUNDER imagers and DMi8 / DM6 widefield stands.

A widefield acquisition writes an ``ATLCameraSettingDefinition`` where a point scanner
writes an ``ATLConfocalSettingDefinition``, so `leica_metadata` already reads its
objective and channel descriptions. What it does not read is the camera side:

* LAS AF (older DM6000 / DMI6000 files): ``ScannerSettingRecord`` identifiers
  ``CameraName`` and ``WFC <n> ExposureTime`` / ``Gain`` / ``WaveLength`` /
  ``UserDefName`` (sic, sometimes ``UesrDefName``), per channel ``n`` — the names
  Bio-Formats' ``LIFReader`` reads.
* LAS X: ``WideFieldChannelConfigurator/WideFieldChannelInfo`` per channel, carrying the
  filter cube (``FluoCubeName``), contrasting method, exposure and wavelengths, and the
  camera on the setting definition. Bio-Formats does not read these and no file of this
  kind was at hand, so every name below is looked up tolerantly and anything not found
  stays absent. Units are not assumed: an exposure is recorded as the text LAS X wrote.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from leica_metadata import attr_float, attr_text, scan_definition
from leica_legacy import scanner_records

WFC_RECORD = re.compile(r"^WFC\s*(\d+)\s*(\w+)$")


@dataclass(frozen=True)
class WidefieldChannel:
    cube: str | None = None
    contrasting_method: str | None = None
    exposure: str | None = None
    gain: float | None = None
    excitation_nm: float | None = None
    emission_nm: float | None = None
    # A wavelength LAS AF recorded without saying which side of the cube it is on.
    wavelength_nm: float | None = None
    name: str | None = None


@dataclass(frozen=True)
class WidefieldSetting:
    camera: str | None = None
    binning: str | None = None
    # Every attribute on the definition that names computational clearing (THUNDER's
    # ICC / LVCC), verbatim: what it is called varies, that it was on does not.
    clearing: tuple[tuple[str, str], ...] = ()


def _pick_text(element: ET.Element | None, *names: str) -> str | None:
    for name in names:
        value = attr_text(element, name)
        if value is not None:
            return value
    return None


def _wavelength(element: ET.Element | None, *names: str) -> float | None:
    # A wavelength that is not one (0 for "none", -1 for "unset") stays absent.
    for name in names:
        value = attr_float(element, name)
        if value is not None and 200 <= value <= 2000:
            return value
    return None


def camera_definition(image_el: ET.Element) -> ET.Element | None:
    top = scan_definition(image_el)
    return top if top is not None and top.tag == "ATLCameraSettingDefinition" else None


def widefield_setting(image_el: ET.Element) -> WidefieldSetting | None:
    top = camera_definition(image_el)
    records = scanner_records(image_el)
    if top is None and "CameraName" not in records:
        return None
    camera = _pick_text(top, "CameraName", "CameraModel")
    if camera is None and top is not None:
        camera = next(
            (
                value
                for el in top.iter()
                if (value := _pick_text(el, "CameraName", "CameraModel"))
            ),
            None,
        )
    clearing = (
        tuple(
            (key, value)
            for el in top.iter()
            for key, value in el.attrib.items()
            if "clearing" in key.lower() and value.strip()
        )
        if top is not None
        else ()
    )
    return WidefieldSetting(
        camera=camera or records.get("CameraName"),
        binning=_pick_text(top, "Binning", "BinningName"),
        clearing=clearing,
    )


def _modern_channels(top: ET.Element) -> list[WidefieldChannel]:
    return [
        WidefieldChannel(
            cube=_pick_text(info, "FluoCubeName", "FilterCubeName", "CubeName"),
            contrasting_method=_pick_text(info, "ContrastingMethodName", "ContrastingMethod"),
            exposure=_pick_text(info, "ExposureTime", "Exposure"),
            gain=attr_float(info, "Gain"),
            excitation_nm=_wavelength(info, "ExcitationWavelength", "ExcitationWaveLength"),
            emission_nm=_wavelength(info, "EmissionWavelength", "EmissionWaveLength"),
            name=_pick_text(info, "UserDefName", "ChannelName"),
        )
        for info in top.iter("WideFieldChannelInfo")
    ]


def _legacy_channels(image_el: ET.Element) -> list[WidefieldChannel]:
    grouped: dict[int, dict[str, str]] = {}
    for identifier, variant in scanner_records(image_el).items():
        match = WFC_RECORD.match(identifier)
        if match:
            grouped.setdefault(int(match.group(1)), {})[match.group(2)] = variant
    channels = []
    for _, values in sorted(grouped.items()):
        record = ET.Element("record", values)
        channels.append(
            WidefieldChannel(
                exposure=values.get("ExposureTime"),
                gain=attr_float(record, "Gain"),
                wavelength_nm=_wavelength(record, "WaveLength"),
                name=values.get("UserDefName") or values.get("UesrDefName"),
            )
        )
    return channels


def widefield_channels(image_el: ET.Element, count: int) -> list[WidefieldChannel | None]:
    """One entry per channel, index-aligned only when the file lists exactly `count`.

    Anything else would say which cube recorded which channel without knowing.
    """
    top = camera_definition(image_el)
    found = _modern_channels(top) if top is not None else []
    if not found:
        found = _legacy_channels(image_el)
    if len(found) != count:
        return [None] * count
    return list(found)
