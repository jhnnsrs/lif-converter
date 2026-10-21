"""Leica metadata from LAS AF 2.x records: what an older SP5 file holds instead of an
``ATL…SettingDefinition``.

An SP5 file written by LAS AF keeps its hardware as flat records inside the
``HardwareSetting`` attachment rather than as one setting element with attributes:

* ``ScannerSettingRecord`` — ``Identifier`` / ``Variant`` pairs for the scan:
  ``SystemType`` (the system's name, ``TCS SP5``), ``dblPinhole`` (metres), ``dblZoom``,
  ``dblStepSize``, ``nDelayTime_s``, ``CameraName``.
* ``FilterSettingRecord`` — ``ObjectName`` / ``ClassName`` / ``Attribute`` / ``Description``
  / ``Data`` / ``Variant`` for every optical part: the objective (``Objective``,
  ``NumericalAperture``, ``OrderNumber``, ``RefractionIndex``), each detection unit
  (``ClassName="CDetectionUnit"``, ``Attribute="State"``, ``Variant="Active"``), the
  spectrophotometer bands (``ClassName="CSpectrophotometerUnit"``, ``Description`` ending in
  ``(left)`` / ``(right)``), the dyes (``Stain``) and the stage (``XPos`` / ``YPos``).

Those names are the ones Bio-Formats' ``LIFReader`` reads (checked against the string
constants of formats-gpl 8.5.0), not ones checked against a real SP5 file, which this
converter has never seen. So every reader here is tolerant: a record it does not
recognise is ignored, a value it cannot parse is absent, and a file with none of these
records yields ``None`` / nothing, which leaves the image with what `leica_metadata`
read on its own.
"""

from __future__ import annotations

import dataclasses
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass

from leica_metadata import (
    LINE_OFF_PERCENT,
    LaserLine,
    LeicaChannel,
    LeicaDetector,
    LeicaScan,
    attr_float,
    attr_text,
    scan_definition,
)

# Gain and offset are not read by Bio-Formats, so these names are a guess at the obvious
# ones; a record under any other name is left out rather than misread.
GAIN_ATTRIBUTES = ("HighVoltage", "Gain")
OFFSET_ATTRIBUTES = ("VideoOffset", "Offset")


@dataclass(frozen=True)
class FilterRecord:
    object_name: str | None
    class_name: str | None
    attribute: str | None
    description: str | None
    data: str | None
    variant: str | None

    @property
    def number(self) -> float | None:
        return _number(self.variant)


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _image(image_el: ET.Element) -> ET.Element | None:
    # Data/Image only: the image's Children are other images with records of their own.
    return image_el.find("./Data/Image")


def scanner_records(image_el: ET.Element) -> dict[str, str]:
    """``Identifier`` → ``Variant`` for every scanner record, first one winning."""
    image = _image(image_el)
    if image is None:
        return {}
    found: dict[str, str] = {}
    for record in image.iter("ScannerSettingRecord"):
        identifier, variant = attr_text(record, "Identifier"), attr_text(record, "Variant")
        if identifier is not None and variant is not None:
            found.setdefault(identifier, variant)
    return found


def filter_records(image_el: ET.Element) -> list[FilterRecord]:
    image = _image(image_el)
    if image is None:
        return []
    return [
        FilterRecord(
            object_name=attr_text(record, "ObjectName"),
            class_name=attr_text(record, "ClassName"),
            attribute=attr_text(record, "Attribute"),
            description=attr_text(record, "Description"),
            data=attr_text(record, "Data"),
            variant=attr_text(record, "Variant"),
        )
        for record in image.iter("FilterSettingRecord")
    ]


def has_legacy_records(image_el: ET.Element) -> bool:
    return bool(scanner_records(image_el) or filter_records(image_el))


def legacy_system_name(image_el: ET.Element) -> str | None:
    return scanner_records(image_el).get("SystemType")


def _first(records: Sequence[FilterRecord], attribute: str) -> FilterRecord | None:
    return next((r for r in records if r.attribute == attribute), None)


def _magnification(objective: str | None) -> float | None:
    # "HCX PL APO lambda blue 63.0x1.40 OIL UV", "HC PL APO 20x/0.70 IMM CORR"
    match = re.search(r"(\d+(?:\.\d+)?)\s*x", objective or "")
    return float(match.group(1)) if match else None


def _immersion(objective: str | None) -> str | None:
    tokens = re.split(r"[\s/]+", (objective or "").upper())
    return next((t for t in ("OIL", "GLYC", "W", "WATER", "IMM", "DRY") if t in tokens), None)


def legacy_scan(image_el: ET.Element) -> LeicaScan | None:
    """The scan-wide state from LAS AF records, or None when the file has none.

    Only for a file without a setting definition: where one exists it is the better
    record, and `leica_metadata.leica_scan` reads it.
    """
    if scan_definition(image_el) is not None:
        return None
    scanner = scanner_records(image_el)
    filters = filter_records(image_el)
    if not scanner and not filters:
        return None

    objective = _first(filters, "Objective")
    objective_name = objective.variant if objective else None
    aperture = _first(filters, "NumericalAperture")
    order = _first(filters, "OrderNumber")
    refraction = _first(filters, "RefractionIndex")
    x, y = _first(filters, "XPos"), _first(filters, "YPos")

    return LeicaScan(
        microscope_model=None,
        system_serial_number=None,
        objective_name=objective_name,
        objective_number=order.variant if order else None,
        magnification=_magnification(objective_name),
        numerical_aperture=aperture.number if aperture else None,
        immersion=_immersion(objective_name),
        refraction_index=refraction.number if refraction else None,
        pinhole_m=_number(scanner.get("dblPinhole")),
        pinhole_airy=None,
        zoom=_number(scanner.get("dblZoom")),
        scan_speed=None,
        scan_direction=None,
        scan_mode=None,
        pixel_dwell_time_s=None,
        line_average=None,
        frame_average=None,
        line_accumulation=None,
        frame_accumulation=None,
        bit_size=None,
        rotator_angle=None,
        z_mode=None,
        stage_x_m=x.number if x else None,
        stage_y_m=y.number if y else None,
        flip_x=None,
        flip_y=None,
        swap_xy=None,
    )


def _active_units(filters: Sequence[FilterRecord]) -> list[FilterRecord]:
    """The detection units switched on, in channel order (their ``Data``)."""
    active = [
        r
        for r in filters
        if r.class_name == "CDetectionUnit"
        and r.attribute == "State"
        and r.variant == "Active"
        and r.object_name
    ]

    def key(record: FilterRecord) -> float:
        number = _number(record.data)
        return number if number is not None else float("inf")

    return sorted(active, key=key)


def _unit_value(
    filters: Sequence[FilterRecord], unit: str, attributes: Sequence[str]
) -> float | None:
    for record in filters:
        if record.object_name == unit and record.attribute in attributes:
            return record.number
    return None


def _bands(filters: Sequence[FilterRecord]) -> list[tuple[str | None, tuple[float, float]]]:
    """Each spectrophotometer band as (the unit it names, (left, right)), in file order."""
    bands = []
    left: tuple[str | None, float] | None = None
    for record in filters:
        if record.class_name != "CSpectrophotometerUnit" or record.description is None:
            continue
        value = record.number
        if value is None or value <= 0:
            continue
        if record.description.endswith("(left)"):
            left = (record.object_name, value)
        elif record.description.endswith("(right)") and left is not None:
            if value > left[1]:
                bands.append((left[0], (left[1], value)))
            left = None
    return bands


def _legacy_lines(image_el: ET.Element) -> tuple[LaserLine, ...] | None:
    """The lines that fired, or None when they cannot be pinned to the whole image.

    A sequential acquisition fires different lines per setting, and LAS AF records do not
    say which channel each setting recorded — so with more than one setting, nobody's.
    """
    image = _image(image_el)
    if image is None or image.find(".//Sequential_Master") is not None:
        return None
    found = []
    for aotf in image.iter("Aotf"):
        source = attr_text(aotf, "LightSourceName")
        for line in aotf.iter("LaserLineSetting"):
            wavelength = attr_float(line, "LaserLine")
            intensity = attr_float(line, "IntensityDev")
            if wavelength and intensity is not None and intensity > LINE_OFF_PERCENT:
                found.append(LaserLine(wavelength, intensity, source))
    return tuple(found)


def legacy_channels(
    image_el: ET.Element, channels: Sequence[LeicaChannel]
) -> list[LeicaChannel]:
    """`channels` (as `leica_channels` read them) completed from LAS AF records.

    Detectors join channels by order, and only when the active units match the channels
    one to one — the same rule `leica_channels` applies to a file without links. A file
    with a setting definition is returned untouched.
    """
    if scan_definition(image_el) is not None or not channels:
        return list(channels)
    filters = filter_records(image_el)
    scanner = scanner_records(image_el)
    if not filters and not scanner:
        return list(channels)

    units = _active_units(filters)
    if len(units) != len(channels):
        units = []
    bands = _bands(filters)
    by_unit = {unit: band for unit, band in bands if unit}
    ordered_bands = [band for _, band in bands] if len(bands) == len(channels) else []
    stains = [r.variant for r in filters if r.attribute == "Stain" and r.variant]
    stains = stains if len(stains) == len(channels) else []
    lines = _legacy_lines(image_el) or ()
    pinhole = _number(scanner.get("dblPinhole"))

    completed = []
    for index, channel in enumerate(channels):
        detector = None
        if units:
            unit = units[index].object_name or ""
            band = by_unit.get(unit) or (ordered_bands[index] if ordered_bands else None)
            detector = LeicaDetector(
                name=unit,
                type=unit.split()[0] if unit.split() else None,
                gain=_unit_value(filters, unit, GAIN_ATTRIBUTES),
                offset=_unit_value(filters, unit, OFFSET_ATTRIBUTES),
                mode=None,
                dye=None,
                band_nm=band,
            )
        completed.append(
            dataclasses.replace(
                channel,
                dye=channel.dye or (stains[index] if stains else None),
                detector=channel.detector or detector,
                lines=channel.lines or (lines if detector is not None else ()),
                pinhole_m=channel.pinhole_m if channel.pinhole_m is not None else pinhole,
            )
        )
    return completed
