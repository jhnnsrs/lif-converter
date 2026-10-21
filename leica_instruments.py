"""Leica instrument models: which microscope recorded an image, and how it is built.

`leica_metadata` reads what LAS X wrote; this module decides what kind of instrument
wrote it and lays the light path out the way that instrument is built. A STELLARIS,
an SP8 and an SP5 are all point scanners, but they are not the same machine: where the
lines are combined, what separates excitation from emission (an AOBS, or a TD/RSP
dichroic), what selects the detection band (the SP prism and slit), and what the
detectors are (PMT, HyD, HyD SMD, Power HyD S/X/R). A THUNDER or DMi8 is a camera on a
widefield stand and shares none of that.

The family is read from the ``HardwareSetting`` attachment's ``SystemTypeName``
(``STELLARIS 5`` on the files checked), from the legacy ``SystemType`` scanner record on
LAS AF files, and — for widefield, first — from the setting definition being a camera
one. Names are matched loosely (case, spaces and dashes ignored) because only
``STELLARIS 5`` has been seen in a real file. Anything not recognised is the generic
`LeicaInstrument`, whose light path is exactly `leica_metadata.channel_lightpath`.

Recorded and modelled are kept apart. An element the file names (a laser line, the AOBS
it set, the TD dichroic in its wheel, the detector, the pinhole) is recorded. An element
only the instrument's design implies (the scanner, the spectral selector, a coupling
the lines that bypass the AOBS must take) is labelled "(implied)", says which model
implied it where the element can carry a description, and never carries a number.

mikro has no instrument entity, so the identification rides on the ``microscope`` device
as settings: ``instrument-model`` (this module's key, e.g. ``leica.stellaris``),
``family``, ``variant``, ``system-type``, the system as ``model`` and the stand as
``stand``.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from mikro.api.schema import (
    BeamSplitterElementInput,
    BeamStateInput,
    CCDElementInput,
    ChannelKind,
    DetectorElementInput,
    FilterElementInput,
    FilterKind,
    LampElementInput,
    LaserElementInput,
    LightEdgeInput,
    LightpathGraphInput,
    LightPortInput,
    MirrorElementInput,
    ObjectiveElementInput,
    OpticalElementInput,
    OptikitStateInput,
    PinholeElementInput,
    PortRole,
    SampleElementInput,
    SpectrumInput,
)
from mikro.inputs.lightpath import in_out as _in_out
from mikro.inputs.lightpath import port as _port

from converter_metadata import device, devices, setting
from leica_legacy import has_legacy_records, legacy_channels, legacy_scan, legacy_system_name
from leica_metadata import (
    LaserLine,
    LeicaChannel,
    LeicaDetector,
    LeicaDimension,
    LeicaScan,
    LeicaSystem,
    channel_lightpath,
    channel_optikit,
    leica_channels,
    leica_scan,
    leica_system,
    nm,
    optics_devices,
    um,
)
from leica_widefield import (
    WidefieldChannel,
    WidefieldSetting,
    widefield_channels,
    widefield_setting,
)

logger = logging.getLogger(__name__)

Hop = tuple[str, str, str, str, BeamStateInput | None, str]


# ---------------------------------------------------------------------------
# Reading an image
# ---------------------------------------------------------------------------


def read_system(image_el: ET.Element) -> LeicaSystem | None:
    """The system, from the attachment or — on a LAS AF file — its scanner records."""
    system = leica_system(image_el)
    if system is not None and system.name is not None:
        return system
    legacy = legacy_system_name(image_el)
    if legacy is None:
        return system
    if system is None:
        return LeicaSystem(None, legacy, None, None, None)
    return dataclasses.replace(system, name=legacy)


def read_scan(image_el: ET.Element) -> LeicaScan | None:
    return leica_scan(image_el) or legacy_scan(image_el)


def read_channels(image_el: ET.Element) -> list[LeicaChannel]:
    """`leica_channels`, completed from LAS AF records and camera settings where present.

    The completion reads layouts no real file has confirmed (see `leica_legacy`), so if
    it fails the channels stay as `leica_channels` read them: that costs the image its
    detectors, not its conversion.
    """
    channels = leica_channels(image_el)
    try:
        completed = legacy_channels(image_el, channels)
        cameras = widefield_channels(image_el, len(completed))
        return [
            dataclasses.replace(channel, widefield=camera) if camera is not None else channel
            for channel, camera in zip(completed, cameras)
        ]
    except Exception:
        logger.exception("Could not complete the channels from LAS AF or camera records")
        return channels


def normalised(name: str | None) -> str:
    """``TCS SP8 X`` → ``tcssp8x``: what the family tokens are matched against."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


# ---------------------------------------------------------------------------
# Building a light path
# ---------------------------------------------------------------------------


@dataclass
class _Path:
    """Elements and the hops between them, built together so that no edge can name a
    port its element does not declare (nothing checks that before the server does)."""

    model: str
    elements: list[OpticalElementInput] = field(default_factory=list)
    hops: list[Hop] = field(default_factory=list)

    def add(self, element: OpticalElementInput) -> OpticalElementInput:
        self.elements.append(element)
        return element

    def hop(
        self,
        source: tuple[str, str] | None,
        target: tuple[str, str],
        beam: BeamStateInput | None,
        medium: str = "AIR",
    ) -> None:
        if source is not None:
            self.hops.append((*source, *target, beam, medium))

    def implied(self, label: str) -> str:
        return f"{label} (implied)"

    @property
    def implied_note(self) -> str:
        return f"implied by {self.model}"

    def graph(self) -> LightpathGraphInput | None:
        if len(self.elements) < 2 or not self.hops:
            return None
        return LightpathGraphInput(
            elements=self.elements,
            edges=[
                LightEdgeInput(
                    id=f"{src}:{src_port}->{dst}:{dst_port}",
                    source_element_id=src,
                    source_port_id=src_port,
                    target_element_id=dst,
                    target_port_id=dst_port,
                    beam=beam,
                    medium=medium,
                )
                for src, src_port, dst, dst_port, beam, medium in self.hops
            ],
        )


def _four_ports(band: SpectrumInput | None = None) -> list[LightPortInput]:
    """An element both beams cross: excitation one way, emission the other."""
    return [
        _port("ex-in", "excitation in", PortRole.INPUT),
        _port("ex-out", "excitation out", PortRole.OUTPUT),
        _port("det-in", "emission in", PortRole.INPUT),
        _port("det-out", "emission out", PortRole.OUTPUT, band),
    ]


def _objective(scan: LeicaScan | None) -> ObjectiveElementInput | None:
    if scan is None or not (scan.objective_name or scan.numerical_aperture or scan.magnification):
        return None
    return ObjectiveElementInput(
        id="objective",
        label=scan.objective_name or "Objective",
        brand="Leica",
        model=scan.objective_name,
        serial_number=scan.objective_number,
        magnification=scan.magnification,
        numerical_aperture=scan.numerical_aperture,
        immersion_medium=scan.immersion_kind,
        ports=[
            _port("illum-in", "excitation in", PortRole.INPUT),
            _port("illum-out", "excitation to sample", PortRole.OUTPUT),
            _port("det-in", "fluorescence in", PortRole.INPUT),
            _port("det-out", "fluorescence out", PortRole.OUTPUT),
        ],
    )


def _band(detector: LeicaDetector | None) -> SpectrumInput | None:
    if detector is None or detector.band_nm is None:
        return None
    return SpectrumInput(min=nm(detector.band_nm[0]), max=nm(detector.band_nm[1]))


def _medium(scan: LeicaScan | None) -> str:
    immersion = scan.immersion_kind if scan else None
    return immersion.value if immersion is not None else "AIR"


def _line_label(line: LaserLine) -> str:
    return f"{line.laser_name or line.source or 'Laser'} {line.wavelength_nm:g} nm"


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeicaInstrument:
    """An instrument model. This base is the generic one: today's confocal path."""

    key: str = "leica.confocal"
    family: str = "Leica"
    tokens: tuple[str, ...] = ()

    def matches(self, system: LeicaSystem | None) -> bool:
        name = normalised(system.name if system else None)
        return bool(name) and any(token in name for token in self.tokens)

    def variant(self, system: LeicaSystem | None) -> str | None:
        """What the system name says beyond the family: ``X`` of ``TCS SP8 X``."""
        if system is None or system.name is None:
            return None
        for token in self.tokens:
            pattern = r"[\s\-_]*".join(re.escape(c) for c in token)
            match = re.search(pattern, system.name, flags=re.IGNORECASE)
            if match:
                return system.name[match.end() :].strip(" -_") or None
        return None

    # -- devices --------------------------------------------------------------

    def identity(self, system: LeicaSystem | None, scan: LeicaScan | None):
        """The ``microscope`` device: what the instrument is, as the file names it.

        None for an image that records neither (a snapshot): naming a model for it
        would be the converter's guess, not the file's.
        """
        if system is None and scan is None:
            return None
        stand = scan.microscope_model if scan else None
        name = system.name if system else None
        return device(
            "microscope",
            "microscope",
            setting("manufacturer", string="Leica Microsystems"),
            setting("model", string=name or stand),
            setting("stand", string=stand if name else None),
            setting("serial-number", string=scan.system_serial_number if scan else None),
            setting("instrument-model", string=self.key),
            setting("family", string=self.family),
            setting("variant", string=self.variant(system)),
            setting(
                "system-type",
                number=float(system.system_type)
                if system and system.system_type is not None
                else None,
            ),
            setting("data-source", string=system.data_source if system else None),
            setting("inverted", flag=scan.is_inverted if scan else None),
            setting("sted-capable", flag=scan.can_do_sted if scan else None),
            setting("sted-active", flag=scan.sted_active if scan else None),
        )

    def devices(
        self,
        system: LeicaSystem | None,
        scan: LeicaScan | None,
        widefield: WidefieldSetting | None = None,
    ) -> list:
        return devices(self.identity(system, scan), *optics_devices(scan))

    def channel_devices(self, channel: LeicaChannel) -> OptikitStateInput | None:
        return channel_optikit(channel)

    # -- the light path ---------------------------------------------------------

    def channel_lightpath(
        self,
        channel: LeicaChannel,
        scan: LeicaScan | None,
        sample_label: str,
        widefield: WidefieldSetting | None = None,
        dims: Mapping[str, LeicaDimension] | None = None,
    ) -> LightpathGraphInput | None:
        return channel_lightpath(channel, scan, sample_label)


@dataclass(frozen=True)
class ConfocalInstrument(LeicaInstrument):
    """A Leica point scanner, laid out as one is built.

    Excitation: lines → AOTF → main beam splitter (AOBS or TD/RSP dichroic) → [the
    coupling of lines that bypass the AOBS] → scanner → objective → sample. Emission
    back the same way for a descanned detector: objective → scanner → coupling → main
    splitter → pinhole → spectral selector → detector. A non-descanned detector (RLD,
    the DIVE/MP NDDs) takes the light straight off the objective; a transmitted one
    (TLD) sits behind the sample.
    """

    spectral_label: str = "Spectral detection"
    # An instrument that has an AOBS whether or not the file set it (STELLARIS).
    always_aobs: bool = False

    def detector_model(self, detector: LeicaDetector) -> str | None:
        # "HyD S 1" → "HyD S", "PMT 2" → "PMT": the name without its position.
        base = re.sub(r"\s*\d+$", "", detector.name or "").strip()
        return base or detector.type

    def channel_lightpath(
        self,
        channel: LeicaChannel,
        scan: LeicaScan | None,
        sample_label: str,
        widefield: WidefieldSetting | None = None,
        dims: Mapping[str, LeicaDimension] | None = None,
    ) -> LightpathGraphInput | None:
        path = _Path(self.key)
        detector = channel.detector
        placement = detector.placement if detector else "descanned"
        band = _band(detector)
        medium = _medium(scan)
        excitation = BeamStateInput(
            wavelength=nm(channel.lines[0].wavelength_nm) if len(channel.lines) == 1 else None
        )
        emission = BeamStateInput(
            wavelength=nm(sum(detector.band_nm) / 2) if detector and detector.band_nm else None,
            mode_hint=detector.mode if detector else None,
        )
        if placement == "transmitted":
            emission = excitation

        # Sources. Every line that fired, and the depletion lines of a STED setting.
        for i, line in enumerate(channel.lines):
            path.add(
                LaserElementInput(
                    id=f"line-{i}",
                    label=_line_label(line),
                    model=line.laser_name,
                    nominal_wavelength=nm(line.wavelength_nm),
                    channel=ChannelKind.FREE_SPACE,
                    ports=[_port("out", "output", PortRole.OUTPUT)],
                )
            )
        for j, line in enumerate(channel.sted_lines):
            path.add(
                LaserElementInput(
                    id=f"sted-{j}",
                    label=f"STED depletion {line.wavelength_nm:g} nm",
                    nominal_wavelength=nm(line.wavelength_nm),
                    channel=ChannelKind.FREE_SPACE,
                    ports=[_port("out", "output", PortRole.OUTPUT)],
                )
            )

        # The main splitter, and which lines pass it.
        aobs_lines = [i for i, line in enumerate(channel.lines) if line.aobs_intensity is not None]
        splitter = self.main_splitter(path, channel, aobs_lines)
        # Only an AOBS the file set says which lines go around it: an implied one is
        # taken to carry them all.
        bypassing = [i for i in range(len(channel.lines)) if aobs_lines and i not in aobs_lines]
        bypass_lines = [channel.lines[i] for i in bypassing]
        through = [f"line-{i}" for i in (aobs_lines if aobs_lines else range(len(channel.lines)))]
        bypass = [f"line-{i}" for i in bypassing]
        sted = [f"sted-{j}" for j in range(len(channel.sted_lines))]
        if splitter is not None:
            bypass += sted
        else:
            through += sted

        tail = self.combine(path, through, excitation)
        if splitter is not None:
            path.add(splitter)
            path.hop(tail, (splitter.id, "ex-in"), excitation)
            tail = (splitter.id, "ex-out")

        coupling = None
        if bypass:
            coupling = path.add(
                FilterElementInput(
                    id="coupling",
                    label=path.implied(
                        "STED coupling"
                        if not bypass_lines
                        else "UV coupling"
                        if all(line.wavelength_nm < 420 for line in bypass_lines)
                        else "Laser coupling"
                    ),
                    description=f"{path.implied_note}: lines that do not pass the AOBS",
                    filter_kind=FilterKind.DICHROIC,
                    ports=[
                        *_four_ports(),
                        *(_port(f"in-{k}", source, PortRole.INPUT) for k, source in enumerate(bypass)),
                    ],
                )
            )
            for k, source in enumerate(bypass):
                path.hop((source, "out"), ("coupling", f"in-{k}"), None)
            path.hop(tail, ("coupling", "ex-in"), excitation)
            tail = ("coupling", "ex-out")

        scanner = None
        if scan is not None:
            scanner = path.add(
                MirrorElementInput(
                    id="scanner",
                    label=path.implied("Resonant scanner" if scan.resonant else "Galvo scanner"),
                    ports=_four_ports(),
                )
            )
            path.hop(tail, ("scanner", "ex-in"), excitation)
            tail = ("scanner", "ex-out")

        objective = _objective(scan)
        if objective is not None:
            path.add(objective)
            path.hop(tail, ("objective", "illum-in"), excitation)
            tail = ("objective", "illum-out")

        sample_ports = [_port("illum-in", "illuminated volume", PortRole.INPUT)]
        sample_ports.append(
            _port("trans-out", "transmitted light", PortRole.OUTPUT)
            if placement == "transmitted"
            else _port("em-out", "fluorescence", PortRole.OUTPUT, band)
        )
        path.add(
            SampleElementInput(
                id="sample",
                label=sample_label,
                description=f"stained for {channel.label}" if channel.label else None,
                ports=sample_ports,
            )
        )
        path.hop(tail, ("sample", "illum-in"), excitation, medium)

        if placement == "transmitted":
            tail = ("sample", "trans-out")
        else:
            tail = ("sample", "em-out")
            if objective is not None:
                path.hop(tail, ("objective", "det-in"), emission, medium)
                tail = ("objective", "det-out")
            if placement == "descanned":
                for element in (scanner, coupling, splitter):
                    if element is not None:
                        path.hop(tail, (element.id, "det-in"), emission)
                        tail = (element.id, "det-out")
                if channel.pinhole_m is not None:
                    path.add(
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
                    path.hop(tail, ("pinhole", "in"), emission)
                    tail = ("pinhole", "out")
                if detector is not None:
                    path.add(
                        FilterElementInput(
                            id="spectral",
                            label=path.implied(self.spectral_label),
                            description=path.implied_note,
                            filter_kind=FilterKind.TUNEABLE,
                            ports=_in_out(band),
                        )
                    )
                    path.hop(tail, ("spectral", "in"), emission)
                    tail = ("spectral", "out")

        if detector is not None:
            path.add(
                DetectorElementInput(
                    id="detector",
                    label=detector.name or "Detector",
                    manufacturer="Leica",
                    model=self.detector_model(detector),
                    serial_number=detector.vendor_serial,
                    gain=detector.gain,
                    ports=[_port("in", "sensor", PortRole.INPUT, band)],
                )
            )
            path.hop(tail, ("detector", "in"), emission)

        return path.graph()

    def combine(
        self, path: _Path, sources: Sequence[str], beam: BeamStateInput | None
    ) -> tuple[str, str] | None:
        """The lines into one beam: through the AOTF where there is more than one."""
        if not sources:
            return None
        if len(sources) == 1:
            return (sources[0], "out")
        ports = [_port(f"in-{i}", source, PortRole.INPUT) for i, source in enumerate(sources)]
        ports.append(_port("out", "combined", PortRole.OUTPUT))
        path.add(BeamSplitterElementInput(id="aotf", label="AOTF", ports=ports))
        for i, source in enumerate(sources):
            path.hop((source, "out"), ("aotf", f"in-{i}"), None)
        return ("aotf", "out")

    def main_splitter(
        self, path: _Path, channel: LeicaChannel, aobs_lines: Sequence[int]
    ) -> OpticalElementInput | None:
        """The AOBS the file set, else the dichroic its wheel names, else nothing —
        except on an instrument that is never without its AOBS.

        Not yet added to `path`: it goes in after the lines are combined in front of it.
        """
        if aobs_lines:
            notches = ", ".join(f"{channel.lines[i].wavelength_nm:g}" for i in aobs_lines)
            return (
                BeamSplitterElementInput(
                    id="aobs",
                    label=f"AOBS ({notches} nm)",
                    manufacturer="Leica",
                    model="AOBS",
                    ports=_four_ports(),
                )
            )
        if channel.beam_splitter:
            return (
                FilterElementInput(
                    id="beam-splitter",
                    label=channel.beam_splitter,
                    model=channel.beam_splitter,
                    filter_kind=FilterKind.DICHROIC,
                    ports=_four_ports(),
                )
            )
        if self.always_aobs:
            return (
                BeamSplitterElementInput(
                    id="aobs",
                    label=path.implied("AOBS"),
                    manufacturer="Leica",
                    model="AOBS",
                    ports=_four_ports(),
                )
            )
        return None


@dataclass(frozen=True)
class TcsSP5(ConfocalInstrument):
    key: str = "leica.tcs-sp5"
    family: str = "TCS SP5"
    tokens: tuple[str, ...] = ("sp5",)
    spectral_label: str = "Prism + slit (SP detector)"


@dataclass(frozen=True)
class TcsSP8(ConfocalInstrument):
    key: str = "leica.tcs-sp8"
    family: str = "TCS SP8"
    tokens: tuple[str, ...] = ("sp8",)
    spectral_label: str = "Prism + slit (SP detector)"


@dataclass(frozen=True)
class Stellaris(ConfocalInstrument):
    key: str = "leica.stellaris"
    family: str = "STELLARIS"
    tokens: tuple[str, ...] = ("stellaris",)
    always_aobs: bool = True

    def detector_model(self, detector: LeicaDetector) -> str | None:
        # A STELLARIS' HyDs are the Power HyD S / X / R (SiPM).
        base = super().detector_model(detector)
        if base and base.startswith("HyD"):
            return f"Power {base}"
        return base


@dataclass(frozen=True)
class Widefield(LeicaInstrument):
    """A camera system: THUNDER imagers, DMi8 / DM6 widefield stands.

    Light source → filter cube → objective → sample → objective → cube → camera. The
    cube is there only where the file names one: a brightfield channel has none, and
    the file not saying is no reason to draw one.
    """

    key: str = "leica.widefield"
    family: str = "Widefield"
    tokens: tuple[str, ...] = ("thunder", "dmi8", "dmi6000", "dm6", "dm4", "dm2500")

    def matches(self, system: LeicaSystem | None) -> bool:
        if system is None:
            return False
        source = (system.data_source or "").lower()
        return system.camera or "camera" in source or "wide" in source or super().matches(system)

    def devices(
        self,
        system: LeicaSystem | None,
        scan: LeicaScan | None,
        widefield: WidefieldSetting | None = None,
    ) -> list:
        return devices(
            self.identity(system, scan),
            *(d for d in optics_devices(scan) if d.label != "scanner"),
            device(
                "camera",
                "camera",
                setting("model", string=widefield.camera if widefield else None),
                setting("binning", string=widefield.binning if widefield else None),
                *(
                    setting(f"computational-clearing-{key}", string=value)
                    for key, value in (widefield.clearing if widefield else ())
                ),
            ),
        )

    def channel_devices(self, channel: LeicaChannel) -> OptikitStateInput | None:
        base = channel_optikit(channel)
        camera: WidefieldChannel | None = channel.widefield  # type: ignore[assignment]
        extra = device(
            f"camera-{channel.index}",
            "camera",
            setting("filter-cube", string=camera.cube if camera else None),
            setting("contrasting-method", string=camera.contrasting_method if camera else None),
            setting("exposure", string=camera.exposure if camera else None),
            setting("gain", number=camera.gain if camera else None),
            setting("excitation", quantity=nm(camera.excitation_nm) if camera else None),
            setting("emission", quantity=nm(camera.emission_nm) if camera else None),
            setting("wavelength", quantity=nm(camera.wavelength_nm) if camera else None),
            setting("name", string=camera.name if camera else None),
        )
        # Not the pinhole: a camera channel has none, whatever the definition carries.
        kept = [d for d in (base.devices if base else []) if not d.label.startswith("pinhole-")]
        found = devices(*kept, extra)
        return OptikitStateInput(devices=found) if found else None

    def channel_lightpath(
        self,
        channel: LeicaChannel,
        scan: LeicaScan | None,
        sample_label: str,
        widefield: WidefieldSetting | None = None,
        dims: Mapping[str, LeicaDimension] | None = None,
    ) -> LightpathGraphInput | None:
        path = _Path(self.key)
        camera: WidefieldChannel | None = channel.widefield  # type: ignore[assignment]
        medium = _medium(scan)
        excitation = BeamStateInput(
            wavelength=nm(camera.excitation_nm) if camera and camera.excitation_nm else None
        )
        emission = BeamStateInput(
            wavelength=nm(camera.emission_nm) if camera and camera.emission_nm else None
        )

        path.add(
            LampElementInput(
                id="source",
                label=path.implied("Light source"),
                channel=ChannelKind.FREE_SPACE,
                ports=[_port("out", "output", PortRole.OUTPUT)],
            )
        )
        tail: tuple[str, str] | None = ("source", "out")

        cube = None
        if camera is not None and camera.cube:
            cube = path.add(
                FilterElementInput(
                    id="cube",
                    label=camera.cube,
                    model=camera.cube,
                    filter_kind=FilterKind.DICHROIC,
                    ports=_four_ports(),
                )
            )
            path.hop(tail, ("cube", "ex-in"), excitation)
            tail = ("cube", "ex-out")

        objective = _objective(scan)
        if objective is not None:
            path.add(objective)
            path.hop(tail, ("objective", "illum-in"), excitation)
            tail = ("objective", "illum-out")

        path.add(
            SampleElementInput(
                id="sample",
                label=sample_label,
                description=f"stained for {channel.label}" if channel.label else None,
                ports=[
                    _port("illum-in", "illuminated field", PortRole.INPUT),
                    _port("em-out", "emitted light", PortRole.OUTPUT),
                ],
            )
        )
        path.hop(tail, ("sample", "illum-in"), excitation, medium)
        tail = ("sample", "em-out")
        if objective is not None:
            path.hop(tail, ("objective", "det-in"), emission, medium)
            tail = ("objective", "det-out")
        if cube is not None:
            path.hop(tail, ("cube", "det-in"), emission)
            tail = ("cube", "det-out")

        x = dims.get("X") if dims else None
        y = dims.get("Y") if dims else None
        name = widefield.camera if widefield else None
        path.add(
            CCDElementInput(
                id="camera",
                label=name or path.implied("Camera"),
                model=name,
                resolution=[x.count, y.count] if x is not None and y is not None else None,
                ports=[_port("in", "sensor", PortRole.INPUT)],
            )
        )
        path.hop(tail, ("camera", "in"), emission)
        return path.graph()


GENERIC = LeicaInstrument()
# Widefield first: a camera definition decides whatever the system is called.
INSTRUMENTS: tuple[LeicaInstrument, ...] = (Widefield(), Stellaris(), TcsSP8(), TcsSP5())


def identify(system: LeicaSystem | None, legacy: bool = False) -> LeicaInstrument:
    """The model for this system; the generic one when nothing matches.

    `legacy` says the image has LAS AF records and no setting definition — an SP5 era
    file — which makes it an SP5 when its name does not say otherwise.
    """
    for instrument in INSTRUMENTS:
        if instrument.matches(system):
            return instrument
    if legacy:
        return INSTRUMENTS[-1]
    return GENERIC


# ---------------------------------------------------------------------------
# One image, read once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeicaAcquisition:
    """Everything one image says about the instrument, and the model that reads it."""

    instrument: LeicaInstrument
    system: LeicaSystem | None
    scan: LeicaScan | None
    widefield: WidefieldSetting | None

    def devices(self) -> list:
        return self.instrument.devices(self.system, self.scan, self.widefield)

    def channel_devices(self, channel: LeicaChannel) -> OptikitStateInput | None:
        return self.instrument.channel_devices(channel)

    def channel_lightpath(
        self,
        channel: LeicaChannel,
        sample_label: str,
        dims: Mapping[str, LeicaDimension] | None = None,
    ) -> LightpathGraphInput | None:
        return self.instrument.channel_lightpath(
            channel, self.scan, sample_label, self.widefield, dims
        )


def read_acquisition(image_el: ET.Element) -> LeicaAcquisition:
    """The image's instrument, read once. The parts read from unconfirmed layouts (LAS AF
    records, camera settings) degrade to absent rather than fail: the modern scan alone
    is what the converter had before there were models."""
    system = leica_system(image_el)
    scan = leica_scan(image_el)
    legacy = False
    widefield = None
    try:
        system = read_system(image_el)
        if scan is None:
            scan = legacy_scan(image_el)
            legacy = scan is not None and has_legacy_records(image_el)
        widefield = widefield_setting(image_el)
    except Exception:
        logger.exception("Could not read the LAS AF or camera records")
    return LeicaAcquisition(
        instrument=identify(system, legacy),
        system=system,
        scan=scan,
        widefield=widefield,
    )
