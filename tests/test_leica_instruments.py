"""Offline checks of the instrument models, one fixture per Leica family.

Only the STELLARIS fixture is shaped like a real file: its attributes, the 405 diode
firing with ``AOBSIntensityDev="-1"`` beside WLL lines at 100, and the ``LaserArray`` are
copied from a STELLARIS 5 header. The SP8, SP5 and widefield fixtures follow what LAS X
and LAS AF are documented to write (Bio-Formats' reader, for the LAS AF records), and so
can only confirm that reading; the checks that matter for those are the ones at the end,
that a layout they do not expect degrades rather than fails.
"""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import app as A
import leica_instruments as I
import leica_metadata as L

# --- fixtures ------------------------------------------------------------------------

DIMS = (
    '<DimensionDescription DimID="1" NumberOfElements="512" Origin="0"'
    ' Length="1.0e-04" Unit="m" BytesInc="1" BitInc="0" />'
    '<DimensionDescription DimID="2" NumberOfElements="256" Origin="0"'
    ' Length="0.5e-04" Unit="m" BytesInc="512" BitInc="0" />'
)


def channel(dye: str | None = None, detector: str | None = None, step: int = 0) -> str:
    props = ""
    if detector is not None:
        props += f"<ChannelProperty><Key>DetectorName</Key><Value>{detector}</Value></ChannelProperty>"
        props += f"<ChannelProperty><Key>SequentialSettingIndex</Key><Value>{step}</Value></ChannelProperty>"
    if dye is not None:
        props += f"<ChannelProperty><Key>DyeName</Key><Value>Leica/{dye}</Value></ChannelProperty>"
    return f'<ChannelDescription Resolution="8" LUTName="Green">{props}</ChannelDescription>'


def image(
    channels: str,
    hardware: str,
    attrs: str = 'SystemTypeName="STELLARIS 5" SystemType="13" DataSourceTypeName="Confocal"'
    ' Software="LAS X 4.9.0.30221"',
) -> ET.Element:
    return ET.fromstring(
        f"""
<Element Name="img"><Data><Image>
  <ImageDescription><Channels>{channels}</Channels><Dimensions>{DIMS}</Dimensions></ImageDescription>
  <Attachment Name="HardwareSetting" {attrs}>{hardware}</Attachment>
</Image></Data></Element>
"""
    )


def confocal(
    lines: str,
    detectors: str,
    extra: str = "",
    top: str = "",
    wheels: str = "",
    sted: str = "",
) -> str:
    """A scan-wide definition and one sequential step, as LAS X nests them."""
    step = f"""
<ATLConfocalSettingDefinition UserSettingName="S1" Pinhole="1.0E-04" PinholeAiry="1" {extra}>
  <FilterWheel>{wheels}</FilterWheel>
  <Spectro>
    <MultiBand Channel="1" LeftWorld="410" RightWorld="475" />
    <MultiBand Channel="3" LeftWorld="500" RightWorld="560" />
    <MultiBand Channel="4" LeftWorld="650" RightWorld="750" />
  </Spectro>
  <AotfList>{lines}</AotfList>
  {sted}
  <DetectorList>{detectors}</DetectorList>
  <LaserArray>
    <Laser LaserName="Diode 405" LightSourceName="UV Light" Wavelength="405" />
    <Laser LaserName="WLL" LightSourceName="SuperContVisible Light" Wavelength="0" />
  </LaserArray>
</ATLConfocalSettingDefinition>
"""
    return f"""
<ATLConfocalSettingDefinition MicroscopeModel="DMI8-CS" IsInverseMicroscopeModel="1"
    SystemSerialNumber="8200000630" ObjectiveName="HC PL APO CS2 63x/1.30 GLYC"
    Magnification="63" NumericalAperture="1.3" Immersion="GLYC" Zoom="1"
    IsResonantScanner="0" CanDoSTED="0" {top} />
<LDM_Block_Sequential><LDM_Block_Sequential_List>{step}</LDM_Block_Sequential_List></LDM_Block_Sequential>
"""


def detector(name: str, number: int, scan_type: str = "Internal", type_: str = "SiPM") -> str:
    return (
        f'<Detector Name="{name}" Type="{type_}" ScanType="{scan_type}" Channel="{number}"'
        f' IsActive="1" Gain="2.5" Offset="0" VendorSerial="1L00{number}"'
        f' AcquisitionModeName="PhotonCounting" />'
    )


STELLARIS_LINES = (
    '<Aotf LightSourceName="UV Light">'
    '<LaserLineSetting LaserLine="405" IntensityDev="10" AOBSIntensityDev="-1" /></Aotf>'
    '<Aotf LightSourceName="SuperContVisible Light">'
    '<LaserLineSetting LaserLine="485" IntensityDev="10" AOBSIntensityDev="100" />'
    '<LaserLineSetting LaserLine="653" IntensityDev="30" AOBSIntensityDev="100" />'
    '<LaserLineSetting LaserLine="590" IntensityDev="0" AOBSIntensityDev="-1" /></Aotf>'
)


@pytest.fixture
def stellaris():
    return image(
        channel("ALEXA 405", "HyD S 1"),
        confocal(STELLARIS_LINES, detector("HyD S 1", 1) + detector("HyD X 2", 2)),
    )


def chain(graph) -> list[str]:
    return [element.id for element in graph.elements]


def assert_ports_hold(graph) -> None:
    """Every edge leaves by an output its element declares and enters by an input, and
    no id is used twice: nothing else checks the graph before the server does."""
    ids = [e.id for e in graph.elements]
    assert len(ids) == len(set(ids)), ids
    roles = {(e.id, p.id): p.role for e in graph.elements for p in e.ports}
    for edge in graph.edges:
        assert roles.get((edge.source_element_id, edge.source_port_id)) == I.PortRole.OUTPUT, edge.id
        assert roles.get((edge.target_element_id, edge.target_port_id)) == I.PortRole.INPUT, edge.id
    edge_ids = [edge.id for edge in graph.edges]
    assert len(edge_ids) == len(set(edge_ids)), edge_ids


def path_of(element: ET.Element, index: int = 0):
    acquisition = I.read_acquisition(element)
    channels = I.read_channels(element)
    graph = acquisition.channel_lightpath(channels[index], "sample", L.dimensions(element))
    if graph is not None:
        assert_ports_hold(graph)
    return acquisition, channels, graph


def by_id(graph, element_id):
    return next(e for e in graph.elements if e.id == element_id)


# --- identification -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, key, variant",
    [
        ("STELLARIS 5", "leica.stellaris", "5"),
        ("Stellaris 8 STED", "leica.stellaris", "8 STED"),
        ("TCS SP8 X", "leica.tcs-sp8", "X"),
        ("TCS-SP8", "leica.tcs-sp8", None),
        ("TCS SP8 STED 3X", "leica.tcs-sp8", "STED 3X"),
        ("tcs sp5 II", "leica.tcs-sp5", "II"),
        ("THUNDER Imager Live Cell", "leica.widefield", "Imager Live Cell"),
        ("DMi8", "leica.widefield", None),
        ("TCS SPE", "leica.confocal", None),
        (None, "leica.confocal", None),
    ],
)
def test_the_family_is_read_from_the_system_name(name, key, variant):
    system = L.LeicaSystem(None, name, "Confocal", None, None)
    instrument = I.identify(system)
    assert instrument.key == key
    assert instrument.variant(system) == variant


def test_a_camera_definition_is_widefield_whatever_its_name():
    system = L.LeicaSystem(None, "STELLARIS 5", "Confocal", None, None, camera=True)
    assert I.identify(system).key == "leica.widefield"


def test_the_system_is_the_model_and_the_stand_is_the_stand(stellaris):
    acquisition = I.read_acquisition(stellaris)
    assert acquisition.instrument.key == "leica.stellaris"
    [microscope, *_] = acquisition.devices()
    values = {s.name: s.text or s.number or s.flag for s in microscope.settings}
    assert values["model"] == "STELLARIS 5"
    assert values["stand"] == "DMI8-CS"
    assert values["instrument-model"] == "leica.stellaris"
    assert values["system-type"] == 13.0
    assert values["inverted"] is True


# --- STELLARIS -----------------------------------------------------------------------


def test_stellaris_lines_through_the_aobs_and_the_uv_diode_around_it(stellaris):
    _, [only], graph = path_of(stellaris)
    assert [line.aobs_intensity for line in only.lines] == [None, 100.0, 100.0]
    assert [line.laser_name for line in only.lines] == ["Diode 405", "WLL", "WLL"]
    assert chain(graph) == [
        "line-0", "line-1", "line-2", "aotf", "aobs", "coupling", "scanner",
        "objective", "sample", "pinhole", "spectral", "detector",
    ]
    # The AOBS notches are the lines it was set for, not every line that fired.
    assert by_id(graph, "aobs").label == "AOBS (485, 653 nm)"
    edges = {edge.id for edge in graph.edges}
    assert "line-0:out->coupling:in-0" in edges
    assert not any(e.startswith("line-0") and "aobs" in e for e in edges)
    assert by_id(graph, "coupling").label == "UV coupling (implied)"
    assert by_id(graph, "detector").model == "Power HyD S"
    assert by_id(graph, "detector").serial_number == "1L001"
    # Descanned: back through the scanner, the coupling and the AOBS to the pinhole.
    assert {
        "objective:det-out->scanner:det-in",
        "scanner:det-out->coupling:det-in",
        "coupling:det-out->aobs:det-in",
        "aobs:det-out->pinhole:in",
    } <= edges


def test_implied_elements_say_so_and_recorded_ones_do_not(stellaris):
    _, _, graph = path_of(stellaris)
    implied = {e.id for e in graph.elements if e.label.endswith("(implied)")}
    assert implied == {"coupling", "scanner", "spectral"}
    assert by_id(graph, "spectral").description == "implied by leica.stellaris"
    assert by_id(graph, "spectral").label == "Spectral detection (implied)"


def test_a_stellaris_without_aobs_settings_still_has_its_aobs():
    element = image(
        channel(None, "HyD S 1"),
        confocal(
            '<Aotf LightSourceName="SuperContVisible Light">'
            '<LaserLineSetting LaserLine="561" IntensityDev="5" /></Aotf>',
            detector("HyD S 1", 1),
        ),
    )
    _, _, graph = path_of(element)
    assert by_id(graph, "aobs").label == "AOBS (implied)"
    assert "coupling" not in chain(graph)


# --- SP8 -----------------------------------------------------------------------------

SP8 = 'SystemTypeName="TCS SP8 X" DataSourceTypeName="Confocal"'


def test_an_sp8_with_a_td_dichroic_and_a_prism():
    element = image(
        channel(None, "HyD 3"),
        confocal(
            '<Aotf LightSourceName="Visible Light">'
            '<LaserLineSetting LaserLine="488" IntensityDev="5" />'
            '<LaserLineSetting LaserLine="561" IntensityDev="5" /></Aotf>',
            detector("HyD 3", 3, type_="HyD"),
            wheels='<Wheel FilterWheelName="Excitation Beam Splitter" FilterName="TD 488/561/633" />',
        ),
        SP8,
    )
    acquisition, [only], graph = path_of(element)
    assert acquisition.instrument.key == "leica.tcs-sp8"
    assert only.beam_splitter == "TD 488/561/633"
    assert chain(graph) == [
        "line-0", "line-1", "aotf", "beam-splitter", "scanner",
        "objective", "sample", "pinhole", "spectral", "detector",
    ]
    splitter = by_id(graph, "beam-splitter")
    assert splitter.filter_kind == I.FilterKind.DICHROIC
    assert not splitter.label.endswith("(implied)")
    assert by_id(graph, "spectral").label == "Prism + slit (SP detector) (implied)"
    assert by_id(graph, "detector").model == "HyD"


def test_sted_depletion_joins_beside_the_aobs():
    element = image(
        channel(None, "HyD 1"),
        confocal(
            '<Aotf LightSourceName="SuperContVisible Light">'
            '<LaserLineSetting LaserLine="590" IntensityDev="5" AOBSIntensityDev="100" /></Aotf>',
            detector("HyD 1", 1, type_="HyD"),
            extra='IsSTEDActive="1"',
            sted='<STED_DepletionLine Wavelength="775" IsPulsedLaser="1" LineIndex="0" />',
        ),
        SP8,
    )
    _, [only], graph = path_of(element)
    assert [(s.wavelength_nm, s.pulsed) for s in only.sted_lines] == [(775.0, True)]
    assert by_id(graph, "coupling").label == "STED coupling (implied)"
    assert "sted-0:out->coupling:in-0" in {e.id for e in graph.edges}
    devices = [d.label for d in I.read_acquisition(element).channel_devices(only).devices]
    assert "sted-0-0" in devices


def test_an_empty_depletion_line_without_sted_is_nothing():
    element = image(
        channel(None, "HyD 1"),
        confocal(
            "",
            detector("HyD 1", 1),
            sted='<STED_DepletionLine Wavelength="0" IsPulsedLaser="0" LineIndex="0" />',
        ),
        SP8,
    )
    assert I.read_channels(element)[0].sted_lines == ()


def test_a_non_descanned_detector_skips_the_scanner_pinhole_and_prism():
    element = image(
        channel(None, "HyD RLD 1"),
        confocal(
            '<Aotf LightSourceName="MP"><LaserLineSetting LaserLine="920" IntensityDev="20" /></Aotf>',
            detector("HyD RLD 1", 1, scan_type="RLD", type_="HyD"),
        ),
        'SystemTypeName="TCS SP8 DIVE"',
    )
    _, _, graph = path_of(element)
    edges = [e.id for e in graph.edges]
    assert edges[-1] == "objective:det-out->detector:in"
    assert "pinhole" not in chain(graph) and "spectral" not in chain(graph)


def test_a_transmitted_detector_sits_behind_the_sample():
    element = image(
        channel(None, "Trans PMT"),
        confocal(
            '<Aotf LightSourceName="Visible Light"><LaserLineSetting LaserLine="488" IntensityDev="5" /></Aotf>',
            detector("Trans PMT", 100, scan_type="TLD", type_="PMT"),
        ),
        SP8,
    )
    _, _, graph = path_of(element)
    assert [e.id for e in graph.edges][-1] == "sample:trans-out->detector:in"
    assert "pinhole" not in chain(graph)


# --- SP5 (LAS AF records) ------------------------------------------------------------

SP5_RECORDS = """
<ScannerSetting>
  <ScannerSettingRecord Identifier="SystemType" Variant="TCS SP5" />
  <ScannerSettingRecord Identifier="dblPinhole" Variant="9.55E-05" />
  <ScannerSettingRecord Identifier="dblZoom" Variant="2" />
</ScannerSetting>
<FilterSetting>
  <FilterSettingRecord ObjectName="HCX PL APO CS 63.0x1.40 OIL UV" ClassName="CTurret"
      Attribute="Objective" Variant="HCX PL APO CS 63.0x1.40 OIL UV" />
  <FilterSettingRecord ObjectName="HCX PL APO CS 63.0x1.40 OIL UV" ClassName="CTurret"
      Attribute="NumericalAperture" Variant="1.4" />
  <FilterSettingRecord ObjectName="HCX PL APO CS 63.0x1.40 OIL UV" ClassName="CTurret"
      Attribute="OrderNumber" Variant="11506192" />
  <FilterSettingRecord ObjectName="PMT 2" ClassName="CDetectionUnit" Attribute="State"
      Data="2" Variant="Active" />
  <FilterSettingRecord ObjectName="HyD 3" ClassName="CDetectionUnit" Attribute="State"
      Data="3" Variant="Active" />
  <FilterSettingRecord ObjectName="PMT 1" ClassName="CDetectionUnit" Attribute="State"
      Data="1" Variant="Inactive" />
  <FilterSettingRecord ObjectName="PMT 2" ClassName="CDetectionUnit" Attribute="HighVoltage"
      Data="2" Variant="750" />
  <FilterSettingRecord ObjectName="PMT 2" ClassName="CSpectrophotometerUnit"
      Description="Band 2 (left)" Variant="500" />
  <FilterSettingRecord ObjectName="PMT 2" ClassName="CSpectrophotometerUnit"
      Description="Band 2 (right)" Variant="550" />
  <FilterSettingRecord ObjectName="HyD 3" ClassName="CSpectrophotometerUnit"
      Description="Band 3 (left)" Variant="580" />
  <FilterSettingRecord ObjectName="HyD 3" ClassName="CSpectrophotometerUnit"
      Description="Band 3 (right)" Variant="640" />
  <FilterSettingRecord ObjectName="XY" ClassName="CXYZStage" Attribute="XPos" Variant="0.0421" />
</FilterSetting>
<Aotf LightSourceName="Visible">
  <LaserLineSetting LaserLine="488" IntensityDev="12" />
  <LaserLineSetting LaserLine="561" IntensityDev="8" />
</Aotf>
"""


@pytest.fixture
def sp5():
    return image(channel() + channel(), SP5_RECORDS, attrs='Application="LAS AF"')


def test_an_sp5_is_read_from_its_las_af_records(sp5):
    acquisition = I.read_acquisition(sp5)
    assert acquisition.instrument.key == "leica.tcs-sp5"
    assert acquisition.system.name == "TCS SP5"
    scan = acquisition.scan
    assert (scan.magnification, scan.numerical_aperture, scan.immersion) == (63.0, 1.4, "OIL")
    assert scan.pinhole_m == pytest.approx(9.55e-05)
    assert scan.stage_x_m == pytest.approx(0.0421)

    first, second = I.read_channels(sp5)
    assert (first.detector.name, first.detector.band_nm, first.detector.gain) == (
        "PMT 2", (500.0, 550.0), 750.0
    )
    assert (second.detector.name, second.detector.band_nm) == ("HyD 3", (580.0, 640.0))
    assert [line.wavelength_nm for line in first.lines] == [488.0, 561.0]


def test_an_sp5_light_path(sp5):
    _, _, graph = path_of(sp5)
    assert chain(graph) == [
        "line-0", "line-1", "aotf", "scanner", "objective", "sample",
        "pinhole", "spectral", "detector",
    ]
    assert by_id(graph, "detector").model == "PMT"
    assert by_id(graph, "spectral").label == "Prism + slit (SP detector) (implied)"


def test_las_af_records_without_a_name_are_an_sp5(sp5):
    records = SP5_RECORDS.replace(
        '<ScannerSettingRecord Identifier="SystemType" Variant="TCS SP5" />', ""
    )
    acquisition = I.read_acquisition(image(channel() + channel(), records, attrs=""))
    assert acquisition.instrument.key == "leica.tcs-sp5"


def test_las_af_detectors_that_do_not_match_the_channels_are_not_attached(sp5):
    channels = I.read_channels(image(channel(), SP5_RECORDS, attrs=""))
    assert channels[0].detector is None
    assert channels[0].lines == ()


# --- widefield -----------------------------------------------------------------------

CAMERA = """
<ATLCameraSettingDefinition MicroscopeModel="DMI8" ObjectiveName="HC PL APO 20x/0.80 DRY"
    Magnification="20" NumericalAperture="0.8" Immersion="DRY" CameraName="DFC9000 GTC"
    ComputationalClearing="ICC">
  {channels}
</ATLCameraSettingDefinition>
"""
CUBES = """
<WideFieldChannelConfigurator>
  <WideFieldChannelInfo FluoCubeName="DFT51010" ContrastingMethodName="FLUO"
      ExposureTime="50 ms" EmissionWavelength="525" ExcitationWavelength="475" />
  <WideFieldChannelInfo FluoCubeName="DFT51010" ContrastingMethodName="FLUO"
      ExposureTime="80 ms" EmissionWavelength="610" ExcitationWavelength="555" />
</WideFieldChannelConfigurator>
"""
THUNDER = 'SystemTypeName="THUNDER Imager Live Cell" DataSourceTypeName="Camera"'


def test_a_widefield_channel_is_a_camera_behind_a_cube():
    element = image(channel("GFP") + channel("mCherry"), CAMERA.format(channels=CUBES), THUNDER)
    acquisition, channels, graph = path_of(element, 1)
    assert acquisition.instrument.key == "leica.widefield"
    assert channels[1].widefield.cube == "DFT51010"
    assert chain(graph) == ["source", "cube", "objective", "sample", "camera"]
    camera = by_id(graph, "camera")
    assert type(camera).__name__ == "CCDElementInput"
    assert (camera.label, camera.resolution) == ("DFC9000 GTC", (512, 256))
    assert by_id(graph, "source").label == "Light source (implied)"
    assert graph.edges[0].beam.wavelength is not None  # 555 nm excitation

    devices = {d.label: d for d in acquisition.devices()}
    assert "scanner" not in devices
    clearing = {s.name: s.text for s in devices["camera"].settings}
    assert clearing["computational-clearing-ComputationalClearing"] == "ICC"
    state = acquisition.channel_devices(channels[1])
    labels = [d.label for d in state.devices]
    assert "camera-1" in labels and not any(label.startswith("pinhole") for label in labels)


def test_a_camera_definition_without_channel_info_is_still_a_camera():
    element = image(channel("GFP"), CAMERA.format(channels=""), 'SystemTypeName="DMi8"')
    _, [only], graph = path_of(element)
    assert only.widefield is None
    assert chain(graph) == ["source", "objective", "sample", "camera"]


# --- the generic model ---------------------------------------------------------------


def test_an_unknown_system_keeps_todays_light_path(stellaris):
    element = image(
        channel("ALEXA 405", "HyD S 1"),
        confocal(STELLARIS_LINES, detector("HyD S 1", 1)),
        'SystemTypeName="TCS SPE"',
    )
    acquisition, [only], graph = path_of(element)
    assert acquisition.instrument is I.GENERIC
    expected = L.channel_lightpath(only, acquisition.scan, "sample")
    assert graph.model_dump() == expected.model_dump()


def test_an_image_that_names_nothing_gets_no_identity():
    element = ET.fromstring('<Element Name="snap"><Data><Image /></Data></Element>')
    acquisition = I.read_acquisition(element)
    assert acquisition.instrument is I.GENERIC
    assert acquisition.devices() == []


# --- degrading -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hardware, attrs",
    [
        # A truncated record set: records with nothing in them.
        ("<FilterSetting><FilterSettingRecord /><FilterSettingRecord ClassName='CDetectionUnit' />"
         "</FilterSetting><ScannerSetting><ScannerSettingRecord Identifier='dblPinhole' />"
         "</ScannerSetting>", ""),
        # Numbers that are not.
        (SP5_RECORDS.replace('Variant="9.55E-05"', 'Variant="abc"')
         .replace('Variant="500"', 'Variant="left"').replace('Data="2"', 'Data="two"'),
         'SystemType="x"'),
        # A camera definition with nothing in it.
        ("<ATLCameraSettingDefinition><WideFieldChannelConfigurator>"
         "<WideFieldChannelInfo EmissionWavelength='-1' />"
         "</WideFieldChannelConfigurator></ATLCameraSettingDefinition>", ""),
        # No SystemTypeName, a garbled confocal definition.
        ("<ATLConfocalSettingDefinition Magnification='x' IsResonantScanner='maybe'>"
         "<DetectorList><Detector Name='HyD 1' IsActive='1' Channel='one' /></DetectorList>"
         "<AotfList><Aotf><LaserLineSetting LaserLine='488' IntensityDev='5' AOBSIntensityDev='?' />"
         "</Aotf></AotfList></ATLConfocalSettingDefinition>", 'SystemType="13"'),
    ],
)
def test_an_unexpected_layout_degrades_instead_of_failing(hardware, attrs):
    element = image(channel() + channel(), hardware, attrs)
    acquisition = I.read_acquisition(element)
    assert isinstance(acquisition.instrument, I.LeicaInstrument)
    acquisition.devices()
    for each in I.read_channels(element):
        acquisition.channel_devices(each)
        graph = acquisition.channel_lightpath(each, "sample", L.dimensions(element))
        if graph is not None:
            assert_ports_hold(graph)


def test_a_reader_that_raises_costs_the_detectors_not_the_image(monkeypatch, sp5):
    def broken(*args, **kwargs):
        raise RuntimeError("unexpected record")

    monkeypatch.setattr(I, "legacy_channels", broken)
    monkeypatch.setattr(I, "widefield_setting", broken)
    channels = I.read_channels(sp5)
    assert len(channels) == 2 and channels[0].detector is None
    assert I.read_acquisition(sp5).widefield is None


# --- the app ------------------------------------------------------------------------


def test_the_app_puts_the_model_on_the_anchors(stellaris):
    acquisition = I.read_acquisition(stellaris)
    [only] = I.read_channels(stellaris)
    spec = A.ChannelSpec(
        index=0,
        label="ALEXA 405",
        color=(0, 0, 255, 255),
        histogram=A.ValueHistogramInput(
            histogram=[1.0], bins=[0.0, 1.0], min=0.0, max=1.0, p1=0.0, p99=1.0
        ),
        clim=(0.0, 1.0),
        leica=only,
    )
    [anchor] = A.channel_anchors([spec], acquisition, "img", L.dimensions(stellaris))
    assert "aobs" in [e.id for e in anchor.light_graph.elements]

    fake_image = SimpleNamespace(
        tilescan=None, timestamps=None, sizes={}, path="img", name="img", uuid="u",
        dtype=np.dtype("uint8"),
    )
    state = A.series_optikit(
        SimpleNamespace(name="f.lif", id=1), fake_image, None, acquisition, {}, None, None
    )
    by_label = {d.label: d for d in state.devices}
    software = {s.name: s.text for s in by_label["acquisition"].settings}["software"]
    assert software == "LAS X 4.9.0.30221"
    assert {s.name: s.text for s in by_label["microscope"].settings}["model"] == "STELLARIS 5"
