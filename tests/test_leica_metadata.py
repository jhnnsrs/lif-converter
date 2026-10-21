"""Offline checks of the Leica XML reader, on fragments shaped like real LAS X headers.

The fixture is an SP8 / STELLARIS sequential acquisition cut down to the attributes the
reader touches: two sequential settings with two active detectors each, four channels
that name their setting and detector, and the display windows in both places LAS X
keeps them. The attribute values are the ones found in real files.
"""

import json
import xml.etree.ElementTree as ET

import pytest

import leica_metadata as L

CHANNEL = """
<ChannelDescription DataType="0" ChannelTag="0" Resolution="{bits}" LUTName="{lut}"
    BytesInc="{inc}" BitInc="0">
  {props}
</ChannelDescription>
"""

PROPS = """
  <ChannelProperty><Key>DetectorName</Key><Value>{detector}</Value></ChannelProperty>
  <ChannelProperty><Key>DyeName</Key><Value>Leica/{dye}</Value></ChannelProperty>
  <ChannelProperty><Key>SequentialSettingIndex</Key><Value>{step}</Value></ChannelProperty>
"""

DETECTOR = """
<Detector Name="{name}" Type="SiPM" Channel="{channel}" IsActive="{active}"
    DyeName="Leica/{dye}" Gain="2.5" Offset="0" AcquisitionModeName="PhotonCounting"
    DetectionRangeBegin="320" DetectionRangeEnd="1">
  <LutInfo LutDescription="{lut}" LutBlackValue="{black}" LutWhiteValue="{white}"
      LutGammaValue="1" />
</Detector>
"""

SETTING = """
<ATLConfocalSettingDefinition UserSettingName="{name}" Pinhole="1.0286E-04"
    PinholeAiry="1">
  <Spectro>
    <MultiBand Channel="1" LeftWorld="{band1_lo}" RightWorld="{band1_hi}" />
    <MultiBand Channel="2" LeftWorld="654" RightWorld="750" />
    <MultiBand Channel="3" LeftWorld="562" RightWorld="666" />
  </Spectro>
  <AotfList>
    <Aotf LightSourceName="Visible Light">{lines}</Aotf>
  </AotfList>
  <DetectorList>{detectors}</DetectorList>
</ATLConfocalSettingDefinition>
"""

LINE = '<LaserLineSetting LaserLine="{wl}" IntensityDev="{intensity}" />'

TOP = """
<ATLConfocalSettingDefinition UserSettingName="S54" MicroscopeModel="DMI8-CS"
    SystemSerialNumber="8200000630" ObjectiveName="HC PL APO CS2    63x/1.30 GLYC "
    ObjectiveNumber="11506353" Magnification="63" NumericalAperture="1.3"
    Immersion="GLYC" RefractionIndex="1.46" Pinhole="1.02915564956813E-04"
    PinholeAiry="1.00040254144613" Zoom="1" ScanSpeed="400" ScanDirectionXName="Bidirectional"
    PixelDwellTime="0.0000010375" LineAverage="1" FrameAverage="1" Line_Accumulation="2"
    FrameAccumulation="1" BitSize="8" StagePosX="0.06612836946257"
    StagePosY="0.0378991016894" FlipX="0" FlipY="1" SwapXY="1" ScanMode="xyz" />
"""


def image_xml(
    channels: str,
    sequential: str = "",
    viewer: str = "",
    dims: str = "",
    top: str = TOP,
) -> ET.Element:
    dims = dims or (
        '<DimensionDescription DimID="1" NumberOfElements="1024" Origin="1.376765e-20"'
        ' Length="1.845238e-04" Unit="m" BytesInc="1" BitInc="0" />'
        '<DimensionDescription DimID="2" NumberOfElements="1024" Origin="1.376765e-20"'
        ' Length="1.845238e-04" Unit="m" BytesInc="1024" BitInc="0" />'
        '<DimensionDescription DimID="3" NumberOfElements="36" Origin="1.51853e-03"'
        ' Length="1.171940e-05" Unit="m" BytesInc="4194304" BitInc="0" />'
    )
    return ET.fromstring(
        f"""
<Element Name="DP_5_1" UniqueID="3f7a8e61-ac68-11ed-bb17-0050622022c0">
  <Data><Image>
    <ImageDescription>
      <Channels>{channels}</Channels>
      <Dimensions>{dims}</Dimensions>
    </ImageDescription>
    <Attachment Name="HardwareSetting">
      {top}
      <LDM_Block_Sequential><LDM_Block_Sequential_List>{sequential}</LDM_Block_Sequential_List></LDM_Block_Sequential>
    </Attachment>
    <Attachment Name="ViewerScaling">{viewer}</Attachment>
  </Image></Data>
  <Children><Element Name="a child that is not this image" /></Children>
</Element>
"""
    )


# Four channels, as DP_5_1 records them: Setting 1 runs HyD S 1 (ALEXA 405) and HyD S 3
# (ALEXA 546), Setting 3 runs HyD S 1 again (ALEXA 488) and HyD S 2 (ATTO 647N). The same
# detector name in two settings is the case a lookup by name alone gets wrong.
DP_CHANNELS = [
    ("Gray", "HyD S 1", "ALEXA 405", 0),
    ("Yellow", "HyD S 3", "ALEXA 546", 0),
    ("Cyan", "HyD S 1", "ALEXA 488", 1),
    ("Magenta", "HyD S 2", "ATTO 647N", 1),
]


def dp_sequential(**overrides) -> str:
    def detector(name, channel, dye, lut, white="0.5", active="1"):
        return DETECTOR.format(
            name=name, channel=channel, dye=dye, lut=lut, black="0", white=white,
            active=active,
        )

    one = SETTING.format(
        name="Setting 1",
        band1_lo=410,
        band1_hi=517,
        lines=LINE.format(wl=405, intensity="1.38777878078145E-16")
        + LINE.format(wl=557, intensity="3.5"),
        detectors=detector("HyD S 1", 1, "ALEXA 405", "Gray", white="0.768627451")
        + detector("HyD S 2", 2, "unused", "Red", active="0")
        + detector("HyD S 3", 3, overrides.get("dye_546", "ALEXA 546"), "Yellow"),
    )
    three = SETTING.format(
        name="Setting 3",
        band1_lo=504,
        band1_hi=564,
        lines=LINE.format(wl=497, intensity="2.5") + LINE.format(wl=646, intensity="1.5"),
        detectors=detector("HyD S 1", 1, "ALEXA 488", "Cyan", white="1")
        + detector("HyD S 2", 2, "ATTO 647N", "Magenta"),
    )
    return one + three


def dp_channels(with_props: bool = True, count: int = 4) -> str:
    return "".join(
        CHANNEL.format(
            bits=8,
            lut=lut,
            inc=index * 1048576,
            props=PROPS.format(detector=detector, dye=dye, step=step) if with_props else "",
        )
        for index, (lut, detector, dye, step) in enumerate(DP_CHANNELS[:count])
    )


VIEWER_DEFAULT = '<ChannelScalingInfo BlackValue="0" WhiteValue="1" GammaValue="1" />'


@pytest.fixture
def dp():
    return image_xml(dp_channels(), dp_sequential(), VIEWER_DEFAULT * 4)


def test_each_channel_finds_its_own_setting_and_detector(dp):
    channels = L.leica_channels(dp)
    assert [c.label for c in channels] == ["ALEXA 405", "ALEXA 546", "ALEXA 488", "ATTO 647N"]
    assert [c.setting_name for c in channels] == ["Setting 1"] * 2 + ["Setting 3"] * 2
    assert [c.detector.name for c in channels] == ["HyD S 1", "HyD S 3", "HyD S 1", "HyD S 2"]
    # HyD S 1 recorded twice, through two different bands: the setting decides which.
    assert channels[0].detector.band_nm == (410.0, 517.0)
    assert channels[2].detector.band_nm == (504.0, 564.0)
    assert channels[1].detector.band_nm == (562.0, 666.0)
    assert channels[0].detector.mode == "PhotonCounting"
    assert channels[0].pinhole_airy == 1.0


def test_a_closed_line_is_not_a_laser_that_fired(dp):
    first, _, third, _ = L.leica_channels(dp)
    # 405 at 1.4e-16 % is floating-point dust, not an excitation line.
    assert [line.wavelength_nm for line in first.lines] == [557.0]
    assert [(line.wavelength_nm, line.intensity_percent) for line in third.lines] == [
        (497.0, 2.5),
        (646.0, 1.5),
    ]


def test_a_default_viewer_window_gives_way_to_the_detector_lut(dp):
    first, second, third, _ = L.leica_channels(dp)
    assert (first.window.black, first.window.white) == (0.0, 0.768627451)
    assert second.window.white == 0.5
    # The detector's window is the default too: nothing to claim.
    assert third.window is None


def test_a_set_viewer_window_wins():
    viewer = (
        '<ChannelScalingInfo BlackValue="0.1" WhiteValue="0.6" GammaValue="1" />'
        + VIEWER_DEFAULT * 3
    )
    channels = L.leica_channels(image_xml(dp_channels(), dp_sequential(), viewer))
    assert (channels[0].window.black, channels[0].window.white) == (0.1, 0.6)
    assert channels[1].window.white == 0.5


def test_viewer_windows_that_do_not_count_the_channels_are_not_aligned():
    viewer = '<ChannelScalingInfo BlackValue="0.1" WhiteValue="0.6" GammaValue="1" />' * 3
    channels = L.leica_channels(image_xml(dp_channels(), dp_sequential(), viewer))
    assert channels[0].window.white == 0.768627451


def test_a_detector_that_recorded_another_dye_is_not_attached():
    image = image_xml(dp_channels(), dp_sequential(dye_546="ALEXA 594"), VIEWER_DEFAULT * 4)
    channels = L.leica_channels(image)
    assert channels[1].detector is None
    assert channels[1].lines == () and channels[1].setting_name is None
    # The channel's own facts stay.
    assert (channels[1].label, channels[1].lut, channels[1].bits) == ("ALEXA 546", "Yellow", 8)
    assert channels[0].detector is not None


def test_without_properties_the_active_detectors_are_taken_in_order():
    channels = L.leica_channels(image_xml(dp_channels(with_props=False), dp_sequential()))
    assert [c.detector.dye for c in channels] == [
        "ALEXA 405",
        "ALEXA 546",
        "ALEXA 488",
        "ATTO 647N",
    ]


def test_without_properties_a_count_mismatch_attaches_nothing():
    three = dp_channels(with_props=False, count=3)
    channels = L.leica_channels(image_xml(three, dp_sequential()))
    assert len(channels) == 3
    assert all(c.detector is None for c in channels)


def test_without_sequential_settings_the_scan_wide_one_is_the_step():
    top = TOP.replace(
        "/>",
        ">"
        + "<DetectorList>"
        + DETECTOR.format(
            name="HyD S 2", channel=2, dye="ALEXA 647", lut="Red", black="0", white="1",
            active="1",
        )
        + "</DetectorList></ATLConfocalSettingDefinition>",
        1,
    )
    channel = CHANNEL.format(
        bits=12,
        lut="Red",
        inc=0,
        props=PROPS.format(detector="HyD S 2", dye="ALEXA 647", step=0),
    )
    [only] = L.leica_channels(image_xml(channel, "", top=top))
    assert only.detector.name == "HyD S 2"
    assert only.setting_name == "S54"
    assert only.bits == 12


def test_the_scan(dp):
    scan = L.leica_scan(dp)
    assert scan.objective_name == "HC PL APO CS2    63x/1.30 GLYC"
    assert (scan.magnification, scan.numerical_aperture) == (63.0, 1.3)
    assert scan.immersion_kind == L.ObjectiveImmersion.GLYCEROL
    assert scan.line_accumulation == 2.0
    assert (scan.flip_x, scan.flip_y, scan.swap_xy) == (False, True, True)
    assert scan.stage_x_m == pytest.approx(0.06612836946257)


def test_dimensions(dp):
    dims = L.dimensions(dp)
    assert set(dims) == {"X", "Y", "Z"}
    # Length runs from the first element to the last: N - 1 spacings, the same pixel size
    # Bio-Formats reports for this file.
    assert dims["X"].spacing_um == pytest.approx(0.18037517106549364)
    assert dims["Z"].spacing_um == pytest.approx(0.33484)
    assert dims["Z"].origin == pytest.approx(1.51853e-03)


def test_a_snapshot_in_pixels_has_no_spacing():
    dims = L.dimensions(
        image_xml(
            "",
            dims='<DimensionDescription DimID="1" NumberOfElements="1097" Origin="0"'
            ' Length="1.098e+03" Unit="Pixel" BytesInc="3" BitInc="0" />',
        )
    )
    assert dims["X"].spacing_um is None


def test_channel_optikit_and_lightpath(dp):
    scan = L.leica_scan(dp)
    _, _, third, _ = L.leica_channels(dp)
    state = L.channel_optikit(third)
    labels = [d.label for d in state.devices]
    assert labels == ["channel-2", "detector-2", "laser-2-0", "laser-2-1", "pinhole-2"]

    graph = L.channel_lightpath(third, scan, "DP_5_1")
    kinds = [type(e).__name__ for e in graph.elements]
    assert kinds == [
        "LaserElementInput",
        "LaserElementInput",
        "BeamSplitterElementInput",
        "ObjectiveElementInput",
        "SampleElementInput",
        "PinholeElementInput",
        "DetectorElementInput",
    ]
    # Every edge names a port its element declares — nothing checks that locally but
    # this.
    ports = {(e.id, p.id) for e in graph.elements for p in e.ports}
    for edge in graph.edges:
        assert (edge.source_element_id, edge.source_port_id) in ports
        assert (edge.target_element_id, edge.target_port_id) in ports
    # The light crosses glycerol at the front lens.
    assert {edge.medium for edge in graph.edges if "sample" in edge.id} == {"GLYCEROL"}


def test_a_channel_with_nothing_attached_has_no_lightpath():
    bare = L.LeicaChannel(0, "Gray", 8, None, None, None, (), None, None, None)
    assert L.channel_lightpath(bare, None, "x") is None


def test_the_record_is_leica_xml_not_ome(dp):
    record = json.loads(L.image_record("f.lif", "7", "DP_5_1", "uuid", None, dp))
    assert record["format"] == "leica-lif"
    xml = ET.fromstring(record["xml"])
    assert xml.find("./Data/Image/Attachment[@Name='HardwareSetting']") is not None
    # This image's element, not its descendants'.
    assert xml.find("./Children") is None
