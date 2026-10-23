# lif-converter

Converts Leica LIF (and LOF) files into mikro datasets **natively**: no Bio-Formats, no
JVM, no OME in between. One pyramided `ArrayDataset` per image. A tile scan gives one
dataset per tile.

It is the sibling of `ome-converter`, with the same output contract, and reads the file
with [liffile](https://pypi.org/project/liffile/):

- **Memory-mapped reads.** Each image's block is mapped where it lies in the file, chunked
  one plane at a time. Nothing is decoded into RAM, and the whole file is never copied to
  a scratch Zarr. Only pyramid level 1 (¼ of the base, ⅛ for a z-stack) is persisted,
  next to the download, so that the coarser levels do not each re-read the base.
- **Leica's own metadata.** The image's XML element goes on the image anchor verbatim,
  as `{"format": "leica-lif", "xml": …}` in the anchor's metadata blob. Each channel
  anchor gets its dye and its LUT. It also gets the sequential setting and detector that
  recorded it (gain, offset, photon-counting mode, detection band), the AOTF laser lines
  that fired, and the pinhole in µm and Airy units, as device settings and a light path.
  The image anchor pins no axis, so it holds over the whole image. It carries the stand,
  objective (NA, immersion), zoom, scan speed, dwell time, averaging and the stage. It also
  carries the acquisition window, which runs from the first frame's time stamp to the last
  (UTC). A tile gets its own frames' window, and a merged image gets its scan's window, since
  LAS X stamps the merged image with the first tile's frames only. A time series also
  states each timepoint's own window on its `{t: i}` anchor, so irregular intervals
  survive. mikro derives the dataset's `acquisitionStart`/`acquisitionEnd` from these
  windows. `leica_metadata.py` documents where in the XML each of these lives.
- **Leica-derived display.** Each channel is drawn in its LAS X LUT colour and opens on
  the LAS X display window where one was saved, as a fraction of the recorded bit depth.
  Only otherwise does it fall back to p1/p99 of what the layer shows.
- **Tuned pyramids.** `levels=0` (the default) halves until the coarsest level is
  ≤ 512 px. That gives one level for a 512 px field and six for an 11k stitched image.

## Instrument models

Each image is laid out on the microscope that recorded it (`leica_instruments.py`). The
family is read from `HardwareSetting@SystemTypeName`, or, on a LAS AF file, from its
`SystemType` scanner record. A camera setting definition always means widefield. Names
are matched loosely, and only `STELLARIS 5` has been checked against real files.

| Model | Light path |
|---|---|
| `leica.stellaris` | Lines → AOTF → AOBS (notches = the lines LAS X set it for) → scanner → objective → sample, and back through the AOBS → pinhole → spectral detection → Power HyD. The 405 diode, which the AOBS does not carry, enters through a UV coupling. |
| `leica.tcs-sp8` | As above, but with the AOBS or the TD/RSP dichroic named in the filter wheel, an SP prism + slit and PMT/HyD/HyD SMD detectors. |
| `leica.tcs-sp5` | As SP8. LAS AF 2.x files without an `ATL…SettingDefinition` are read from their `ScannerSettingRecord`/`FilterSettingRecord`s (`leica_legacy.py`). |
| `leica.widefield` | THUNDER / DMi8 / DM6: light source → filter cube → objective → sample → cube → camera (`leica_widefield.py`). |
| `leica.confocal` | Anything else: the generic confocal path, unchanged. |

Common to all models:
- STED depletion lines join through a coupling beside the AOBS.
- Non-descanned (RLD, DIVE) detectors take the light straight off the objective.
- Transmitted (TLD) detectors sit behind the sample.
- An element the file names is recorded. One only the design implies (the scanner, the spectral selector, a coupling, the AOBS of a STELLARIS that did not set it) is labelled "(implied)" and carries no numbers.

mikro has no instrument entity yet, so the model is recorded as settings on the image anchor's `microscope` device:
- `instrument-model`, `family`, `variant` and `system-type`;
- the system as `model` and the stand as `stand`.

The SP5-legacy names come from Bio-Formats' reader, and the modern widefield names are unverified. If either reader meets a layout it does not expect, the image loses those details but still converts.

## Placement and tile scans

Every calibrated image is registered into one micrometer world at its stage position.
That is the `TileScanInfo` centre of each tile, or of a merged image, falling back to
`StagePosX/Y`. A snapshot measured in pixels is not registered, and is staged in its own
pixel grid instead.

A tile scan is one LIF image with an `M` axis. Every tile is uploaded and registered. If
the file holds the merged image of those tiles, only the merged image gets a scene;
otherwise the tiles share one mosaic scene (up to 48 layers).

The merge is found **by geometry**: LAS X writes a merged image's centre as the mean of
its tiles' centres. Names are unreliable for this: `R1` / `R1_Merged` sit side by side,
but `pig overview/Series` is merged into `pig_overview_merged` at the root, and one file
can hold two `R1` scans in different folders. The name rule only applies where no
positions are recorded.

## The action

**Convert Leica LIF file to Datasets** (`convert_lif_file_to_datasets`) takes a mikro
`File`, plus `levels` (0 = auto), `method`, `stage_scene`, `world_name` and `snapshot` (on by default: each staged scene gets a maximum projection of its layers as its snapshot). Each staged scene is also set as the default scene of every dataset in it, so it is the scene a dataset opens in and where its thumbnail comes from. It returns the
datasets it made. If one image or tile fails, it is logged and the rest still convert.
FLIM images, and XLEF/XLIF experiments (whose sidecar files are not uploaded), are
refused.

## Developing

```sh
uv run pytest             # offline: no server, no file
uv run arkitekt check
uv run arkitekt plugin build -p linux/amd64
```
