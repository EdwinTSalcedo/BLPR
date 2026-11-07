# Prepare BLPR-D for release

`prepare_blpr_d_public.py` combines the held-out evaluation archive
`dataset_challenge_val.zip` and the calibration archive `BLPR-D-Test.zip` by
default. Together they contain 727 images, which are written into a new flat
folder as `img1.jpg`, `img2.jpg`, …, plus one `labels.csv`. It sorts names within
each source archive for stable numbering, converts HEIC inputs to JPEG, and
leaves both source archives untouched.

The combined CSV retains the evaluation labels, including lux, distance and
angle, and adds the calibration plate, perspective and revision fields. It
normalizes the calibration CSV's `Image` and `PerspecviveN` headers to `imagen`
and `perspectiveN`. The `subset` column marks each annotation row as
`evaluation` or `calibration`.

## Run

Use the `blpr` Conda environment, with the existing face-anonymisation dependencies
and the repository's `ultralytics` installation. From `BLPR-repository/`:

```bash
conda activate blpr

# Inspect both source archives and their CSV joins; no inference or files are written.
python face_anonymisation/prepare_blpr_d_public.py --dry-run

# Optional small preview from the first (evaluation) archive.
python face_anonymisation/prepare_blpr_d_public.py \
  --limit 2 --review \
  --output-dir ../BLPR-datasets/BLPR-D-preview

# Optional preview from the calibration archive.
python face_anonymisation/prepare_blpr_d_public.py \
  --source-zip ../BLPR-datasets/BLPR-D/BLPR-D-Test.zip \
  --limit 2 --review \
  --output-dir ../BLPR-datasets/BLPR-D-calibration-preview

# Build the final combined release, reusing the completed 503-image evaluation
# release and anonymising only the 224-image calibration archive.
python face_anonymisation/prepare_blpr_d_public.py \
  --review \
  --reuse-public-dir ../BLPR-datasets/BLPR-D-public-fast
```

The default sources are `../BLPR-datasets/BLPR-D/dataset_challenge_val.zip` and
`../BLPR-datasets/BLPR-D/BLPR-D-Test.zip`; the default destination is
`../BLPR-datasets/BLPR-D-public-final`. Defaults resolve relative to the script, so
they also work from other working directories. Use `--source-zip PATH` to
replace the defaults, repeat `--source-zip` to combine different archives, and
use `--output-dir PATH` to choose another destination. `--labels-csv PATH`
applies a companion CSV to the first source; matching sibling `labels.csv` files
are selected automatically.

HEIC decoding uses the Pillow plugin from `pillow-heif`, included in
`face_anonymisation/requirements.txt`. Install it in the `blpr` environment if
needed:

```bash
python -m pip install pillow-heif
```

The existing release can be reused only when its audit records confirm the same
source archive, detector models and anonymisation parameters. When `--review` is
requested, it must also contain the saved review images.

The destination contains only the numbered images and `labels.csv`. A sibling
folder, `BLPR-D-public-final-audit`, holds the filename mapping, anonymisation report,
parameters and optional review images. Existing output/audit folders are never overwritten. A failed or
interrupted run stays under `BLPR-D-public-final-partial` and is not published as a
complete output folder; inspect its audit summary before retrying with a fresh
output path.

## Anonymisation settings and plate protection

The script calls the same detector, merge, blur, overlap-filter and pixel-restore
functions used by BLPR-A. It uses the documented BLPR-A settings:

| Setting | Value |
| --- | --- |
| Face detectors | YuNet and SCRFD, CPU |
| YuNet confidence | 0.5 |
| SCRFD confidence | 0.4 |
| Cross-model merge IoU | 0.45 |
| Padding on each face-box side | 0.30 |
| Gaussian blur strength | 0.5 |
| JPEG quality for altered images | 95 |
| Rejected face detection | Covers at least 25% of a protected plate box |

Detections found by only one face detector are retained. The overlap rule uses
the original face box before padding and divides intersection area by plate area.
Other false face detections, such as text on route signs, can remain. Protected plate pixels
are restored after blurring, before JPEG encoding. Images without retained face
detections are copied byte for byte, except HEIC inputs, which are transcoded to
JPEG. JPEG re-encoding can change pixels outside blurred areas.

BLPR-D supplies plate strings, without ground-truth plate boxes. The local
`models/best_lpd.pt` supplies **predicted** protection boxes using class 1 (`plate`),
confidence 0.5, and input size 640. This is the same rejection/restoration rule as
BLPR-A, with a different source of protection boxes. Missed or incorrect plate
predictions can affect protection and must be reviewed. These predictions never
replace or augment the CSV ground truth.

BLPR-D images reach 12000 × 9000 pixels, while the BLPR-A images are 2048 × 2048.
For speed, the default `--detection-strategy overview` resizes each source image to
a 2048-pixel maximum side for inference and maps boxes back to the source image.
Blurring and saving still use the original dimensions. This is much faster than
running detectors over dozens of full-resolution tiles, though it can miss small,
distant faces. To use native-resolution overlapping tiles instead, pass
`--detection-strategy tiled`; the tile size and overlap are controlled by
`--tile-size` and `--tile-overlap`. The selected strategy is recorded in the audit
summary.

The local model defaults are `face_detection_yunet_2023mar.onnx`, `det_2.5g.onnx`,
and `models/best_lpd.pt`; no weights are downloaded automatically. A model failure
stops the run rather than copying an unprocessed original into the final release.

With `--review`, audit images show **green** padded face regions, **yellow** predicted
plate boxes, and **red** rejected face detections. Review both missed faces and
incorrect exclusions before publishing; a zero-detection result does not establish
that an image contains no faces.

After the public output is complete, create a fresh contact sheet of five random
source/output pairs on each run:

```bash
python face_anonymisation/visualize_blpr_d_samples.py \
  --output-dir ../BLPR-datasets/BLPR-D-public-final
```

The sheet shows the source image, public output, amplified pixel changes, and the
detector review image when available (the anonymization run must use `--review`
to save those review images). The PNG is saved in the audit folder by default.
Pass `--seed N` to repeat a selection, `--samples N` to change its size, or
`--output PATH` to choose a PNG path.
