# PKLot Dataset Setup Guide

## Dataset

This project uses the PKLot Dataset from Kaggle. The dataset must be downloaded manually by the user and kept local to the machine running the project.

Do not commit the full PKLot dataset to GitHub.

## Expected Contents

The PKLot dataset contains parking lot surveillance images and annotation files. Original PKLot-style exports may use XML annotation files with parking-slot coordinates and occupancy labels.

Depending on the archive layout, the dataset may include weather or category folders with names such as:

- Sunny
- Rainy
- Cloudy
- Overcast

Exact local counts should only be trusted after running the dataset checker.

## Roboflow/COCO PKLot Version

The current local dataset is the Roboflow COCO export of PKLot. It uses split folders and COCO annotation JSON files:

```text
data/raw/PKLot/train/_annotations.coco.json
data/raw/PKLot/valid/_annotations.coco.json
data/raw/PKLot/test/_annotations.coco.json
```

Each split folder also contains JPG images. Phase 2 parses COCO bounding boxes and labels to generate project-ready parking-slot metadata for occupancy classification.

XML support can remain available for original PKLot layouts, but current work should use the COCO split layout above.

## Phase 2 COCO Preprocessing

Phase 2 creates metadata first instead of blindly exporting every crop image. This keeps the repo practical on disk and lets later training scripts crop parking slots on the fly from known COCO boxes.

Generate slot metadata CSV files:

```bash
python -m src.data.prepare_pklot_coco
```

Outputs:

```text
data/processed/metadata/slot_annotations.csv
data/processed/metadata/split_summary.csv
data/processed/metadata/category_summary.csv
```

The slot metadata normalizes COCO category names such as `space-empty`, `vacant`, `empty`, `space-occupied`, and `occupied` to:

```text
occupied
vacant
```

Occupied is the positive class. Vacant is the negative class.

For a quick smoke test, limit images per split:

```bash
python -m src.data.prepare_pklot_coco --limit-images 10 --output-dir data/processed/metadata_smoke
```

Export a small crop sample set for visual verification only:

```bash
python -m src.data.export_crop_samples --samples-per-class 20
```

Create a contact sheet after sample crops exist:

```bash
python -m src.data.make_crop_contact_sheet
```

Full crop export is optional and should not be run by default:

```bash
python -m src.data.export_crop_samples --export-all
```

By default, later model training should read `slot_annotations.csv` and crop from source images on the fly to avoid storing hundreds of thousands of small generated image files.

## Recommended Location

Download the dataset manually from Kaggle, unzip it, and place it at:

```text
data/raw/PKLot/
```

The full dataset is ignored by Git.

## Custom Location

If the dataset lives somewhere else, set `PKLOT_DATA_DIR` to the PKLot root directory:

```bash
export PKLOT_DATA_DIR=/absolute/path/to/PKLot
```

An example variable is provided in `.env.example`. Do not commit a real `.env` file.

## Local Checks

Run the Phase 0 setup check:

```bash
python3 -m src.utils.check_setup
```

Check whether PKLot is available locally:

```bash
python3 -m src.data.check_pklot_dataset
```

Print the dataset path resolved by the project configuration:

```bash
python3 -c "from src.utils.config import get_pklot_dir; print(get_pklot_dir())"
```

If the dataset is not downloaded yet, the checker will fail with instructions. That is expected before manually downloading and unzipping PKLot.

## Next Phase

Phase 2 handles COCO annotation parsing, parking-slot metadata generation, limited crop samples, and visual QA. Phase 1 only verifies that the local dataset folder has image files and recognized annotation files.
