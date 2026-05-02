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

Each split folder also contains JPG images. Phase 2 will parse COCO bounding boxes and labels to generate parking-slot crops for occupancy classification.

XML support can remain available for original PKLot layouts, but current work should use the COCO split layout above.

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

Phase 2 will handle annotation parsing and parking-slot crop generation. Phase 1 only verifies that the local dataset folder has image files and recognized annotation files.
