from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSICAL_MODELS_DIR = MODELS_DIR / "classical"
CNN_MODELS_DIR = MODELS_DIR / "cnn"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
RESULT_IMAGES_DIR = RESULTS_DIR / "images"
RESULT_VIDEOS_DIR = RESULTS_DIR / "videos"
FIGURES_DIR = RESULTS_DIR / "figures"

DOCS_DIR = PROJECT_ROOT / "docs"
