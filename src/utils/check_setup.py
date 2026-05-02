from __future__ import annotations

from src.utils.config import (
    CNN_MODELS_DIR,
    CLASSICAL_MODELS_DIR,
    DATA_DIR,
    DOCS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RESULT_IMAGES_DIR,
    RESULT_VIDEOS_DIR,
    SAMPLES_DIR,
    SPLITS_DIR,
)


REQUIRED_DOCS = [
    "AI_CONTEXT.md",
    "PROJECT_RULES.md",
    "PHASE_PLAN.md",
    "METRICS_REQUIREMENTS.md",
    "UI_MOCKUP_REQUIREMENTS.md",
]

REQUIRED_DIRS = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLES_DIR,
    SPLITS_DIR,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "data",
    PROJECT_ROOT / "src" / "classical",
    PROJECT_ROOT / "src" / "neural",
    PROJECT_ROOT / "src" / "evaluation",
    PROJECT_ROOT / "src" / "visualization",
    PROJECT_ROOT / "src" / "utils",
    PROJECT_ROOT / "app",
    CLASSICAL_MODELS_DIR,
    CNN_MODELS_DIR,
    METRICS_DIR,
    RESULT_IMAGES_DIR,
    RESULT_VIDEOS_DIR,
    FIGURES_DIR,
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "slides",
]


def main() -> int:
    missing_docs = [name for name in REQUIRED_DOCS if not (DOCS_DIR / name).is_file()]
    missing_dirs = [path for path in REQUIRED_DIRS if not path.is_dir()]

    print("Smart Parking Detection System setup check")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Required docs: {len(REQUIRED_DOCS) - len(missing_docs)}/{len(REQUIRED_DOCS)} present")
    print(f"Required directories: {len(REQUIRED_DIRS) - len(missing_dirs)}/{len(REQUIRED_DIRS)} present")

    if missing_docs:
        print("Missing docs:")
        for name in missing_docs:
            print(f"- {DOCS_DIR / name}")

    if missing_dirs:
        print("Missing directories:")
        for path in missing_dirs:
            print(f"- {path}")

    if missing_docs or missing_dirs:
        print("Setup status: incomplete")
        return 1

    print("Setup status: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
