from __future__ import annotations

from pathlib import Path

from src.utils.config import DEFAULT_PKLOT_DIR, get_pklot_dir


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
XML_EXTENSION = ".xml"
WEATHER_KEYWORDS = ("sunny", "rainy", "cloudy", "overcast")
SAMPLE_LIMIT = 5


def _collect_files(dataset_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _detect_weather_categories(paths: list[Path], dataset_dir: Path) -> list[str]:
    categories: set[str] = set()

    for path in paths:
        try:
            parts = path.relative_to(dataset_dir).parts
        except ValueError:
            parts = path.parts

        for part in parts:
            part_lower = part.lower()
            for keyword in WEATHER_KEYWORDS:
                if keyword in part_lower:
                    categories.add(keyword)

    return sorted(categories)


def _format_sample_paths(paths: list[Path], dataset_dir: Path) -> list[str]:
    samples: list[str] = []

    for path in paths[:SAMPLE_LIMIT]:
        try:
            samples.append(str(path.relative_to(dataset_dir)))
        except ValueError:
            samples.append(str(path))

    return samples


def _print_samples(title: str, samples: list[str]) -> None:
    print(title)
    if not samples:
        print("- None found")
        return

    for sample in samples:
        print(f"- {sample}")


def _print_missing_instructions(dataset_dir: Path) -> None:
    print("Dataset status: missing")
    print("Download the PKLot Dataset manually from Kaggle, then either:")
    print(f"- unzip it into {DEFAULT_PKLOT_DIR}")
    print("- or set PKLOT_DATA_DIR=/absolute/path/to/PKLot")
    print("Do not commit the full dataset or Kaggle credentials.")
    print(f"Resolved dataset path: {dataset_dir}")


def main() -> int:
    dataset_dir = get_pklot_dir()

    print("PKLot dataset check")
    print(f"Dataset path: {dataset_dir}")
    print(f"Exists: {'Yes' if dataset_dir.exists() else 'No'}")

    if not dataset_dir.exists():
        _print_missing_instructions(dataset_dir)
        return 1

    image_paths = _collect_files(dataset_dir, IMAGE_EXTENSIONS)
    xml_paths = _collect_files(dataset_dir, {XML_EXTENSION})
    weather_categories = _detect_weather_categories(image_paths + xml_paths, dataset_dir)

    print(f"Images found: {len(image_paths)}")
    print(f"XML files found: {len(xml_paths)}")
    print(
        "Weather categories detected: "
        + (", ".join(weather_categories) if weather_categories else "None")
    )
    _print_samples("Sample image paths:", _format_sample_paths(image_paths, dataset_dir))
    _print_samples("Sample XML paths:", _format_sample_paths(xml_paths, dataset_dir))

    if not image_paths or not xml_paths:
        print("Dataset status: incomplete")
        print("Expected both parking lot image files and XML annotation files.")
        print("No XML parsing or crop generation was performed.")
        return 2

    print("Dataset status: OK")
    print("No XML parsing or crop generation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
