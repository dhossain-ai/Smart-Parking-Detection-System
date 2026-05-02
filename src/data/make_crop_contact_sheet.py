from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.data.coco_utils import readable_relative_path
from src.utils.config import FIGURES_DIR, PROJECT_ROOT


LABELS = ("occupied", "vacant")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a contact sheet from exported crop samples."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/crop_samples"),
        help="Crop sample root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/crop_samples_contact_sheet.jpg"),
        help="Output contact sheet image.",
    )
    parser.add_argument(
        "--examples-per-row",
        type=int,
        default=8,
        help="Maximum examples to show for each split/label row.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        help="Displayed crop tile size.",
    )
    return parser.parse_args()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _collect_rows(input_dir: Path, examples_per_row: int) -> list[tuple[str, list[Path]]]:
    rows: list[tuple[str, list[Path]]] = []

    for split in SPLITS:
        for label in LABELS:
            crop_dir = input_dir / split / label
            if not crop_dir.is_dir():
                continue

            images = sorted(
                path
                for path in crop_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if images:
                rows.append((f"{split} / {label}", images[:examples_per_row]))

    return rows


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def make_contact_sheet(
    input_dir: Path,
    output: Path,
    examples_per_row: int,
    tile_size: int,
) -> int:
    input_path = _resolve_project_path(input_dir)
    output_path = _resolve_project_path(output)
    rows = _collect_rows(input_path, examples_per_row)

    if not rows:
        raise FileNotFoundError(
            f"No crop samples found under {input_path}. Run export_crop_samples first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_width = 150
    padding = 12
    header_height = 36
    row_height = tile_size + padding
    columns = max(len(images) for _, images in rows)
    sheet_width = label_width + padding + columns * (tile_size + padding)
    sheet_height = header_height + padding + len(rows) * row_height

    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    font = _load_font()
    draw.text((padding, 10), "PKLot crop samples", fill=(30, 35, 42), font=font)

    y = header_height
    for row_label, images in rows:
        draw.text((padding, y + 22), row_label, fill=(30, 35, 42), font=font)
        x = label_width + padding

        for image_path in images:
            with Image.open(image_path) as image:
                crop = image.convert("RGB").resize(
                    (tile_size, tile_size),
                    Image.Resampling.BILINEAR,
                )
            sheet.paste(crop, (x, y))
            draw.rectangle(
                (x, y, x + tile_size - 1, y + tile_size - 1),
                outline=(180, 188, 200),
                width=1,
            )
            x += tile_size + padding

        y += row_height

    sheet.save(output_path, quality=95)
    print(f"Contact sheet created: {readable_relative_path(output_path, PROJECT_ROOT)}")
    return 0


def main() -> int:
    args = _parse_args()
    return make_contact_sheet(
        input_dir=args.input_dir,
        output=args.output,
        examples_per_row=args.examples_per_row,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
