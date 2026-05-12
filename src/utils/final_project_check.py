from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CheckItem:
    label: str
    path: str
    critical: bool = False
    command: str | None = None


DOCS = [
    CheckItem("Final report", "docs/FINAL_REPORT.md", critical=True),
    CheckItem("Demo script", "docs/DEMO_SCRIPT.md", critical=True),
    CheckItem("Final submission checklist", "docs/FINAL_SUBMISSION_CHECKLIST.md", critical=True),
    CheckItem("Streamlit app guide", "docs/STREAMLIT_APP_GUIDE.md", critical=True),
    CheckItem("Slide outline", "slides/presentation_outline.md", critical=True),
]

MODELS = [
    CheckItem(
        "Classical model",
        "models/classical/classical_lbp_hsv_hog_svm.joblib",
        command="python -m src.classical.train_classical",
    ),
    CheckItem(
        "MobileNetV3 final model",
        "models/cnn/best_mobilenetv3_transfer_final.pth",
        command=(
            "python -m src.neural.train_cnn --model-version mobilenet_v3_small --pretrained "
            "--image-size 224 --epochs 30 --batch-size 64 --samples-per-class 12000 "
            "--patience 8 --weight-decay 0.0001 "
            "--output-model models/cnn/best_mobilenetv3_transfer_final.pth "
            "--output-dir results/metrics/mobilenetv3_transfer_final"
        ),
    ),
]

METRICS = [
    CheckItem(
        "Model comparison",
        "results/metrics/comparison/model_comparison.csv",
        command="python -m src.evaluation.compare_models",
    ),
    CheckItem(
        "Requirement checklist",
        "results/metrics/comparison/requirement_checklist.csv",
        command="python -m src.evaluation.compare_models",
    ),
    CheckItem(
        "Classical metrics",
        "results/metrics/classical/classical_metrics.json",
        command="python -m src.classical.train_classical",
    ),
    CheckItem(
        "MobileNetV3 metrics",
        "results/metrics/mobilenetv3_transfer_final/cnn_metrics.json",
        command=(
            "python -m src.neural.train_cnn --model-version mobilenet_v3_small --pretrained "
            "--image-size 224 --epochs 30 --batch-size 64 --samples-per-class 12000 "
            "--patience 8 --weight-decay 0.0001 "
            "--output-model models/cnn/best_mobilenetv3_transfer_final.pth "
            "--output-dir results/metrics/mobilenetv3_transfer_final"
        ),
    ),
]

DEMO_OUTPUTS = [
    CheckItem(
        "CNN image side-by-side demo",
        "results/images/demo_image_cnn_side_by_side.jpg",
        command="python -m src.visualization.demo_image --model-type cnn",
    ),
]

APP = [
    CheckItem("Streamlit app", "app/streamlit_app.py", critical=True),
]


def exists(relative_path: str) -> bool:
    return (PROJECT_ROOT / relative_path).is_file()


def print_group(title: str, items: list[CheckItem]) -> list[CheckItem]:
    print(f"\n{title}")
    print("-" * len(title))
    missing: list[CheckItem] = []
    for item in items:
        status = "OK" if exists(item.path) else "missing"
        print(f"{status:7} {item.path} ({item.label})")
        if status == "missing":
            missing.append(item)
    return missing


def print_regeneration_help(missing_items: list[CheckItem]) -> None:
    commands = []
    for item in missing_items:
        if item.command and item.command not in commands:
            commands.append(item.command)

    if not commands:
        return

    print("\nHelpful regeneration commands")
    print("----------------------------")
    for command in commands:
        print(command)


def main() -> int:
    print("Smart Parking Detection System final project check")
    print(f"Project root: {PROJECT_ROOT}")

    missing_docs = print_group("Docs", DOCS)
    missing_models = print_group("Models", MODELS)
    missing_metrics = print_group("Metrics", METRICS)
    missing_demos = print_group("Demo outputs", DEMO_OUTPUTS)
    missing_app = print_group("App", APP)

    all_missing = missing_docs + missing_models + missing_metrics + missing_demos + missing_app
    print_regeneration_help(all_missing)

    missing_critical = [item for item in all_missing if item.critical]
    print("\nFinal status")
    print("------------")
    if missing_critical:
        print("Failed: required source/docs files are missing.")
        for item in missing_critical:
            print(f"missing {item.path}")
        return 1

    if all_missing:
        print("Passed with notes: source/docs are present; some generated local artifacts are missing.")
        print("Missing generated outputs do not fail this check because they are ignored by Git.")
    else:
        print("Passed: all checked files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
