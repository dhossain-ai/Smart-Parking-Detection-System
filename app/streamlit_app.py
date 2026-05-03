from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PATHS = {
    "model_comparison": "results/metrics/comparison/model_comparison.csv",
    "requirement_checklist": "results/metrics/comparison/requirement_checklist.csv",
    "false_occupancy": "results/metrics/comparison/false_occupancy_comparison.csv",
    "final_report": "results/metrics/comparison/final_evaluation_report.md",
    "weather_report": "results/metrics/comparison/weather_robustness_report.md",
    "weather_placeholder": "results/metrics/comparison/weather_robustness_placeholder.csv",
    "classical_metrics": "results/metrics/classical/classical_metrics.json",
    "cnn_metrics": "results/metrics/cnn_tuned/cnn_metrics.json",
    "image_cnn": "results/images/demo_image_cnn_output.jpg",
    "image_cnn_side": "results/images/demo_image_cnn_side_by_side.jpg",
    "image_classical_side": "results/images/demo_image_classical_side_by_side.jpg",
    "image_summary": "results/metrics/demo_image/demo_image_summary.json",
    "image_cnn_summary": "results/metrics/demo_image/demo_image_cnn_summary.json",
    "image_classical_summary": "results/metrics/demo_image/demo_image_classical_summary.json",
    "video_cnn": "results/videos/demo_video_cnn_output.mp4",
    "video_cnn_side": "results/videos/demo_video_cnn_side_by_side.mp4",
    "video_classical": "results/videos/demo_video_classical_output.mp4",
    "video_classical_side": "results/videos/demo_video_classical_side_by_side.mp4",
    "video_trend": "results/metrics/demo_video/demo_video_occupancy_trend.csv",
    "video_cnn_summary": "results/metrics/demo_video/demo_video_cnn_summary.json",
    "video_classical_summary": "results/metrics/demo_video/demo_video_classical_summary.json",
    "fig_model_metrics": "results/figures/model_metrics_comparison.png",
    "fig_requirement": "results/figures/requirement_checklist.png",
    "fig_false_occupancy": "results/figures/false_occupancy_comparison.png",
    "fig_classical_cm": "results/figures/classical_confusion_matrix.png",
    "fig_cnn_cm": "results/figures/cnn_tuned_confusion_matrix.png",
}


def project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_exists(relative_path: str | Path) -> bool:
    return project_path(relative_path).is_file()


def load_json(relative_path: str | Path) -> dict[str, Any] | None:
    path = project_path(relative_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        st.warning(f"Could not parse JSON: `{relative_path}`")
        return None


def load_csv(relative_path: str | Path) -> pd.DataFrame | None:
    path = project_path(relative_path)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        st.warning(f"Could not parse CSV: `{relative_path}`")
        return None


def load_markdown(relative_path: str | Path) -> str | None:
    path = project_path(relative_path)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def run_command(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def show_image_if_exists(relative_path: str | Path, caption: str, use_container_width: bool = True) -> None:
    path = project_path(relative_path)
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=use_container_width)
    else:
        st.warning(f"Missing image: `{relative_path}`")


def show_video_if_exists(relative_path: str | Path, caption: str) -> None:
    path = project_path(relative_path)
    if path.is_file():
        st.markdown(f"**{caption}**")
        st.video(str(path))
    else:
        st.warning(f"Missing video: `{relative_path}`")


def metric_value(metrics: dict[str, Any] | None, key: str, default: str = "NA") -> str:
    if not metrics:
        return default
    value = (metrics.get("test") or {}).get(key)
    if isinstance(value, (int, float)):
        return f"{value:.5f}"
    return default if value in (None, "") else str(value)


def status_from_checklist(method: str) -> str:
    checklist = load_csv(PATHS["requirement_checklist"])
    if checklist is None:
        return "Unavailable"
    rows = checklist.loc[checklist["method"] == method]
    if rows.empty:
        return "Unavailable"
    if bool((rows["passed"] == "Yes").all()):
        return "Met"
    if bool((rows["passed"] == "Yes").any()):
        return "Partially met"
    return "Not met"


def show_command_result(result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode == 0:
        st.success("Command completed successfully.")
    else:
        st.error(f"Command failed with exit code {result.returncode}.")
    with st.expander("Command output", expanded=result.returncode != 0):
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")


def card(title: str, body: str, tone: str = "default") -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-title">{title}</div>
            <div class="metric-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.7rem;
            max-width: 1280px;
        }
        .hero {
            padding: 1.1rem 1.3rem;
            border-radius: 8px;
            background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
            color: white;
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2rem;
            margin-bottom: 0.25rem;
        }
        .hero p {
            margin: 0;
            color: #dbeafe;
        }
        .metric-card {
            min-height: 102px;
            border: 1px solid #dbe4ef;
            border-left: 5px solid #0f766e;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #ffffff;
            box-shadow: 0 1px 8px rgba(15, 23, 42, 0.06);
            margin-bottom: 0.75rem;
        }
        .metric-card.green { border-left-color: #16a34a; }
        .metric-card.red { border-left-color: #dc2626; }
        .metric-card.blue { border-left-color: #2563eb; }
        .metric-card .metric-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748b;
            font-weight: 700;
        }
        .metric-card .metric-body {
            font-size: 1.35rem;
            color: #0f172a;
            font-weight: 750;
            margin-top: 0.28rem;
        }
        .note-box {
            border: 1px solid #bae6fd;
            background: #f0f9ff;
            color: #075985;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        .warning-box {
            border: 1px solid #fed7aa;
            background: #fff7ed;
            color: #9a3412;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dashboard_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Smart Parking Detection System</h1>
            <p>Offline Local Monitoring Tool</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    classical = load_json(PATHS["classical_metrics"])
    cnn = load_json(PATHS["cnn_metrics"])
    image_available = "Yes" if file_exists(PATHS["image_cnn_side"]) else "No"
    video_available = "Yes" if file_exists(PATHS["video_cnn_side"]) else "No"
    dataset_slots = metric_value(cnn, "records", default=metric_value(classical, "records"))

    cols = st.columns(4)
    with cols[0]:
        card("Classical Accuracy", metric_value(classical, "accuracy"), "green")
    with cols[1]:
        card("CNN Accuracy", metric_value(cnn, "accuracy"), "blue")
    with cols[2]:
        card("Classical Requirement", status_from_checklist("Classical"), "green")
    with cols[3]:
        card("CNN Requirement", status_from_checklist("CNN"), "red")

    cols = st.columns(3)
    with cols[0]:
        card("Dataset Slots", dataset_slots, "blue")
    with cols[1]:
        card("Image Demo Available", image_available, "green" if image_available == "Yes" else "red")
    with cols[2]:
        card("Video Demo Available", video_available, "green" if video_available == "Yes" else "red")

    st.markdown(
        """
        <div class="note-box">
        No external internet APIs are used. The app runs local OpenCV + scikit-learn + PyTorch demo pipelines.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if file_exists(PATHS["image_cnn_side"]):
        show_image_if_exists(PATHS["image_cnn_side"], "CNN image demo side-by-side")


def image_detection_page() -> None:
    st.header("Image Detection")
    st.info("This demo uses known parking-slot coordinates from COCO annotations.")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Regenerate CNN image demo", type="primary"):
            with st.spinner("Running local CNN image demo..."):
                result = run_command([sys.executable, "-m", "src.visualization.demo_image", "--model-type", "cnn"])
            show_command_result(result)
    with cols[1]:
        if st.button("Regenerate classical image demo"):
            with st.spinner("Running local classical image demo..."):
                result = run_command([sys.executable, "-m", "src.visualization.demo_image", "--model-type", "classical"])
            show_command_result(result)

    tab_cnn, tab_classical, tab_summary = st.tabs(["CNN Output", "Classical Output", "Summary"])
    with tab_cnn:
        show_image_if_exists(PATHS["image_cnn"], "CNN processed image")
        show_image_if_exists(PATHS["image_cnn_side"], "CNN original vs processed")
    with tab_classical:
        show_image_if_exists(PATHS["image_classical_side"], "Classical original vs processed")
    with tab_summary:
        summary = load_json(PATHS["image_summary"])
        if summary:
            st.json(summary)
        cnn_summary = load_json(PATHS["image_cnn_summary"])
        classical_summary = load_json(PATHS["image_classical_summary"])
        cols = st.columns(2)
        with cols[0]:
            st.subheader("CNN latest saved summary")
            st.json(cnn_summary or {"status": "missing"})
        with cols[1]:
            st.subheader("Classical latest saved summary")
            st.json(classical_summary or {"status": "missing"})


def video_detection_page() -> None:
    st.header("Video Detection")
    st.markdown("**Annotated PKLot frame-sequence video demo**")
    st.info("The current dataset provides annotated image frames, so the video demo is generated from PKLot test frames.")

    cols = st.columns(2)
    with cols[0]:
        if st.button("Regenerate CNN video demo", type="primary"):
            with st.spinner("Running local CNN frame-sequence video demo..."):
                result = run_command([
                    sys.executable,
                    "-m",
                    "src.visualization.demo_video",
                    "--model-type",
                    "cnn",
                    "--num-frames",
                    "30",
                    "--fps",
                    "5",
                ])
            show_command_result(result)
    with cols[1]:
        if st.button("Regenerate classical video demo"):
            with st.spinner("Running local classical frame-sequence video demo..."):
                result = run_command([
                    sys.executable,
                    "-m",
                    "src.visualization.demo_video",
                    "--model-type",
                    "classical",
                    "--num-frames",
                    "15",
                    "--fps",
                    "5",
                ])
            show_command_result(result)

    tab_cnn, tab_classical, tab_trend = st.tabs(["CNN Videos", "Classical Videos", "Occupancy Trend"])
    with tab_cnn:
        cols = st.columns(2)
        with cols[0]:
            show_video_if_exists(PATHS["video_cnn"], "CNN processed video")
        with cols[1]:
            show_video_if_exists(PATHS["video_cnn_side"], "CNN original vs processed video")
    with tab_classical:
        cols = st.columns(2)
        with cols[0]:
            show_video_if_exists(PATHS["video_classical"], "Classical processed video")
        with cols[1]:
            show_video_if_exists(PATHS["video_classical_side"], "Classical original vs processed video")
    with tab_trend:
        trend = load_csv(PATHS["video_trend"])
        if trend is not None:
            st.dataframe(trend, use_container_width=True)
            if "occupancy_rate" in trend.columns:
                st.line_chart(trend.set_index("frame_index")["occupancy_rate"])
        else:
            st.warning("Occupancy trend CSV is not available yet.")


def compare_models_page() -> None:
    st.header("Classical vs CNN Comparison")
    st.markdown(
        """
        <div class="warning-box">
        Classical method meets its requirement. Current CNN method partially meets the strict modern requirement.
        </div>
        """,
        unsafe_allow_html=True,
    )
    for label, path in [
        ("Model comparison", PATHS["model_comparison"]),
        ("Requirement checklist", PATHS["requirement_checklist"]),
        ("False occupancy comparison", PATHS["false_occupancy"]),
    ]:
        frame = load_csv(path)
        st.subheader(label)
        if frame is not None:
            st.dataframe(frame, use_container_width=True)
        else:
            st.warning(f"Missing table: `{path}`")

    cols = st.columns(3)
    with cols[0]:
        show_image_if_exists(PATHS["fig_model_metrics"], "Model metrics comparison")
    with cols[1]:
        show_image_if_exists(PATHS["fig_requirement"], "Requirement checklist")
    with cols[2]:
        show_image_if_exists(PATHS["fig_false_occupancy"], "False occupancy comparison")


def metrics_page() -> None:
    st.header("Metrics")
    classical = load_json(PATHS["classical_metrics"])
    cnn = load_json(PATHS["cnn_metrics"])
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Classical test metrics")
        if classical:
            st.dataframe(pd.DataFrame([classical.get("test", {})]), use_container_width=True)
            with st.expander("Raw classical metrics JSON"):
                st.json(classical)
        else:
            st.warning("Classical metrics JSON is missing.")
    with cols[1]:
        st.subheader("CNN tuned test metrics")
        if cnn:
            st.dataframe(pd.DataFrame([cnn.get("test", {})]), use_container_width=True)
            with st.expander("Raw CNN tuned metrics JSON"):
                st.json(cnn)
        else:
            st.warning("CNN tuned metrics JSON is missing.")

    cols = st.columns(2)
    with cols[0]:
        show_image_if_exists(PATHS["fig_classical_cm"], "Classical confusion matrix")
    with cols[1]:
        show_image_if_exists(PATHS["fig_cnn_cm"], "CNN tuned confusion matrix")

    report = load_markdown(PATHS["final_report"])
    st.subheader("Final evaluation report")
    if report:
        st.markdown(report)
    else:
        st.warning("Final evaluation report is missing.")


def weather_page() -> None:
    st.header("Weather Robustness")
    st.markdown(
        """
        The original PKLot dataset includes sunny/rainy/cloudy conditions.
        The current Roboflow COCO export does not preserve reliable weather labels.
        The project includes a weather-label template for future weather-specific evaluation.
        No fake weather-wise accuracy is reported.
        """
    )
    report = load_markdown(PATHS["weather_report"])
    if report:
        st.markdown(report)
    else:
        st.warning("Weather robustness report is missing.")

    placeholder = load_csv(PATHS["weather_placeholder"])
    if placeholder is not None:
        st.dataframe(placeholder, use_container_width=True)


def explainability_page() -> None:
    st.header("Explainability")
    st.markdown(
        """
        Explainability / Grad-CAM is planned as an optional enhancement.

        Current visual explanation: red/green overlays and confidence scores on known parking-slot regions.
        No Grad-CAM output is shown here because it has not been implemented yet.
        """
    )


def calibration_page() -> None:
    st.header("Calibration")
    st.markdown(
        """
        This project uses known COCO bounding boxes for PKLot.

        For a new real camera or arbitrary video, parking slots must be calibrated once for that fixed camera view.
        Calibration JSON support can be added later so the app can reuse slot polygons or rectangles for live feeds.

        This keeps the task as parking-slot occupancy classification, not general vehicle detection.
        """
    )


def settings_page() -> None:
    st.header("Settings / About")
    st.markdown(
        """
        Smart Parking Detection System is a fully offline local computer vision demo.

        **Positive class:** `occupied = positive class`

        **Libraries used:** OpenCV, NumPy, Pandas, scikit-image, scikit-learn, PyTorch, Matplotlib, Plotly, Streamlit.

        **Offline constraint:** no external internet APIs, hosted inference services, or cloud vision systems are used.
        """
    )
    st.code("python -m src.visualization.demo_image --model-type cnn", language="bash")
    st.code("python -m src.visualization.demo_video --model-type cnn --num-frames 30 --fps 5", language="bash")


def main() -> None:
    st.set_page_config(
        page_title="Smart Parking Detection System",
        page_icon="P",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    st.sidebar.title("Smart Parking")
    st.sidebar.caption("Offline local dashboard")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Image Detection",
            "Video Detection",
            "Compare Models",
            "Metrics",
            "Weather Robustness",
            "Explainability",
            "Calibration",
            "Settings/About",
        ],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.success("Offline mode")
    st.sidebar.markdown("Green = vacant  \nRed = occupied")

    pages = {
        "Dashboard": dashboard_page,
        "Image Detection": image_detection_page,
        "Video Detection": video_detection_page,
        "Compare Models": compare_models_page,
        "Metrics": metrics_page,
        "Weather Robustness": weather_page,
        "Explainability": explainability_page,
        "Calibration": calibration_page,
        "Settings/About": settings_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
