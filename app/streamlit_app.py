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


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def show_image_if_exists(relative_path: str | Path, caption: str, use_container_width: bool = True) -> None:
    path = project_path(relative_path)
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=use_container_width)
    else:
        st.warning(f"Missing image: `{relative_path}`")


def show_image_bytes_if_exists(relative_path: str | Path, caption: str) -> None:
    path = project_path(relative_path)
    if path.is_file():
        st.image(path.read_bytes(), caption=caption, use_container_width=True)
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
    st.markdown(
        """
        <div class="note-box">
        Random/selected images come from the PKLot test split and use COCO parking-slot coordinates.
        This tests the model on unseen test images. For a custom image, parking-slot calibration or
        annotation coordinates are required.
        </div>
        """,
        unsafe_allow_html=True,
    )

    test_slots = load_csv("data/splits/test_slots.csv")
    image_options: list[str] = []
    slot_counts: dict[str, int] = {}
    if test_slots is not None and {"image_path", "file_name"}.issubset(test_slots.columns):
        grouped = (
            test_slots.groupby("image_path", as_index=False)
            .agg(file_name=("file_name", "first"), slot_count=("image_path", "size"))
            .sort_values(["slot_count", "image_path"], ascending=[False, True])
        )
        image_options = grouped["image_path"].astype(str).tolist()
        slot_counts = dict(zip(grouped["image_path"].astype(str), grouped["slot_count"].astype(int)))
    else:
        st.warning("Test split manifest is missing or does not contain image paths.")
    random_options = [path for path in image_options if slot_counts.get(path, 0) >= 50] or image_options

    st.subheader("Input Source")
    input_source = st.radio(
        "Image input",
        ["Default demo image", "Random test image", "Select test image"],
        horizontal=True,
    )

    selected_image_path: str | None = None
    if input_source == "Random test image":
        if random_options:
            st.caption(
                "Run Detection will sample a PKLot test image. Images with at least 50 annotated slots "
                "are preferred when available."
            )
        else:
            st.error("No annotated PKLot test images are available for random selection.")
    elif input_source == "Select test image":
        if image_options:
            selected_image_path = st.selectbox(
                "PKLot test image",
                image_options,
                format_func=lambda value: Path(value).name,
            )
            st.session_state["image_demo_selected_image"] = selected_image_path
            st.caption(
                f"Full path: `{selected_image_path}` | Annotated slots: {slot_counts.get(selected_image_path, 'NA')}"
            )
        else:
            st.error("No annotated PKLot test images are available.")
    else:
        st.caption("The default demo image is selected automatically by the local image demo script.")

    model_label = st.radio("Model", ["CNN", "Classical"], horizontal=True)
    model_type = model_label.lower()

    st.text_input(
        "Custom image upload",
        value="Planned: requires parking-slot calibration before detection can run.",
        disabled=True,
    )
    st.caption(
        "Custom image upload is planned. A custom image requires parking-slot calibration because "
        "the model must know where the parking spaces are."
    )

    run_disabled = (
        (input_source == "Select test image" and selected_image_path is None)
        or (input_source == "Random test image" and not random_options)
    )
    if st.button("Run Detection", type="primary", disabled=run_disabled):
        run_image_path = selected_image_path
        if input_source == "Random test image":
            run_image_path = str(pd.Series(random_options).sample(n=1).iloc[0])
            st.session_state["image_demo_random_image"] = run_image_path
        elif input_source == "Select test image":
            st.session_state["image_demo_selected_image"] = run_image_path

        command = [sys.executable, "-m", "src.visualization.demo_image", "--model-type", model_type]
        if run_image_path:
            command.extend(["--image-path", run_image_path])

        st.session_state["image_demo_last_command"] = command_text(command)
        st.markdown("**Command**")
        st.code(st.session_state["image_demo_last_command"], language="bash")

        try:
            with st.spinner(f"Running local {model_label} image detection..."):
                result = run_command(command, timeout=120)
        except subprocess.TimeoutExpired as error:
            st.error("Image detection command timed out after 120 seconds.")
            with st.expander("Command output", expanded=True):
                stdout = timeout_output(error.stdout)
                stderr = timeout_output(error.stderr)
                if stdout:
                    st.code(stdout, language="text")
                if stderr:
                    st.code(stderr, language="text")
            result = None

        if result is not None:
            show_command_result(result)
            if result.returncode == 0:
                summary = load_json(PATHS["image_summary"]) or {}
                st.session_state["image_demo_last_model"] = model_type
                st.session_state["image_demo_last_source"] = input_source
                st.session_state["image_demo_last_image"] = summary.get(
                    "image_path",
                    run_image_path or "Default demo image",
                )
                st.session_state["image_demo_last_summary"] = summary
                st.session_state["image_demo_last_side_by_side"] = summary.get(
                    "output_side_by_side",
                    PATHS["image_cnn_side"] if model_type == "cnn" else PATHS["image_classical_side"],
                )
                st.session_state["image_demo_last_processed"] = summary.get(
                    "output_image",
                    PATHS["image_cnn"] if model_type == "cnn" else "results/images/demo_image_classical_output.jpg",
                )
                st.success("Image detection output refreshed from disk.")
            else:
                st.error(f"Image detection failed with return code {result.returncode}.")

    current_image_label = selected_image_path or "Default demo image"
    if input_source == "Random test image":
        current_image_label = st.session_state.get("image_demo_random_image", "Random test image not run yet")
    latest_model = st.session_state.get("image_demo_last_model", model_type)
    latest_source = st.session_state.get("image_demo_last_source", input_source)
    latest_image = st.session_state.get("image_demo_last_image", current_image_label)

    summary_path = PATHS["image_cnn_summary"] if latest_model == "cnn" else PATHS["image_classical_summary"]
    side_by_side_path = PATHS["image_cnn_side"] if latest_model == "cnn" else PATHS["image_classical_side"]
    processed_path = PATHS["image_cnn"] if latest_model == "cnn" else "results/images/demo_image_classical_output.jpg"
    summary = st.session_state.get("image_demo_last_summary")
    if not summary:
        summary = load_json(summary_path) or load_json(PATHS["image_summary"])
    side_by_side_path = st.session_state.get("image_demo_last_side_by_side", side_by_side_path)
    processed_path = st.session_state.get("image_demo_last_processed", processed_path)

    st.subheader("Latest Output")
    st.caption(f"Source: {latest_source} | Image: `{latest_image}` | Model: {latest_model.upper()}")
    last_command = st.session_state.get("image_demo_last_command")
    if last_command:
        with st.expander("Last command"):
            st.code(last_command, language="bash")
    if summary:
        cols = st.columns(4)
        cols[0].metric("Total slots", summary.get("total_slots", "NA"))
        cols[1].metric("Occupied", summary.get("occupied_count", "NA"))
        cols[2].metric("Vacant", summary.get("vacant_count", "NA"))
        occupancy_rate = summary.get("occupancy_rate")
        occupancy_text = f"{float(occupancy_rate) * 100:.1f}%" if isinstance(occupancy_rate, (int, float)) else "NA"
        cols[3].metric("Occupancy rate", occupancy_text)

    tab_output, tab_side_by_side, tab_summary = st.tabs(["Processed Image", "Side-by-Side", "Summary JSON"])
    with tab_output:
        show_image_bytes_if_exists(processed_path, f"{latest_model.upper()} processed image")
    with tab_side_by_side:
        show_image_bytes_if_exists(side_by_side_path, f"{latest_model.upper()} original vs processed")
    with tab_summary:
        st.json(summary or {"status": "missing"})


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
