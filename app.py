import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from extractor import extract_chart_data, extract_chart_data_hybrid, refine_extraction, auto_refine_extraction
from export import to_csv, to_json, to_excel, to_python, _build_combined_df
from pixel_tracer import AxisRange, generate_overlay

OVERLAYS_DIR = Path(__file__).parent / "overlays"
OVERLAYS_DIR.mkdir(exist_ok=True)

CHART_TYPES = [
    "Auto-detect",
    "Line",
    "Bar",
    "Scatter",
    "Pie",
    "Polar",
    "Heatmap",
    "Ternary",
    "Other",
]

SUPPORTED_TYPES = ["png", "jpg", "jpeg", "gif", "webp"]

st.set_page_config(page_title="Plot Digitizer", layout="wide")

st.title("Plot Digitizer")
st.markdown("Upload a chart image and extract structured data using AI vision.")

# Sidebar
chart_type_hint = st.sidebar.selectbox("Chart type hint", CHART_TYPES)
extraction_method = st.sidebar.radio(
    "Extraction method",
    ["AI-only", "Pixel-enhanced (hybrid)"],
)

# Upload
uploaded = st.file_uploader("Upload a chart image", type=SUPPORTED_TYPES)


def _get_axis_range(result: dict) -> dict | None:
    """Get axis_range from result, inferring from data if not present."""
    if result.get("axis_range"):
        return result["axis_range"]

    # Infer from data_series min/max for AI-only results
    all_x, all_y = [], []
    for series in result.get("data_series", []):
        for pt in series.get("data", []):
            if "x" in pt and "y" in pt:
                all_x.append(pt["x"])
                all_y.append(pt["y"])
    if not all_x or not all_y:
        return None

    # Add a small margin (5%) around the data range
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
    y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
    return {
        "x_min": x_min - x_margin,
        "x_max": x_max + x_margin,
        "y_min": y_min - y_margin,
        "y_max": y_max + y_margin,
    }


def _generate_overlay_cached(image_bytes: bytes, result: dict, result_version: int) -> bytes | None:
    """Generate overlay and cache in session state. Regenerates when result_version changes."""
    cached_version = st.session_state.get("overlay_version")
    if cached_version == result_version and st.session_state.get("overlay_bytes"):
        return st.session_state["overlay_bytes"]

    ar = _get_axis_range(result)
    if ar is None:
        return None

    try:
        overlay = generate_overlay(
            image_bytes,
            result["data_series"],
            AxisRange(ar["x_min"], ar["x_max"], ar["y_min"], ar["y_max"]),
        )
        st.session_state["overlay_bytes"] = overlay
        st.session_state["overlay_version"] = result_version
        _save_overlay(overlay)
        return overlay
    except Exception:
        return None


def _save_overlay(overlay_bytes: bytes) -> None:
    """Save overlay PNG to the overlays/ directory with a timestamped filename."""
    filename = st.session_state.get("overlay_filename")
    version = st.session_state.get("result_version", 0)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_overlay.png"
        st.session_state["overlay_filename"] = filename

    if version > 0:
        base, ext = os.path.splitext(filename)
        save_name = f"{base}_v{version}{ext}"
    else:
        save_name = filename

    (OVERLAYS_DIR / save_name).write_bytes(overlay_bytes)


if uploaded is not None:
    image_bytes = uploaded.getvalue()
    mime_type = uploaded.type or "image/png"

    if st.button("Extract Data", type="primary"):
        # Clear chat state on new extraction
        st.session_state.pop("chat_messages", None)
        st.session_state.pop("overlay_bytes", None)
        st.session_state.pop("overlay_version", None)
        st.session_state.pop("overlay_filename", None)
        st.session_state["result_version"] = 0

        hint = chart_type_hint if chart_type_hint != "Auto-detect" else None
        use_hybrid = extraction_method == "Pixel-enhanced (hybrid)"
        spinner_text = (
            "Analyzing chart with Claude Vision + pixel tracing..."
            if use_hybrid
            else "Analyzing chart with Claude Vision..."
        )
        with st.spinner(spinner_text):
            try:
                if use_hybrid:
                    result = extract_chart_data_hybrid(image_bytes, mime_type, chart_type_hint=hint)
                else:
                    result = extract_chart_data(image_bytes, mime_type, chart_type_hint=hint)
                    result["extraction_method"] = "ai-only"
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                result = None

        if result is not None:
            with st.spinner("Verifying extraction against original chart..."):
                try:
                    result = auto_refine_extraction(image_bytes, mime_type, result)
                    # Pull out the pre-generated overlay if available
                    overlay_from_refine = result.pop("_overlay_bytes", None)
                    if overlay_from_refine:
                        st.session_state["overlay_bytes"] = overlay_from_refine
                        st.session_state["overlay_version"] = 0
                except Exception:
                    pass  # Auto-refine is best-effort; keep original result

            st.session_state["result"] = result
            st.session_state["image_bytes"] = image_bytes
            st.session_state["mime_type"] = mime_type
            st.session_state["chat_messages"] = []
            st.session_state["result_version"] = 0

    if "result" in st.session_state:
        result = st.session_state["result"]

        # Two-column layout: image + metadata
        col_img, col_meta = st.columns(2)
        with col_img:
            st.image(image_bytes, use_container_width=True)
        with col_meta:
            if result.get("extraction_method") == "pixel-enhanced":
                st.success("Pixel-enhanced extraction")
            st.markdown(f"**Chart type:** {result.get('chart_type', 'unknown')}")
            if result.get("title"):
                st.markdown(f"**Title:** {result['title']}")
            if result.get("x_label"):
                st.markdown(f"**X-axis:** {result['x_label']}")
            if result.get("y_label"):
                st.markdown(f"**Y-axis:** {result['y_label']}")
            if result.get("notes"):
                st.info(result["notes"])
            st.markdown(f"**Series:** {len(result.get('data_series', []))}")

        # Overlay visualization (works for both pixel-enhanced and AI-only with XY data)
        result_version = st.session_state.get("result_version", 0)
        overlay = _generate_overlay_cached(image_bytes, result, result_version)
        if overlay is not None:
            st.subheader("Extraction Overlay")
            st.image(overlay, use_container_width=True)

        # Chat-based refinement section
        if overlay is not None:
            st.subheader("Refine Extraction")

            chat_messages = st.session_state.get("chat_messages", [])

            # Display chat history
            for msg in chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            user_input = st.chat_input("Describe what needs correction...")
            if user_input:
                chat_messages.append({"role": "user", "content": user_input})
                st.session_state["chat_messages"] = chat_messages

                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Refining extraction..."):
                        try:
                            stored_image = st.session_state.get("image_bytes", image_bytes)
                            stored_mime = st.session_state.get("mime_type", mime_type)

                            refined = refine_extraction(
                                image_bytes=stored_image,
                                mime_type=stored_mime,
                                overlay_bytes=overlay,
                                current_result=result,
                                chat_history=chat_messages,
                            )

                            # Preserve extraction_method and axis_range from original
                            refined["extraction_method"] = result.get("extraction_method", "ai-only")
                            if result.get("axis_range"):
                                refined["axis_range"] = result["axis_range"]

                            st.session_state["result"] = refined

                            # Assistant response from notes
                            assistant_msg = refined.get("notes", "Extraction updated.")
                            chat_messages.append({"role": "assistant", "content": assistant_msg})
                            st.session_state["chat_messages"] = chat_messages

                            # Bump version to regenerate overlay
                            st.session_state["result_version"] = result_version + 1

                            st.rerun()
                        except Exception as e:
                            error_msg = f"Refinement failed: {e}"
                            st.error(error_msg)
                            chat_messages.append({"role": "assistant", "content": error_msg})
                            st.session_state["chat_messages"] = chat_messages

        # Data table
        st.subheader("Extracted Data")
        df = _build_combined_df(result)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No data points extracted.")

        # Export buttons
        st.subheader("Export")
        col_csv, col_json, col_excel, col_py = st.columns(4)
        with col_csv:
            st.download_button(
                "CSV",
                data=to_csv(result),
                file_name="chart_data.csv",
                mime="text/csv",
            )
        with col_json:
            st.download_button(
                "JSON",
                data=to_json(result),
                file_name="chart_data.json",
                mime="application/json",
            )
        with col_excel:
            st.download_button(
                "Excel",
                data=to_excel(result),
                file_name="chart_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col_py:
            st.download_button(
                "Python",
                data=to_python(result),
                file_name="chart_data.py",
                mime="text/plain",
            )
