import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from extractor import extract_chart_data, extract_chart_data_hybrid, refine_extraction, auto_refine_extraction
from export import to_csv, to_json, to_excel, to_python, _build_combined_df
from pixel_tracer import AxisRange, PlotBounds, generate_overlay

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

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Snap to zero if minimum is close (within 10% of range)
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    if 0 <= x_min <= x_range * 0.1:
        x_min = 0
    if 0 <= y_min <= y_range * 0.1:
        y_min = 0

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
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


def _normalize_image(raw_bytes: bytes, reported_mime: str | None) -> tuple[bytes, str]:
    """Re-encode image to PNG if the format isn't natively supported by the API,
    or if the reported MIME type doesn't match the actual file contents."""
    import io as _io
    try:
        img = Image.open(_io.BytesIO(raw_bytes))
        fmt = (img.format or "").upper()
    except Exception:
        return raw_bytes, reported_mime or "image/png"

    # Map PIL format names to MIME types
    fmt_to_mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}
    actual_mime = fmt_to_mime.get(fmt)

    # Re-encode to PNG if: format unsupported by API (GIF), or MIME mismatch
    needs_reencode = fmt not in ("PNG", "JPEG", "WEBP") or (
        reported_mime and actual_mime and reported_mime != actual_mime
    )
    if needs_reencode:
        buf = _io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    return raw_bytes, actual_mime or reported_mime or "image/png"


if uploaded is not None:
    image_bytes, mime_type = _normalize_image(uploaded.getvalue(), uploaded.type)

    # --- Calibration: user must click the (0, 1.0) origin ---
    st.subheader("Calibration")
    st.markdown(
        "Click on the chart where the **origin (0, 1.0)** is located "
        "(top-left corner of the plot area) and enter the maximum X-axis value."
    )

    # Determine original image dimensions for coordinate scaling
    pil_img = Image.open(uploaded)
    orig_w, orig_h = pil_img.size
    uploaded.seek(0)  # reset after PIL read

    # Render at a fixed display width so we can compute the scale factor.
    display_w = min(orig_w, 700)
    scale = orig_w / display_w

    # Show clickable image
    coords = streamlit_image_coordinates(
        pil_img, width=display_w, key="cal_image", cursor="crosshair",
    )

    # Process click — only act on NEW clicks, not replayed values.
    if coords is not None:
        click_key = (coords["x"], coords["y"])
        if click_key != st.session_state.get("_last_cal_click"):
            st.session_state["_last_cal_click"] = click_key
            px_col = int(coords["x"] * scale)
            px_row = int(coords["y"] * scale)
            st.session_state["cal_origin_pixel"] = (px_col, px_row)

    origin_px = st.session_state.get("cal_origin_pixel")
    if origin_px:
        st.caption(f"Origin pixel: ({origin_px[0]}, {origin_px[1]})")
    else:
        st.info("Click on the (0, 1.0) point on the chart above.")

    x_max_val = st.number_input(
        "Maximum X-axis value", value=5.0, min_value=0.01, key="cal_x_max",
    )

    # Build calibration_points when origin is set.
    # The origin click gives us the top-left pixel (data = 0, 1.0).
    # Auto-detect the bottom-right pixel from the image.
    if origin_px:
        import io as _io
        import numpy as np
        from pixel_tracer import detect_plot_bounds
        img_arr = np.array(Image.open(_io.BytesIO(image_bytes)).convert("RGB"))
        auto_bounds = detect_plot_bounds(img_arr)
        if auto_bounds:
            bottom_right = (auto_bounds.right, auto_bounds.bottom)
        else:
            # Fallback: estimate bottom-right from image dimensions
            bottom_right = (int(orig_w * 0.90), int(orig_h * 0.85))

        st.session_state["calibration_points"] = {
            "pixel_1": origin_px,
            "data_1": (0.0, 1.0),
            "pixel_2": bottom_right,
            "data_2": (x_max_val, 0.0),
        }
        st.success(
            f"Origin (0, 1.0) at pixel ({origin_px[0]}, {origin_px[1]}), "
            f"X-axis range: 0 – {x_max_val}"
        )

    cal_pts = st.session_state.get("calibration_points")
    extract_disabled = origin_px is None
    if st.button("Extract Data", type="primary", disabled=extract_disabled):
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
                    result = extract_chart_data_hybrid(
                        image_bytes, mime_type,
                        chart_type_hint=hint,
                        calibration_points=cal_pts,
                    )
                else:
                    result = extract_chart_data(image_bytes, mime_type, chart_type_hint=hint)
                    result["extraction_method"] = "ai-only"
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                result = None

        if result is not None:
            # Derive bounds_override for auto_refine if calibration was used
            bounds_override = None
            if cal_pts:
                bounds_override, _ = PlotBounds.from_calibration_points(
                    pixel_1=cal_pts["pixel_1"],
                    data_1=cal_pts["data_1"],
                    pixel_2=cal_pts["pixel_2"],
                    data_2=cal_pts["data_2"],
                )
            with st.spinner("Verifying extraction against original chart..."):
                try:
                    result = auto_refine_extraction(
                        image_bytes, mime_type, result,
                        bounds_override=bounds_override,
                    )
                    # Pull out the pre-generated overlay if available
                    overlay_from_refine = result.pop("_overlay_bytes", None)
                    if overlay_from_refine:
                        st.session_state["overlay_bytes"] = overlay_from_refine
                        st.session_state["overlay_version"] = 0
                except Exception:
                    pass  # Auto-refine is best-effort; keep original result

            # Ensure all series start at (0, 1.0) — the calibrated origin.
            # The pixel tracer may not trace all the way to the axis origin.
            for series in result.get("data_series", []):
                pts = series.get("data", [])
                if not pts:
                    continue
                first = pts[0]
                if abs(first.get("x", 0) - 0) > 1e-9 or abs(first.get("y", 1) - 1.0) > 1e-9:
                    series["data"] = [{"x": 0.0, "y": 1.0}] + pts

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
