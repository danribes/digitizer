import streamlit as st

from extractor import extract_chart_data, extract_chart_data_hybrid
from export import to_csv, to_json, to_excel, to_python, _build_combined_df
from pixel_tracer import AxisRange, generate_overlay

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

if uploaded is not None:
    image_bytes = uploaded.getvalue()
    mime_type = uploaded.type or "image/png"

    if st.button("Extract Data", type="primary"):
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
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Extraction failed: {e}")

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

        # Overlay visualization for pixel-enhanced results
        if result.get("extraction_method") == "pixel-enhanced" and result.get("axis_range"):
            st.subheader("Extraction Overlay")
            ar = result["axis_range"]
            try:
                overlay_bytes = generate_overlay(
                    image_bytes,
                    result["data_series"],
                    AxisRange(ar["x_min"], ar["x_max"], ar["y_min"], ar["y_max"]),
                )
                st.image(overlay_bytes, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate overlay: {e}")

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
