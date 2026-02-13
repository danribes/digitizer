import base64
import json
import logging

import anthropic

from pixel_tracer import (
    AxisRange, SeriesSpec, extract_curves_from_image, generate_overlay,
    snap_series_to_pixels, calibrate_axes, assess_extraction_accuracy,
    detect_plot_bounds,
)
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5-20250929"

EXTRACTION_TOOL = {
    "name": "store_chart_data",
    "description": "Store the extracted chart data in a structured format.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "description": "Detected chart type (e.g. line, bar, scatter, pie, polar, heatmap, ternary, other).",
            },
            "title": {
                "type": "string",
                "description": "Chart title if visible, otherwise empty string.",
            },
            "x_label": {
                "type": "string",
                "description": "X-axis label if visible, otherwise empty string.",
            },
            "y_label": {
                "type": "string",
                "description": "Y-axis label if visible, otherwise empty string.",
            },
            "data_series": {
                "type": "array",
                "description": "List of data series extracted from the chart.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Series name from legend or inferred label.",
                        },
                        "data": {
                            "type": "array",
                            "description": "Data points. Use {x, y} for XY plots or {label, value} for categorical/pie charts.",
                            "items": {
                                "type": "object",
                            },
                        },
                    },
                    "required": ["name", "data"],
                },
            },
            "axis_range": {
                "type": "object",
                "description": "The numeric range of each axis as shown on the chart. Read the minimum and maximum values directly from the axis tick labels.",
                "properties": {
                    "x_min": {"type": "number", "description": "Minimum value on the x-axis."},
                    "x_max": {"type": "number", "description": "Maximum value on the x-axis."},
                    "y_min": {"type": "number", "description": "Minimum value on the y-axis."},
                    "y_max": {"type": "number", "description": "Maximum value on the y-axis."},
                },
                "required": ["x_min", "x_max", "y_min", "y_max"],
            },
            "notes": {
                "type": "string",
                "description": "Any caveats, uncertainties, or notes about the extraction.",
            },
        },
        "required": ["chart_type", "title", "x_label", "y_label", "data_series", "axis_range", "notes"],
    },
}

REFINEMENT_TOOL = {
    "name": "store_refined_data",
    "description": "Store the refined/corrected chart data after incorporating user feedback.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "description": "Detected chart type (e.g. line, bar, scatter, pie, polar, heatmap, ternary, other).",
            },
            "title": {
                "type": "string",
                "description": "Chart title if visible, otherwise empty string.",
            },
            "x_label": {
                "type": "string",
                "description": "X-axis label if visible, otherwise empty string.",
            },
            "y_label": {
                "type": "string",
                "description": "Y-axis label if visible, otherwise empty string.",
            },
            "data_series": {
                "type": "array",
                "description": "List of data series extracted from the chart.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Series name from legend or inferred label.",
                        },
                        "data": {
                            "type": "array",
                            "description": "Data points. Use {x, y} for XY plots or {label, value} for categorical/pie charts.",
                            "items": {
                                "type": "object",
                            },
                        },
                    },
                    "required": ["name", "data"],
                },
            },
            "notes": {
                "type": "string",
                "description": "Summary of what was changed in this refinement round.",
            },
        },
        "required": ["chart_type", "title", "x_label", "y_label", "data_series", "notes"],
    },
}

REFINEMENT_SYSTEM_PROMPT = """You are a chart data extraction expert performing a refinement pass. You are given:

1. The ORIGINAL chart image — this is the ground truth.
2. An OVERLAY image — showing the current extracted data plotted on top of the original chart. Colored lines are the extracted curves; the faded background is the original chart.
3. The current extracted data as JSON.
4. User feedback describing what needs correction.

Your job:
- Compare the overlay against the original chart to see where the extraction diverges.
- Incorporate the user's feedback to correct the data_series.
- Re-extract ALL data points for ALL series (not just the ones mentioned), ensuring the corrected data matches the original chart as closely as possible.
- Preserve the chart_type, title, x_label, and y_label unless the user says they are wrong.
- In the notes field, briefly summarize what you changed.

Call the store_refined_data tool with the corrected data."""


SYSTEM_PROMPT = """You are a chart data extraction expert. You analyze chart/plot images and extract all visible data points as accurately as possible.

Instructions:
- Identify the chart type, title, and axis labels.
- Extract ALL visible data points from EVERY series in the chart.
- For XY plots (line, scatter, bar, area), return data as {\"x\": ..., \"y\": ...} objects.
- For pie/donut charts, return data as {\"label\": ..., \"value\": ...} objects.
- For heatmaps, return data as {\"x\": ..., \"y\": ..., \"value\": ...} objects.
- For polar charts, return data as {\"angle\": ..., \"radius\": ...} objects.
- For ternary charts, return data as {\"a\": ..., \"b\": ..., \"c\": ...} objects.
- Read values from axes/gridlines as precisely as possible. Use numeric values where the axis is numeric.
- If a series name is not shown in a legend, infer a reasonable name (e.g. "Series 1").
- Extract the axis range: read the minimum and maximum values from each axis's tick labels.
- Note any uncertainties in the notes field.

Accuracy guidelines:
- Extract a LARGE number of data points (50+ per series when the curve has detail). More points = better fidelity.
- For step functions (Kaplan-Meier survival curves, cumulative distributions, etc.), extract BOTH the start and end of every horizontal segment AND every vertical drop. Each step needs at least two points: one just before the drop and one just after. Do NOT smooth steps into curves.
- Capture exact boundary values: the first and last data point of each series must match what the chart shows precisely (e.g. survival curves start at exactly y=1.0 at x=0).
- When axes have clear numeric gridlines, snap values to the grid where the data visually aligns with gridlines.

Call the store_chart_data tool with the extracted data."""


def extract_chart_data(
    image_bytes: bytes,
    mime_type: str,
    chart_type_hint: str | None = None,
) -> dict:
    """Extract structured data from a chart image using Claude Vision.

    Args:
        image_bytes: Raw image bytes.
        mime_type: MIME type of the image (e.g. "image/png").
        chart_type_hint: Optional hint for chart type to guide extraction.

    Returns:
        Dict with chart_type, title, x_label, y_label, data_series, notes.
    """
    client = anthropic.Anthropic()

    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    user_text = "Extract all data from this chart image."
    if chart_type_hint and chart_type_hint.lower() != "auto-detect":
        user_text += f" This is a {chart_type_hint} chart."

    response = client.messages.create(
        model=MODEL,
        max_tokens=16384,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "store_chart_data"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "store_chart_data":
            return block.input

    raise RuntimeError("Claude did not return structured chart data.")


def refine_extraction(
    image_bytes: bytes,
    mime_type: str,
    overlay_bytes: bytes,
    current_result: dict,
    chat_history: list[dict],
) -> dict:
    """Refine an extraction based on user feedback via chat.

    Sends both the original chart image and the current overlay to Claude,
    along with the current extracted data and full chat history, so Claude
    can see where the extraction diverges and correct it.

    Args:
        image_bytes: Original chart image bytes.
        mime_type: MIME type of the original image.
        overlay_bytes: PNG bytes of the current overlay visualization.
        current_result: Current extraction result dict.
        chat_history: List of {"role": "user"/"assistant", "content": str}.

    Returns:
        Updated result dict (same shape as extract_chart_data output).
    """
    client = anthropic.Anthropic()

    b64_original = base64.standard_b64encode(image_bytes).decode("utf-8")
    b64_overlay = base64.standard_b64encode(overlay_bytes).decode("utf-8")

    # Serialize current data for context
    current_data_json = json.dumps({
        "chart_type": current_result.get("chart_type", ""),
        "title": current_result.get("title", ""),
        "x_label": current_result.get("x_label", ""),
        "y_label": current_result.get("y_label", ""),
        "data_series": current_result.get("data_series", []),
    }, indent=2)

    # Build messages: initial context with images + data, then chat history
    initial_content = [
        {
            "type": "text",
            "text": "Here is the ORIGINAL chart image:",
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": b64_original,
            },
        },
        {
            "type": "text",
            "text": "Here is the OVERLAY image showing the current extraction plotted over the original chart:",
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64_overlay,
            },
        },
        {
            "type": "text",
            "text": f"Current extracted data:\n```json\n{current_data_json}\n```",
        },
    ]

    messages = [{"role": "user", "content": initial_content}]

    # Append chat history (alternating user/assistant messages)
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        # Merge consecutive same-role messages or add new ones
        if messages and messages[-1]["role"] == role:
            # Append to existing message content
            prev = messages[-1]["content"]
            if isinstance(prev, str):
                messages[-1]["content"] = prev + "\n" + content
            elif isinstance(prev, list):
                prev.append({"type": "text", "text": content})
        else:
            messages.append({"role": role, "content": content})

    # Ensure the last message is from the user (API requirement)
    if messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": "Please refine the extraction based on the feedback above."})

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=REFINEMENT_SYSTEM_PROMPT,
        tools=[REFINEMENT_TOOL],
        tool_choice={"type": "tool", "name": "store_refined_data"},
        messages=messages,
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "store_refined_data":
            return block.input

    raise RuntimeError("Claude did not return refined chart data.")


# ---------------------------------------------------------------------------
# Automatic self-assessment and refinement
# ---------------------------------------------------------------------------

AUTO_REFINE_SYSTEM_PROMPT = """You are a strict chart data extraction quality assessor. You are given:

1. The ORIGINAL chart image — this is the ground truth.
2. An OVERLAY image — showing the current extracted data (colored lines) plotted on top of the original chart (faded background). The colored lines should sit EXACTLY on top of the original curves.

Carefully compare the colored extraction lines against the original chart curves. Check EACH of these, at multiple x-positions along every curve:

1. ALIGNMENT: At x=0, x=5, x=10, x=15, x=20, x=25, x=30 (or whatever the axis range is), does each colored extraction line sit directly on top of the corresponding original curve? Even a small systematic offset (e.g. consistently 0.05 too low or too high) is a FAILURE that must be corrected.

2. SHAPE: Do the extraction lines follow the same shape as the originals? Step functions must have steps, not smooth curves. The horizontal and vertical segments must match.

3. ENDPOINTS: Do curves start and end at exactly the right values? Survival curves start at (0, 1.0).

4. SEPARATION: Are the two curves correctly separated? If one extraction line is closer to the wrong original curve, the values are wrong.

5. CROSSINGS: If the extraction lines cross each other at points where the originals don't (or vice versa), this is wrong.

You MUST be critical. If ANY colored line deviates from the original curve by more than ~0.03 in y-value at ANY point, correct it. Do NOT say "looks accurate" unless the colored lines truly overlap the originals everywhere.

When correcting, re-read the y-values directly from the ORIGINAL chart image at many x-positions, using the gridlines as reference. Return 50+ points per series with step-function structure preserved.

Call the store_refined_data tool with the corrected data."""


def _infer_axis_range(result: dict) -> dict | None:
    """Infer axis_range from data_series min/max values.

    Uses the exact data bounds without artificial margins, since the axis
    range should match the chart's actual axes.  If the minimum is close
    to zero, snaps to zero.
    """
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

    # Snap to zero if the minimum is close (within 10% of the range)
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


def auto_refine_extraction(
    image_bytes: bytes,
    mime_type: str,
    result: dict,
    max_rounds: int = 2,
) -> dict:
    """Two-phase extraction refinement: axis calibration then accuracy-driven AI correction.

    Phase 1 — Axis calibration (CV):
        Detect plot bounds and tick marks, verify the pixel-to-data mapping,
        report alignment quality.  Then pixel-snap extracted points to actual
        curve pixels using the calibrated mapping.

    Phase 2 — Extraction accuracy assessment + AI correction:
        Measure per-series extraction error with CV.  If accuracy is below
        threshold, send specific quantitative error feedback to Claude for
        targeted correction.  Re-assess after each AI round.

    Args:
        image_bytes: Original chart image bytes.
        mime_type: MIME type of the original image.
        result: Current extraction result dict (from initial AI extraction).
        max_rounds: Max AI correction rounds (default 2).

    Returns:
        Updated result dict with ``_overlay_bytes``, ``_calibration``, and
        ``_accuracy`` keys added.
    """
    import io as _io

    ar = result.get("axis_range") or _infer_axis_range(result)
    if ar is None:
        logger.info("Cannot auto-refine: no axis range available.")
        return result

    axis = AxisRange(ar["x_min"], ar["x_max"], ar["y_min"], ar["y_max"])
    img = np.array(Image.open(_io.BytesIO(image_bytes)).convert("RGB"))
    bounds = detect_plot_bounds(img)
    if bounds is None or bounds.height < 20:
        logger.warning("Cannot auto-refine: plot bounds detection failed.")
        return result

    # =================================================================
    # PHASE 1: Axis calibration + alignment assessment
    # =================================================================
    calibration = calibrate_axes(img, bounds, axis)
    logger.info(
        "Phase 1 — Axis calibration: quality=%s, x_ticks=%d, y_ticks=%d",
        calibration["alignment_quality"],
        len(calibration["x_tick_cols"]),
        len(calibration["y_tick_rows"]),
    )
    for k, v in calibration["metrics"].items():
        logger.info("  %s: %s", k, v)

    # Pixel-snap correction using the calibrated axis mapping
    try:
        snapped = snap_series_to_pixels(image_bytes, result["data_series"], axis)
        result = {**result, "data_series": snapped}
        logger.info("Pixel-snap correction applied to %d series.", len(snapped))
    except Exception as exc:
        logger.warning("Pixel-snap failed: %s", exc)

    # Initial accuracy assessment after pixel-snap
    accuracy = assess_extraction_accuracy(img, result["data_series"], bounds, axis)
    logger.info(
        "Phase 1 — Post-snap accuracy: MAE=%.4f, within_3%%=%.1f%%, pass=%s",
        accuracy["overall_mae"],
        accuracy["overall_within_3pct"] * 100,
        accuracy["passed"],
    )

    # =================================================================
    # PHASE 2: AI correction rounds driven by CV accuracy feedback
    # =================================================================
    current = result
    for round_num in range(max_rounds):
        if accuracy["passed"]:
            logger.info("Phase 2 — Accuracy passed, skipping AI round %d.", round_num + 1)
            break

        try:
            overlay_bytes = generate_overlay(
                image_bytes,
                current["data_series"],
                axis,
            )
        except Exception as exc:
            logger.warning("Overlay generation failed in round %d: %s", round_num, exc)
            break

        # Build targeted correction prompt using CV feedback
        cv_feedback = accuracy.get("feedback_text", "")
        correction_prompt = (
            "Compare the overlay against the original. "
            "The computer-vision accuracy assessment found these specific errors:\n\n"
            f"{cv_feedback}\n\n"
            "Please correct the data_series to fix these errors. "
            "Re-read y-values from the ORIGINAL chart at the problem regions. "
            "Return ALL data points for ALL series."
            if cv_feedback else
            "Compare the overlay against the original. If the extraction is "
            "inaccurate, correct it. If it looks good, return it unchanged."
        )

        client = anthropic.Anthropic()
        b64_original = base64.standard_b64encode(image_bytes).decode("utf-8")
        b64_overlay = base64.standard_b64encode(overlay_bytes).decode("utf-8")

        current_data_json = json.dumps({
            "chart_type": current.get("chart_type", ""),
            "title": current.get("title", ""),
            "x_label": current.get("x_label", ""),
            "y_label": current.get("y_label", ""),
            "data_series": current.get("data_series", []),
        }, indent=2)

        response = client.messages.create(
            model=MODEL,
            max_tokens=16384,
            system=AUTO_REFINE_SYSTEM_PROMPT,
            tools=[REFINEMENT_TOOL],
            tool_choice={"type": "tool", "name": "store_refined_data"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the ORIGINAL chart image:"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_original,
                        },
                    },
                    {"type": "text", "text": "Here is the OVERLAY showing the current extraction (colored lines) over the original chart (faded background):"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_overlay,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Current extracted data:\n```json\n{current_data_json}\n```\n\n{correction_prompt}",
                    },
                ],
            }],
        )

        refined = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "store_refined_data":
                refined = block.input
                break

        if refined is None:
            logger.warning("AI round %d: no structured response.", round_num + 1)
            break

        # Preserve metadata from original result
        refined["extraction_method"] = current.get("extraction_method", "ai-only")
        if current.get("axis_range"):
            refined["axis_range"] = current["axis_range"]

        logger.info("AI round %d notes: %s", round_num + 1, refined.get("notes", ""))
        current = refined

        # Re-assess accuracy after AI correction
        accuracy = assess_extraction_accuracy(img, current["data_series"], bounds, axis)
        logger.info(
            "Phase 2 — Round %d accuracy: MAE=%.4f, within_3%%=%.1f%%, pass=%s",
            round_num + 1,
            accuracy["overall_mae"],
            accuracy["overall_within_3pct"] * 100,
            accuracy["passed"],
        )

    # Store diagnostics on the result
    current["_calibration"] = calibration
    current["_accuracy"] = accuracy

    # Generate final overlay
    try:
        final_overlay = generate_overlay(image_bytes, current["data_series"], axis)
        current["_overlay_bytes"] = final_overlay
    except Exception:
        pass

    return current


# ---------------------------------------------------------------------------
# Metadata extraction (for pixel-enhanced hybrid mode)
# ---------------------------------------------------------------------------

METADATA_TOOL = {
    "name": "store_chart_metadata",
    "description": "Store chart metadata needed for pixel-based curve tracing.",
    "input_schema": {
        "type": "object",
        "properties": {
            "chart_type": {
                "type": "string",
                "description": "Chart type: line, step, bar, scatter, pie, polar, heatmap, ternary, other.",
            },
            "title": {
                "type": "string",
                "description": "Chart title if visible, otherwise empty string.",
            },
            "x_label": {
                "type": "string",
                "description": "X-axis label if visible.",
            },
            "y_label": {
                "type": "string",
                "description": "Y-axis label if visible.",
            },
            "x_min": {
                "type": "number",
                "description": "Minimum value on the x-axis.",
            },
            "x_max": {
                "type": "number",
                "description": "Maximum value on the x-axis.",
            },
            "y_min": {
                "type": "number",
                "description": "Minimum value on the y-axis.",
            },
            "y_max": {
                "type": "number",
                "description": "Maximum value on the y-axis.",
            },
            "series": {
                "type": "array",
                "description": "List of data series visible in the chart.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Series name from legend or inferred.",
                        },
                        "color_rgb": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Approximate RGB color [R, G, B] (0-255) of this series' line/area.",
                        },
                        "monotonic": {
                            "type": "string",
                            "enum": ["decreasing", "increasing", "none"],
                            "description": "Whether the series is monotonically decreasing (e.g. survival curves), increasing, or neither.",
                        },
                        "line_style": {
                            "type": "string",
                            "enum": ["solid", "dashed", "dotted", "unknown"],
                            "description": "Line style of this series: solid, dashed, dotted, or unknown if not distinguishable.",
                        },
                    },
                    "required": ["name", "color_rgb", "monotonic"],
                },
            },
            "pixel_traceable": {
                "type": "boolean",
                "description": "True if the chart has clear colored lines/steps against a light background that can be traced by pixel color matching. False for bar charts, pie charts, heatmaps, or charts with overlapping fills.",
            },
        },
        "required": [
            "chart_type", "title", "x_label", "y_label",
            "x_min", "x_max", "y_min", "y_max",
            "series", "pixel_traceable",
        ],
    },
}

METADATA_SYSTEM_PROMPT = """You are a chart analysis expert. Examine the chart image and extract its metadata — do NOT extract data points.

Instructions:
- Identify the chart type, title, and axis labels.
- Read the axis ranges: the minimum and maximum numeric values on each axis.
- For each data series, identify its name (from legend or inferred) and its approximate RGB color as seen in the image.
- Determine if each series is monotonically decreasing (e.g. Kaplan-Meier survival curves), increasing, or neither.
- Identify the line style of each series: "solid" for continuous lines, "dashed" for lines with longer gaps, "dotted" for lines with short dots/gaps, or "unknown" if you cannot tell. This is especially important when multiple series share the same color and are differentiated only by line style.
- Set pixel_traceable to true ONLY if the chart has clear colored lines or step curves against a white/light background. Bars, pies, heatmaps, filled areas with overlap, and scatter-only charts are NOT pixel-traceable.

Call the store_chart_metadata tool with the metadata."""


def extract_chart_metadata(
    image_bytes: bytes,
    mime_type: str,
    chart_type_hint: str | None = None,
) -> dict:
    """Extract chart metadata (axes, colors, type) via Claude — no data points.

    Used as the first step of hybrid pixel-enhanced extraction.
    """
    client = anthropic.Anthropic()
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    user_text = "Analyze this chart and extract its metadata (axis ranges, series colors, chart type)."
    if chart_type_hint:
        user_text += f" This is a {chart_type_hint} chart."

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=METADATA_SYSTEM_PROMPT,
        tools=[METADATA_TOOL],
        tool_choice={"type": "tool", "name": "store_chart_metadata"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "store_chart_metadata":
            return block.input

    raise RuntimeError("Claude did not return chart metadata.")


PIXEL_TRACEABLE_TYPES = {"line", "step"}


def extract_chart_data_hybrid(
    image_bytes: bytes,
    mime_type: str,
    chart_type_hint: str | None = None,
) -> dict:
    """Hybrid extraction: Claude metadata + pixel-based curve tracing.

    Falls back to AI-only extraction on any failure.

    Returns:
        Dict with chart_type, title, x_label, y_label, data_series, notes,
        and extraction_method ("pixel-enhanced" or "ai-only").
    """
    try:
        meta = extract_chart_metadata(image_bytes, mime_type, chart_type_hint)
    except Exception:
        logger.warning("Metadata extraction failed, falling back to AI-only.")
        result = extract_chart_data(image_bytes, mime_type, chart_type_hint)
        result["extraction_method"] = "ai-only"
        return result

    # Check if pixel tracing is viable
    chart_type = meta.get("chart_type", "").lower()
    if not meta.get("pixel_traceable") or chart_type not in PIXEL_TRACEABLE_TYPES:
        logger.info("Chart not pixel-traceable (type=%s), using AI-only.", chart_type)
        result = extract_chart_data(image_bytes, mime_type, chart_type_hint)
        result["extraction_method"] = "ai-only"
        return result

    # Build pixel tracer inputs from metadata
    try:
        axis_range = AxisRange(
            x_min=float(meta["x_min"]),
            x_max=float(meta["x_max"]),
            y_min=float(meta["y_min"]),
            y_max=float(meta["y_max"]),
        )

        series_specs = []
        for s in meta.get("series", []):
            rgb = tuple(s["color_rgb"][:3])
            mono = s.get("monotonic", "none")
            ls = s.get("line_style")
            if ls not in ("solid", "dashed", "dotted"):
                ls = None
            series_specs.append(SeriesSpec(
                name=s["name"],
                rgb=rgb,
                tolerance=50,
                monotonic=mono if mono != "none" else None,
                line_style=ls,
            ))

        if not series_specs:
            raise ValueError("No series specs from metadata.")

        data_series = extract_curves_from_image(
            image_bytes, series_specs, axis_range, n_points=250,
        )

        # Validate: need at least 10 total data points
        total_points = sum(len(s["data"]) for s in data_series)
        if total_points < 10:
            raise ValueError(f"Pixel tracing produced only {total_points} points.")

    except Exception as exc:
        logger.warning("Pixel tracing failed (%s), falling back to AI-only.", exc)
        result = extract_chart_data(image_bytes, mime_type, chart_type_hint)
        result["extraction_method"] = "ai-only"
        return result

    return {
        "chart_type": meta.get("chart_type", "unknown"),
        "title": meta.get("title", ""),
        "x_label": meta.get("x_label", ""),
        "y_label": meta.get("y_label", ""),
        "data_series": data_series,
        "notes": "Extracted using pixel-enhanced method (Claude metadata + CV curve tracing).",
        "extraction_method": "pixel-enhanced",
        "axis_range": {
            "x_min": axis_range.x_min,
            "x_max": axis_range.x_max,
            "y_min": axis_range.y_min,
            "y_max": axis_range.y_max,
        },
    }
