import base64
import json
import logging

import anthropic

from pixel_tracer import AxisRange, SeriesSpec, extract_curves_from_image, generate_overlay

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
            "notes": {
                "type": "string",
                "description": "Any caveats, uncertainties, or notes about the extraction.",
            },
        },
        "required": ["chart_type", "title", "x_label", "y_label", "data_series", "notes"],
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

AUTO_REFINE_SYSTEM_PROMPT = """You are a chart data extraction quality assessor. You are given:

1. The ORIGINAL chart image — this is the ground truth.
2. An OVERLAY image — showing the current extracted data (colored lines) plotted on top of the original chart (faded background).

Compare the colored extraction lines against the original chart curves in the background. Look for:
- Curves that don't start or end at the correct values
- Extracted lines that diverge significantly from the original curves
- Missing steps in step-function curves (e.g. Kaplan-Meier survival curves)
- Series that are confused or swapped
- Regions where the extraction is noticeably above or below the original

You also have the current extracted data as JSON.

If the extraction looks accurate (lines closely follow the originals), return the data unchanged and note "Extraction looks accurate" in the notes field.

If you see discrepancies, correct the data_series to better match the original chart. Return ALL data points for ALL series. In the notes field, briefly describe what you corrected.

Call the store_refined_data tool with the (corrected or confirmed) data."""


def _infer_axis_range(result: dict) -> dict | None:
    """Infer axis_range from data_series min/max values."""
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
    x_margin = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
    y_margin = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
    return {
        "x_min": x_min - x_margin,
        "x_max": x_max + x_margin,
        "y_min": y_min - y_margin,
        "y_max": y_max + y_margin,
    }


def auto_refine_extraction(
    image_bytes: bytes,
    mime_type: str,
    result: dict,
    max_rounds: int = 1,
) -> dict:
    """Generate an overlay and have Claude self-assess and correct the extraction.

    After the initial extraction, this function:
    1. Generates an overlay (extracted curves on top of the original chart)
    2. Sends both images to Claude for comparison
    3. Claude either confirms accuracy or returns corrected data

    Args:
        image_bytes: Original chart image bytes.
        mime_type: MIME type of the original image.
        result: Current extraction result dict.
        max_rounds: Number of assess-and-correct rounds (default 1).

    Returns:
        Tuple-like dict: updated result with 'overlay_bytes' key added.
    """
    ar = result.get("axis_range") or _infer_axis_range(result)
    if ar is None:
        logger.info("Cannot auto-refine: no axis range available.")
        return result

    current = result
    for round_num in range(max_rounds):
        try:
            overlay_bytes = generate_overlay(
                image_bytes,
                current["data_series"],
                AxisRange(ar["x_min"], ar["x_max"], ar["y_min"], ar["y_max"]),
            )
        except Exception as exc:
            logger.warning("Overlay generation failed in auto-refine round %d: %s", round_num, exc)
            break

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
                    {"type": "text", "text": f"Current extracted data:\n```json\n{current_data_json}\n```\n\nCompare the overlay against the original. If the extraction is inaccurate, correct it. If it looks good, return it unchanged."},
                ],
            }],
        )

        refined = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "store_refined_data":
                refined = block.input
                break

        if refined is None:
            logger.warning("Auto-refine round %d: no structured response.", round_num)
            break

        # Preserve metadata from original result
        refined["extraction_method"] = current.get("extraction_method", "ai-only")
        if current.get("axis_range"):
            refined["axis_range"] = current["axis_range"]

        logger.info("Auto-refine round %d: %s", round_num + 1, refined.get("notes", ""))
        current = refined

    # Store the final overlay bytes on the result
    try:
        final_overlay = generate_overlay(
            image_bytes,
            current["data_series"],
            AxisRange(ar["x_min"], ar["x_max"], ar["y_min"], ar["y_max"]),
        )
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
