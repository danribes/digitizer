import base64
import json
import logging

import anthropic

from pixel_tracer import AxisRange, SeriesSpec, extract_curves_from_image

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
        max_tokens=4096,
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
            series_specs.append(SeriesSpec(
                name=s["name"],
                rgb=rgb,
                tolerance=50,
                monotonic=mono if mono != "none" else None,
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
