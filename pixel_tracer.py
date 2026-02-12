"""Generic pixel-based curve tracer for chart images.

Generalized from extract_km_cv.py. All hardcoded values replaced with
parameters so it works with any line/step chart given axis ranges and
series color specs from Claude metadata extraction.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter


@dataclass
class PlotBounds:
    """Pixel boundaries of the plot area."""

    left: int
    right: int
    top: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass
class AxisRange:
    """Data-coordinate ranges for the axes."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class SeriesSpec:
    """Specification for one data series to trace."""

    name: str
    rgb: tuple[int, int, int]
    tolerance: int = 50
    monotonic: str | None = None  # "decreasing", "increasing", or None


def _find_longest_runs(
    dark: np.ndarray,
) -> tuple[dict, dict]:
    """Find the longest horizontal and vertical contiguous dark runs.

    Returns (best_h, best_v) dicts with row/col, start, end, len keys.
    """
    h, w = dark.shape

    best_h: dict = {"row": -1, "len": 0, "start": 0, "end": 0}
    for row in range(h):
        run_start = None
        for col in range(w):
            if dark[row, col]:
                if run_start is None:
                    run_start = col
            else:
                if run_start is not None:
                    run_len = col - run_start
                    if run_len > best_h["len"]:
                        best_h = {"row": row, "len": run_len, "start": run_start, "end": col - 1}
                    run_start = None
        if run_start is not None:
            run_len = w - run_start
            if run_len > best_h["len"]:
                best_h = {"row": row, "len": run_len, "start": run_start, "end": w - 1}

    best_v: dict = {"col": -1, "len": 0, "start": 0, "end": 0}
    for col in range(w):
        run_start = None
        for row in range(h):
            if dark[row, col]:
                if run_start is None:
                    run_start = row
            else:
                if run_start is not None:
                    run_len = row - run_start
                    if run_len > best_v["len"]:
                        best_v = {"col": col, "len": run_len, "start": run_start, "end": row - 1}
                    run_start = None
        if run_start is not None:
            run_len = h - run_start
            if run_len > best_v["len"]:
                best_v = {"col": col, "len": run_len, "start": run_start, "end": h - 1}

    return best_h, best_v


def detect_plot_bounds(img: np.ndarray, min_run: int = 150) -> PlotBounds | None:
    """Detect plot area by finding long dark horizontal/vertical pixel runs.

    Tries progressively lighter thresholds to handle gray axis lines.
    If a y-axis line cannot be found, falls back to using the x-axis
    start column as the left bound.

    Args:
        img: RGB image array (H, W, 3), uint8.
        min_run: Minimum pixel run length to consider an axis line.

    Returns:
        PlotBounds or None if detection fails.
    """
    h, w = img.shape[:2]
    gray = img.mean(axis=2)

    best_h: dict | None = None
    best_v: dict | None = None

    # Try progressively lighter thresholds
    for threshold in (80, 128, 160):
        dark = gray < threshold
        bh, bv = _find_longest_runs(dark)

        if best_h is None or bh["len"] > best_h["len"]:
            if bh["len"] >= min_run:
                best_h = bh
        if best_v is None or bv["len"] > best_v["len"]:
            if bv["len"] >= min_run:
                best_v = bv

        if best_h is not None and best_v is not None:
            break

    if best_h is None:
        return None

    # If we found x-axis but no y-axis, infer left bound from x-axis start
    # and top bound by scanning upward for the topmost dark curve pixel.
    if best_v is None:
        left = best_h["start"]
        # Scan for topmost dark pixel above the x-axis in the plot area
        dark_any = gray < 160
        top = best_h["row"]
        for row in range(best_h["row"] - 1, -1, -1):
            if np.any(dark_any[row, left:best_h["end"] + 1]):
                top = row
        return PlotBounds(
            left=left,
            right=best_h["end"],
            top=top,
            bottom=best_h["row"],
        )

    return PlotBounds(
        left=best_v["col"],
        right=best_h["end"],
        top=best_v["start"],
        bottom=best_h["row"],
    )


def pixel_to_data(
    col: float, row: float, bounds: PlotBounds, axis_range: AxisRange
) -> tuple[float, float]:
    """Convert pixel coordinates to data coordinates."""
    x_frac = (col - bounds.left) / (bounds.right - bounds.left)
    y_frac = (bounds.bottom - row) / (bounds.bottom - bounds.top)
    x = axis_range.x_min + x_frac * (axis_range.x_max - axis_range.x_min)
    y = axis_range.y_min + y_frac * (axis_range.y_max - axis_range.y_min)
    return x, y


def make_color_mask(
    img: np.ndarray, target_rgb: tuple[int, int, int], tolerance: int = 50
) -> np.ndarray:
    """Create boolean mask for pixels matching the target color.

    For gray targets (low channel spread), adds a spread constraint to avoid
    matching colored pixels. Excludes near-white (background) and near-black
    (axes/gridlines) pixels.

    Args:
        img: RGB image array (H, W, 3), uint8.
        target_rgb: Target (R, G, B) color.
        tolerance: Per-channel tolerance for matching.

    Returns:
        Boolean array (H, W).
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    tr, tg, tb = target_rgb

    # Per-channel match
    in_range = (
        (r >= tr - tolerance) & (r <= tr + tolerance)
        & (g >= tg - tolerance) & (g <= tg + tolerance)
        & (b >= tb - tolerance) & (b <= tb + tolerance)
    )

    # Exclude near-white (background) and near-black (axes)
    brightness = r.astype(int) + g.astype(int) + b.astype(int)
    not_white = brightness < 700  # ~233 per channel
    not_black = brightness > 100  # ~33 per channel
    mask = in_range & not_white & not_black

    # For gray targets, add channel spread constraint
    channel_spread = int(max(target_rgb)) - int(min(target_rgb))
    if channel_spread < 30:
        rgb_stack = np.stack([r, g, b], axis=2).astype(np.int16)
        pixel_spread = rgb_stack.max(axis=2) - rgb_stack.min(axis=2)
        mask = mask & (pixel_spread < 30)

    return mask


def trace_curve(
    mask: np.ndarray,
    bounds: PlotBounds,
    axis_range: AxisRange,
    n_points: int = 250,
    monotonic: str | None = None,
) -> list[dict]:
    """Trace a curve from a color mask within the plot bounds.

    Per-column segment detection with continuity tracking, median filter,
    optional monotonicity enforcement, and downsampling.

    Args:
        mask: Boolean mask (H, W) of matching pixels.
        bounds: Pixel boundaries of the plot area.
        axis_range: Data-coordinate ranges.
        n_points: Number of output points after downsampling.
        monotonic: "decreasing", "increasing", or None.

    Returns:
        List of {"x": float, "y": float} dicts.
    """
    row_min = bounds.top + 5
    row_max = bounds.bottom - 10

    if row_min >= row_max:
        row_min = bounds.top
        row_max = bounds.bottom

    col_to_row: dict[int, float] = {}
    prev_row: float | None = None

    for col in range(bounds.left, bounds.right + 1):
        rows = np.where(mask[row_min:row_max + 1, col])[0] + row_min
        if len(rows) == 0:
            continue

        # Split into contiguous segments
        diffs = np.diff(rows)
        splits = np.where(diffs > 3)[0] + 1
        segments = np.split(rows, splits)

        candidates = [s for s in segments if len(s) >= 2]
        if not candidates:
            continue

        if prev_row is not None:
            if monotonic == "decreasing":
                # Allow downward movement, limit upward jumps
                below_or_near = [s for s in candidates if s.mean() >= prev_row - 5]
                pool = below_or_near if below_or_near else candidates
            elif monotonic == "increasing":
                above_or_near = [s for s in candidates if s.mean() <= prev_row + 5]
                pool = above_or_near if above_or_near else candidates
            else:
                pool = candidates
            best = min(pool, key=lambda s: abs(s.mean() - prev_row))
        else:
            # First detection: pick topmost (highest y-value in data coords)
            if monotonic == "decreasing":
                best = min(candidates, key=lambda s: s.mean())  # topmost pixel row
            else:
                best = min(candidates, key=lambda s: s.mean())

        center = best.mean()
        col_to_row[col] = center
        prev_row = center

    if not col_to_row:
        return []

    # Build arrays
    cols = sorted(col_to_row.keys())
    raw_x = np.array([pixel_to_data(c, 0, bounds, axis_range)[0] for c in cols])
    raw_y = np.array([pixel_to_data(0, col_to_row[c], bounds, axis_range)[1] for c in cols])

    # Median filter to remove artifacts
    if len(raw_y) >= 7:
        smooth_y = median_filter(raw_y, size=7)
    else:
        smooth_y = raw_y.copy()

    # Enforce monotonicity if requested
    if monotonic == "decreasing":
        smooth_y = np.minimum.accumulate(smooth_y)
    elif monotonic == "increasing":
        smooth_y = np.maximum.accumulate(smooth_y)

    # Downsample
    if len(raw_x) > n_points:
        indices = np.linspace(0, len(raw_x) - 1, n_points, dtype=int)
        raw_x = raw_x[indices]
        smooth_y = smooth_y[indices]

    return [
        {"x": round(float(x), 4), "y": round(float(y), 4)}
        for x, y in zip(raw_x, smooth_y)
    ]


def extract_curves_from_image(
    image_bytes: bytes,
    series_specs: list[SeriesSpec],
    axis_range: AxisRange,
    n_points: int = 250,
) -> list[dict]:
    """Extract curves from a chart image using pixel-level color detection.

    Top-level entry point. Auto-detects plot bounds, builds color masks per
    series, traces each curve, and returns data in the standard format.

    Args:
        image_bytes: Raw image bytes (PNG/JPEG/WebP).
        series_specs: List of SeriesSpec for each series to trace.
        axis_range: Data-coordinate axis ranges from metadata.
        n_points: Number of output points per series.

    Returns:
        List of {"name": str, "data": [{"x": float, "y": float}, ...]} dicts.
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    bounds = detect_plot_bounds(img)
    if bounds is None:
        raise ValueError("Could not detect plot boundaries (no axis lines found).")

    results = []
    for spec in series_specs:
        mask = make_color_mask(img, spec.rgb, spec.tolerance)
        points = trace_curve(mask, bounds, axis_range, n_points, spec.monotonic)
        results.append({"name": spec.name, "data": points})

    return results


# A set of distinct colors for overlay lines (tab20 palette).
_OVERLAY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def generate_overlay(
    image_bytes: bytes,
    data_series: list[dict],
    axis_range: AxisRange,
) -> bytes:
    """Render extracted curves overlaid on the original chart image.

    Args:
        image_bytes: Original chart image bytes.
        data_series: List of {"name": str, "data": [{"x", "y"}, ...]} dicts.
        axis_range: Data-coordinate axis ranges.

    Returns:
        PNG image bytes of the overlay figure.
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    bounds = detect_plot_bounds(img)
    if bounds is None:
        raise ValueError("Could not detect plot bounds for overlay.")

    h, w = img.shape[:2]
    x_span = bounds.right - bounds.left
    y_span = bounds.bottom - bounds.top
    dx = axis_range.x_max - axis_range.x_min
    dy = axis_range.y_max - axis_range.y_min

    img_x_left = axis_range.x_min - (bounds.left / x_span) * dx
    img_x_right = axis_range.x_min + ((w - bounds.left) / x_span) * dx
    img_y_top = axis_range.y_min + ((bounds.bottom) / y_span) * dy
    img_y_bottom = axis_range.y_min + ((bounds.bottom - h) / y_span) * dy

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(
        img / 255.0,
        extent=[img_x_left, img_x_right, img_y_bottom, img_y_top],
        aspect="auto", alpha=0.45, zorder=0,
    )

    for i, series in enumerate(data_series):
        pts = series.get("data", [])
        if not pts:
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        ax.plot(xs, ys, color=color, linewidth=1.8,
                label=f"{series['name']} ({len(pts)} pts)", zorder=2)

    ax.set_xlim(axis_range.x_min, axis_range.x_max)
    ax.set_ylim(axis_range.y_min, axis_range.y_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Pixel-Enhanced Extraction Overlay")
    n_series = sum(1 for s in data_series if s.get("data"))
    ncol = 2 if n_series > 8 else 1
    ax.legend(loc="best", fontsize=7, ncol=ncol)
    ax.grid(True, alpha=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
