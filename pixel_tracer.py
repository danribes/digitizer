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
from scipy.optimize import linear_sum_assignment


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
    line_style: str | None = None  # "solid", "dashed", "dotted", or None


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


def _find_best_h_run(
    dark: np.ndarray, row_min: int, row_max: int, min_run: int,
) -> dict | None:
    """Find the longest horizontal dark run within the given row range."""
    _, w = dark.shape
    best: dict | None = None
    for row in range(row_min, row_max + 1):
        run_start = None
        for col in range(w):
            if dark[row, col]:
                if run_start is None:
                    run_start = col
            else:
                if run_start is not None:
                    run_len = col - run_start
                    if run_len >= min_run and (best is None or run_len > best["len"]):
                        best = {"row": row, "len": run_len, "start": run_start, "end": col - 1}
                    run_start = None
        if run_start is not None:
            run_len = w - run_start
            if run_len >= min_run and (best is None or run_len > best["len"]):
                best = {"row": row, "len": run_len, "start": run_start, "end": w - 1}
    return best


def _find_best_v_run(
    dark: np.ndarray, col_min: int, col_max: int, min_run: int,
) -> dict | None:
    """Find the longest vertical dark run within the given column range."""
    h, _ = dark.shape
    best: dict | None = None
    for col in range(col_min, col_max + 1):
        run_start = None
        for row in range(h):
            if dark[row, col]:
                if run_start is None:
                    run_start = row
            else:
                if run_start is not None:
                    run_len = row - run_start
                    if run_len >= min_run and (best is None or run_len > best["len"]):
                        best = {"col": col, "len": run_len, "start": run_start, "end": row - 1}
                    run_start = None
        if run_start is not None:
            run_len = h - run_start
            if run_len >= min_run and (best is None or run_len > best["len"]):
                best = {"col": col, "len": run_len, "start": run_start, "end": h - 1}
    return best


def detect_plot_bounds(img: np.ndarray, min_run: int = 150) -> PlotBounds | None:
    """Detect plot area by finding long dark horizontal/vertical pixel runs.

    Tries progressively lighter thresholds to handle gray axis lines.
    Skips edge rows/columns to avoid matching image borders.
    The x-axis is searched in the lower 2/3 of the image; the y-axis in
    the left 2/3.  If a y-axis cannot be found, falls back to using the
    x-axis start column as the left bound.

    Args:
        img: RGB image array (H, W, 3), uint8.
        min_run: Minimum pixel run length to consider an axis line.

    Returns:
        PlotBounds or None if detection fails.
    """
    h, w = img.shape[:2]
    gray = img.mean(axis=2)

    # Margin to skip: ignore the outermost pixels (image borders)
    edge = max(3, min(h, w) // 50)

    best_h: dict | None = None
    best_v: dict | None = None

    # Try progressively lighter thresholds
    for threshold in (80, 128, 160):
        dark = gray < threshold

        # x-axis: search lower 2/3, skip edge rows
        if best_h is None:
            bh = _find_best_h_run(dark, max(edge, h // 3), h - edge - 1, min_run)
            if bh is not None:
                best_h = bh

        # y-axis: search left 2/3, skip edge columns
        if best_v is None:
            bv = _find_best_v_run(dark, edge, w * 2 // 3, min_run)
            if bv is not None:
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
        for row in range(best_h["row"] - 1, edge, -1):
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


def _cluster_positions(positions: list[int], min_gap: int = 8) -> list[int]:
    """Cluster nearby pixel positions and return their centers."""
    if not positions:
        return []
    positions = sorted(positions)
    clusters: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]


def detect_axis_ticks(
    img: np.ndarray, bounds: PlotBounds,
) -> tuple[list[int], list[int]]:
    """Detect tick mark positions on x-axis and y-axis using CV.

    X-axis ticks: short vertical dark marks just below the x-axis line.
    Y-axis ticks: short horizontal dark marks just left of the y-axis line.

    Returns:
        (x_tick_cols, y_tick_rows) — sorted lists of pixel positions.
    """
    gray = img.mean(axis=2)
    h, w = gray.shape

    # --- X-axis ticks: vertical marks below x-axis ---
    x_cands: list[int] = []
    tick_top = bounds.bottom + 1
    tick_bot = min(h - 1, bounds.bottom + 6)
    for col in range(bounds.left, min(w, bounds.right + 5)):
        dark_count = sum(1 for r in range(tick_top, tick_bot + 1)
                         if r < h and gray[r, col] < 140)
        if dark_count >= 2:
            x_cands.append(col)
    x_tick_cols = _cluster_positions(x_cands, min_gap=8)
    # Exclude the y-axis position itself
    x_tick_cols = [c for c in x_tick_cols if abs(c - bounds.left) > 3]

    # --- Y-axis ticks: horizontal marks left of y-axis ---
    y_cands: list[int] = []
    tick_left = max(0, bounds.left - 6)
    tick_right = bounds.left - 1
    for row in range(bounds.top, bounds.bottom + 1):
        dark_count = sum(1 for c in range(tick_left, tick_right + 1)
                         if c >= 0 and gray[row, c] < 140)
        if dark_count >= 2:
            y_cands.append(row)
    y_tick_rows = _cluster_positions(y_cands, min_gap=8)

    return x_tick_cols, y_tick_rows


def calibrate_axes(
    img: np.ndarray,
    bounds: PlotBounds,
    axis_range: AxisRange,
) -> dict:
    """Calibrate axes using CV tick detection and assess alignment quality.

    Detects tick marks, verifies their spacing is regular (linear axis),
    and checks that the bounds correctly represent the axis range.

    Returns dict with:
        x_tick_cols, y_tick_rows: detected tick positions
        alignment_quality: overall quality assessment string
        metrics: detailed alignment metrics
    """
    x_tick_cols, y_tick_rows = detect_axis_ticks(img, bounds)

    metrics: dict = {
        "x_tick_count": len(x_tick_cols),
        "y_tick_count": len(y_tick_rows),
    }

    # --- X-axis tick regularity ---
    x_regularity = 1.0
    if len(x_tick_cols) >= 2:
        intervals = np.diff(x_tick_cols).astype(float)
        mean_int = np.mean(intervals)
        rmse = float(np.sqrt(np.mean((intervals - mean_int) ** 2)))
        x_regularity = max(0.0, 1.0 - rmse / mean_int) if mean_int > 0 else 0
        metrics["x_tick_interval_mean_px"] = round(float(mean_int), 1)
        metrics["x_tick_interval_rmse_px"] = round(rmse, 2)
        metrics["x_tick_regularity"] = round(x_regularity, 4)

        # Check first tick aligns with bounds.left (x_min)
        first_offset = abs(x_tick_cols[0] - bounds.left)
        metrics["x_first_tick_offset_px"] = first_offset

    # --- Y-axis tick regularity ---
    y_regularity = 1.0
    if len(y_tick_rows) >= 2:
        intervals = np.diff(y_tick_rows).astype(float)
        mean_int = np.mean(intervals)
        rmse = float(np.sqrt(np.mean((intervals - mean_int) ** 2)))
        y_regularity = max(0.0, 1.0 - rmse / mean_int) if mean_int > 0 else 0
        metrics["y_tick_interval_mean_px"] = round(float(mean_int), 1)
        metrics["y_tick_interval_rmse_px"] = round(rmse, 2)
        metrics["y_tick_regularity"] = round(y_regularity, 4)

        first_offset = abs(y_tick_rows[0] - bounds.top)
        metrics["y_first_tick_offset_px"] = first_offset

    # --- Alignment quality verdict ---
    if x_regularity > 0.95 and y_regularity > 0.95:
        quality = "good"
    elif x_regularity > 0.85 and y_regularity > 0.85:
        quality = "acceptable"
    else:
        quality = "poor"

    return {
        "x_tick_cols": x_tick_cols,
        "y_tick_rows": y_tick_rows,
        "alignment_quality": quality,
        "metrics": metrics,
    }


def assess_extraction_accuracy(
    img: np.ndarray,
    data_series: list[dict],
    bounds: PlotBounds,
    axis_range: AxisRange,
    search_radius: int = 30,
) -> dict:
    """CV-based extraction accuracy assessment.

    For each extracted data point, finds the nearest dark curve pixel
    vertically and measures the discrepancy.

    Returns dict with per-series metrics, overall metrics, and
    a feedback_text string describing specific errors for AI correction.
    """
    gray = img.mean(axis=2)

    # Build dark-pixel mask for curve pixels only
    dark = gray < 160
    brightness = img[:, :, 0].astype(int) + img[:, :, 1].astype(int) + img[:, :, 2].astype(int)
    dark = dark & (brightness < 680)

    # Exclude axis lines
    axis_margin = 4
    for r in range(max(0, bounds.bottom - axis_margin),
                   min(img.shape[0], bounds.bottom + axis_margin + 1)):
        dark[r, :] = False
    for c in range(max(0, bounds.left - axis_margin),
                   min(img.shape[1], bounds.left + axis_margin + 1)):
        dark[:, c] = False

    y_full_scale = axis_range.y_max - axis_range.y_min

    series_results = []
    all_errors: list[float] = []

    for series in data_series:
        name = series.get("name", "?")
        pts = series.get("data", [])
        errors: list[float] = []
        point_details: list[dict] = []
        missed = 0

        for pt in pts:
            if "x" not in pt or "y" not in pt:
                continue
            x, y = float(pt["x"]), float(pt["y"])
            col, row_expected = _data_to_pixel(x, y, bounds, axis_range)
            if col <= bounds.left + 5 or col >= bounds.right - 2:
                continue

            # Find nearest dark pixel vertically
            col_lo = max(bounds.left, col - 2)
            col_hi = min(bounds.right, col + 2)
            actual_row = None
            for dist in range(search_radius + 1):
                for dr in ([0] if dist == 0 else [-dist, dist]):
                    r = row_expected + dr
                    if bounds.top <= r <= bounds.bottom and dark[r, col_lo:col_hi + 1].any():
                        actual_row = r
                        break
                if actual_row is not None:
                    break

            if actual_row is None:
                missed += 1
                continue

            _, actual_y = pixel_to_data(col, actual_row, bounds, axis_range)
            error = y - actual_y
            errors.append(error)
            all_errors.append(error)
            point_details.append({"x": x, "y_extracted": y, "y_actual": round(actual_y, 4), "error": round(error, 4)})

        err_arr = np.array(errors) if errors else np.array([0.0])

        # Identify systematic error regions
        region_errors: list[str] = []
        if len(point_details) >= 5:
            # Split into 5 regions along x-axis
            xs = [p["x"] for p in point_details]
            x_lo, x_hi = min(xs), max(xs)
            n_regions = 5
            for ri in range(n_regions):
                rx_lo = x_lo + ri * (x_hi - x_lo) / n_regions
                rx_hi = x_lo + (ri + 1) * (x_hi - x_lo) / n_regions
                region_pts = [p for p in point_details if rx_lo <= p["x"] < rx_hi]
                if len(region_pts) >= 2:
                    region_bias = np.mean([p["error"] for p in region_pts])
                    if abs(region_bias) > 0.015 * y_full_scale:
                        direction = "too high" if region_bias > 0 else "too low"
                        region_errors.append(
                            f"x={rx_lo:.1f}-{rx_hi:.1f}: extracted is {abs(region_bias):.3f} {direction}"
                        )

        sr = {
            "name": name,
            "n_points": len(errors),
            "missed": missed,
            "mae": round(float(np.mean(np.abs(err_arr))), 4),
            "rmse": round(float(np.sqrt(np.mean(err_arr ** 2))), 4),
            "bias": round(float(np.mean(err_arr)), 4),
            "max_error": round(float(np.max(np.abs(err_arr))), 4),
            "within_3pct": round(float(np.mean(np.abs(err_arr) <= 0.03 * y_full_scale)), 4),
            "region_errors": region_errors,
        }
        series_results.append(sr)

    all_err = np.array(all_errors) if all_errors else np.array([0.0])
    overall_mae = float(np.mean(np.abs(all_err)))
    overall_within_3 = float(np.mean(np.abs(all_err) <= 0.03 * y_full_scale))

    # Generate targeted feedback text for AI correction
    feedback_lines = []
    for sr in series_results:
        if sr["mae"] > 0.02 * y_full_scale or sr["within_3pct"] < 0.85:
            feedback_lines.append(f"Series '{sr['name']}' has MAE={sr['mae']:.4f} (bias={sr['bias']:+.4f}).")
            for re in sr["region_errors"]:
                feedback_lines.append(f"  {re}")

    passed = overall_mae <= 0.02 * y_full_scale and overall_within_3 >= 0.90

    return {
        "series": series_results,
        "overall_mae": round(overall_mae, 4),
        "overall_within_3pct": round(overall_within_3, 4),
        "passed": passed,
        "feedback_text": "\n".join(feedback_lines) if feedback_lines else "",
    }


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
    img: np.ndarray,
    target_rgb: tuple[int, int, int],
    tolerance: int = 50,
    bounds: PlotBounds | None = None,
) -> np.ndarray:
    """Create boolean mask for pixels matching the target color.

    For gray targets (low channel spread), adds a spread constraint to avoid
    matching colored pixels. Excludes near-white (background) and near-black
    (axes/gridlines) pixels.

    When the target is near-black (max channel < 60), skips the not_black
    brightness filter (since the curves ARE black) and instead excludes
    axis-line pixels by position when *bounds* is provided.

    Args:
        img: RGB image array (H, W, 3), uint8.
        target_rgb: Target (R, G, B) color.
        tolerance: Per-channel tolerance for matching.
        bounds: Optional plot bounds used for axis-line exclusion on
            near-black targets.

    Returns:
        Boolean array (H, W).
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    tr, tg, tb = target_rgb

    near_black_target = max(target_rgb) < 60

    # Per-channel match
    in_range = (
        (r >= tr - tolerance) & (r <= tr + tolerance)
        & (g >= tg - tolerance) & (g <= tg + tolerance)
        & (b >= tb - tolerance) & (b <= tb + tolerance)
    )

    # Exclude near-white (background)
    brightness = r.astype(int) + g.astype(int) + b.astype(int)
    not_white = brightness < 700  # ~233 per channel

    if near_black_target:
        # Target is black — don't exclude dark pixels; instead mask axis lines
        mask = in_range & not_white
        if bounds is not None:
            axis_margin = 3
            # Mask out rows near the x-axis (bottom)
            for row in range(max(0, bounds.bottom - axis_margin),
                             min(img.shape[0], bounds.bottom + axis_margin + 1)):
                mask[row, :] = False
            # Mask out columns near the y-axis (left)
            for col in range(max(0, bounds.left - axis_margin),
                             min(img.shape[1], bounds.left + axis_margin + 1)):
                mask[:, col] = False
    else:
        not_black = brightness > 100  # ~33 per channel
        mask = in_range & not_white & not_black

    # For gray targets, add channel spread constraint
    channel_spread = int(max(target_rgb)) - int(min(target_rgb))
    if channel_spread < 30:
        rgb_stack = np.stack([r, g, b], axis=2).astype(np.int16)
        pixel_spread = rgb_stack.max(axis=2) - rgb_stack.min(axis=2)
        mask = mask & (pixel_spread < 30)

    return mask


def classify_line_style(
    mask: np.ndarray, bounds: PlotBounds
) -> dict[str, np.ndarray]:
    """Classify dark pixels by line style based on horizontal run length.

    For each row within the plot bounds, computes contiguous horizontal runs
    of True pixels and labels every pixel by the length of its run.

    Thresholds scale with plot width so classification works across image
    resolutions (e.g. 600px-wide and 1200px-wide charts).

    Returns:
        Dict with "solid", "dashed", "dotted" boolean masks.
    """
    # Scale thresholds to plot width (reference: 1200px → 20/6)
    pw = bounds.width
    solid_thresh = max(8, pw // 60)   # ~20 at 1200px, ~10 at 600px, ~6 at 360px
    dash_thresh = max(3, pw // 200)   # ~6 at 1200px,  ~3 at 600px

    solid = np.zeros_like(mask)
    dashed = np.zeros_like(mask)
    dotted = np.zeros_like(mask)

    for row in range(bounds.top, bounds.bottom + 1):
        cols = np.where(mask[row, bounds.left:bounds.right + 1])[0] + bounds.left
        if len(cols) == 0:
            continue
        # Split into contiguous runs
        diffs = np.diff(cols)
        splits = np.where(diffs > 1)[0] + 1
        runs = np.split(cols, splits)
        for run in runs:
            run_len = len(run)
            if run_len >= solid_thresh:
                solid[row, run] = True
            elif run_len >= dash_thresh:
                dashed[row, run] = True
            else:
                dotted[row, run] = True

    return {"solid": solid, "dashed": dashed, "dotted": dotted}


def _find_band_segments(
    mask: np.ndarray, col: int, row_min: int, row_max: int,
    band_w: int = 5, seg_gap: int = 6,
) -> list[float]:
    """Find segment centers using a column band ±band_w around *col*.

    Aggregates mask pixels across 2*band_w+1 columns (using logical OR per
    row) to handle dashed/dotted line gaps, then splits into contiguous
    vertical segments separated by *seg_gap* rows.

    Returns list of row-center values for each segment, sorted top-to-bottom.
    """
    c_lo = max(0, col - band_w)
    c_hi = min(mask.shape[1] - 1, col + band_w)
    band = mask[row_min:row_max + 1, c_lo:c_hi + 1].any(axis=1)
    rows = np.where(band)[0] + row_min
    if len(rows) == 0:
        return []
    diffs = np.diff(rows)
    splits = np.where(diffs > seg_gap)[0] + 1
    segments = np.split(rows, splits)
    return sorted(float(s.mean()) for s in segments if len(s) >= 1)


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


def trace_multi_curves(
    mask: np.ndarray,
    bounds: PlotBounds,
    axis_range: AxisRange,
    n_curves: int,
    n_points: int = 250,
    monotonic: str | None = None,
) -> list[list[dict]]:
    """Trace N curves simultaneously from a single color mask.

    Uses banded column segment detection and bidirectional tracking with
    Hungarian-algorithm assignment for robust separation of same-color curves
    (e.g. solid, dashed, dotted lines).

    Algorithm:
    1. Find the column with the best N-way segment separation (init point).
    2. Track curves forward (right) and backward (left) from the init point.
    3. At each column, aggregate mask pixels across a ±band_w band to handle
       dashed/dotted gaps, then assign segments to trackers via the Hungarian
       algorithm (minimizing total row-distance).
    4. Interpolate across remaining gaps, apply median filter and monotonicity
       enforcement per curve.

    Args:
        mask: Boolean mask (H, W) of matching pixels.
        bounds: Pixel boundaries of the plot area.
        axis_range: Data-coordinate ranges.
        n_curves: Number of curves to trace.
        n_points: Number of output points per curve after downsampling.
        monotonic: "decreasing", "increasing", or None (applied to all curves).

    Returns:
        List of N lists of {"x": float, "y": float} dicts.
    """
    row_min = bounds.top + 5
    row_max = bounds.bottom - 10
    if row_min >= row_max:
        row_min = bounds.top
        row_max = bounds.bottom

    band_w = max(3, bounds.width // 70)  # ~5 for 360px, ~9 for 600px
    seg_gap = max(3, bounds.height // 75)  # ~4 for 305px — tight enough to separate
    match_tol = bounds.height * 0.20

    def find_segs(col: int) -> list[float]:
        return _find_band_segments(mask, col, row_min, row_max, band_w, seg_gap)

    # --- Step 1: find the best initialization column ---
    best_col: int | None = None
    best_sep = 0.0
    for col in range(bounds.left, bounds.right + 1):
        centers = find_segs(col)
        if len(centers) >= n_curves:
            top_n = sorted(centers)[:n_curves]
            min_sep = min(top_n[i + 1] - top_n[i] for i in range(n_curves - 1))
            if min_sep > best_sep:
                best_sep = min_sep
                best_col = col

    if best_col is None:
        # Could not find any column with n_curves segments — fall back
        return [[] for _ in range(n_curves)]

    init_centers = sorted(find_segs(best_col))[:n_curves]

    # --- Step 2: bidirectional tracking ---
    def _track(start_col: int, end_col: int, step: int,
               init_rows: list[float]) -> list[dict[int, float]]:
        curve_data: list[dict[int, float]] = [{} for _ in range(n_curves)]
        prev_rows = list(init_rows)
        for i, r in enumerate(prev_rows):
            curve_data[i][start_col] = r

        col = start_col + step
        while (step > 0 and col <= end_col) or (step < 0 and col >= end_col):
            centers = find_segs(col)
            if centers:
                n_seg = len(centers)
                # Build cost matrix: trackers × segments
                cost = np.full((n_curves, n_seg), 1e9)
                for t in range(n_curves):
                    for s in range(n_seg):
                        cost[t, s] = abs(centers[s] - prev_rows[t])

                new_rows = [None] * n_curves
                if n_seg >= n_curves:
                    row_ind, col_ind = linear_sum_assignment(cost)
                    for t, s in zip(row_ind, col_ind):
                        if cost[t, s] < match_tol:
                            new_rows[t] = centers[s]
                else:
                    col_ind, row_ind = linear_sum_assignment(cost.T)
                    for s, t in zip(col_ind, row_ind):
                        if cost[t, s] < match_tol:
                            new_rows[t] = centers[s]

                # Enforce ordering: tracker 0 row ≤ tracker 1 row ≤ ...
                # (i.e. curve 0 stays above curve 1 stays above curve 2)
                valid = True
                filled = [(t, new_rows[t]) for t in range(n_curves)
                          if new_rows[t] is not None]
                for j in range(len(filled) - 1):
                    t1, r1 = filled[j]
                    t2, r2 = filled[j + 1]
                    if t1 < t2 and r1 > r2:
                        valid = False
                        break

                if valid:
                    for t in range(n_curves):
                        if new_rows[t] is not None:
                            curve_data[t][col] = new_rows[t]
                            prev_rows[t] = new_rows[t]

            col += step
        return curve_data

    fwd = _track(best_col, bounds.right, 1, init_centers)
    bwd = _track(best_col, bounds.left, -1, init_centers)

    # --- Step 3: merge, interpolate, smooth, downsample ---
    results: list[list[dict]] = []
    for i in range(n_curves):
        merged = {**bwd[i], **fwd[i]}
        if not merged:
            results.append([])
            continue

        known_cols = sorted(merged.keys())
        known_rows = np.array([merged[c] for c in known_cols])

        full_cols = np.arange(known_cols[0], known_cols[-1] + 1)
        interp_rows = np.interp(full_cols, known_cols, known_rows)

        raw_x = np.array([pixel_to_data(c, 0, bounds, axis_range)[0]
                          for c in full_cols])
        raw_y = np.array([pixel_to_data(0, r, bounds, axis_range)[1]
                          for r in interp_rows])

        if len(raw_y) >= 7:
            smooth_y = median_filter(raw_y, size=7)
        else:
            smooth_y = raw_y.copy()

        if monotonic == "decreasing":
            smooth_y = np.minimum.accumulate(smooth_y)
        elif monotonic == "increasing":
            smooth_y = np.maximum.accumulate(smooth_y)

        if len(raw_x) > n_points:
            indices = np.linspace(0, len(raw_x) - 1, n_points, dtype=int)
            raw_x = raw_x[indices]
            smooth_y = smooth_y[indices]

        results.append([
            {"x": round(float(x), 4), "y": round(float(y), 4)}
            for x, y in zip(raw_x, smooth_y)
        ])

    return results


def _rgb_similar(a: tuple[int, int, int], b: tuple[int, int, int],
                  threshold: int = 40) -> bool:
    """Check if two RGB colors are similar (per-channel distance)."""
    return all(abs(ca - cb) <= threshold for ca, cb in zip(a, b))


def _group_specs_by_color(
    specs: list[SeriesSpec],
) -> list[list[int]]:
    """Group series spec indices by similar RGB color.

    Returns list of groups, each a list of indices into *specs*.
    """
    groups: list[list[int]] = []
    assigned: set[int] = set()
    for i, spec_i in enumerate(specs):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, len(specs)):
            if j in assigned:
                continue
            if _rgb_similar(spec_i.rgb, specs[j].rgb):
                group.append(j)
                assigned.add(j)
        groups.append(group)
    return groups


def _assign_curves_by_line_style(
    mask: np.ndarray,
    bounds: PlotBounds,
    curves: list[list[dict]],
    specs: list[SeriesSpec],
    axis_range: AxisRange,
) -> list[list[dict]]:
    """Re-order traced curves to match specs using line-style classification.

    If specs have line_style hints, classify the mask by run length and measure
    how much each curve overlaps with each style sub-mask. Only reassigns when
    the classification is clearly confident (the best-matching style for each
    curve is unambiguous). Falls back to vertical-position ordering otherwise.
    """
    style_hints = [s.line_style for s in specs]
    if all(h is None or h == "unknown" for h in style_hints):
        return curves  # No hints — keep positional order

    style_masks = classify_line_style(mask, bounds)

    # For each curve, compute overlap with each style sub-mask
    curve_style_scores: list[dict[str, float]] = []
    for curve_pts in curves:
        scores = {"solid": 0.0, "dashed": 0.0, "dotted": 0.0}
        if not curve_pts:
            curve_style_scores.append(scores)
            continue
        sample = curve_pts[::max(1, len(curve_pts) // 50)]
        for pt in sample:
            x_frac = (pt["x"] - axis_range.x_min) / (axis_range.x_max - axis_range.x_min)
            y_frac = (pt["y"] - axis_range.y_min) / (axis_range.y_max - axis_range.y_min)
            col = int(bounds.left + x_frac * (bounds.right - bounds.left))
            row = int(bounds.bottom - y_frac * (bounds.bottom - bounds.top))
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = row + dr, col + dc
                    if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                        for style_name, smask in style_masks.items():
                            if smask[r, c]:
                                scores[style_name] += 1
        curve_style_scores.append(scores)

    # Check if classification is confident: each curve's dominant style must
    # be unique across curves AND the dominant style must clearly dominate.
    dominant_styles = []
    for scores in curve_style_scores:
        total = sum(scores.values())
        if total == 0:
            dominant_styles.append(None)
            continue
        sorted_styles = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_name, best_val = sorted_styles[0]
        second_val = sorted_styles[1][1] if len(sorted_styles) > 1 else 0
        # Dominant style must have ≥1.5× the runner-up score
        if best_val > second_val * 1.5:
            dominant_styles.append(best_name)
        else:
            dominant_styles.append(None)

    # Only reassign if ALL curves have a clear, unique dominant style
    non_none = [d for d in dominant_styles if d is not None]
    if len(non_none) != len(curves) or len(set(non_none)) != len(non_none):
        return curves  # Ambiguous — keep positional order

    # All curves have unique dominant styles — do the assignment
    assigned_curves: list[list[dict] | None] = [None] * len(specs)
    used: set[int] = set()
    for spec_idx, spec in enumerate(specs):
        hint = spec.line_style
        if hint is None or hint == "unknown":
            continue
        for curve_idx in range(len(curves)):
            if curve_idx in used:
                continue
            if dominant_styles[curve_idx] == hint:
                assigned_curves[spec_idx] = curves[curve_idx]
                used.add(curve_idx)
                break

    remaining = [i for i in range(len(curves)) if i not in used]
    unfilled = [i for i in range(len(specs)) if assigned_curves[i] is None]
    for spec_idx, curve_idx in zip(unfilled, remaining):
        assigned_curves[spec_idx] = curves[curve_idx]

    return [c if c is not None else [] for c in assigned_curves]


def extract_curves_from_image(
    image_bytes: bytes,
    series_specs: list[SeriesSpec],
    axis_range: AxisRange,
    n_points: int = 250,
) -> list[dict]:
    """Extract curves from a chart image using pixel-level color detection.

    Top-level entry point. Auto-detects plot bounds, builds color masks per
    series, traces each curve, and returns data in the standard format.

    When multiple series share the same color, switches to multi-curve tracing
    using trace_multi_curves and optionally uses line-style classification to
    assign curves to the correct series.

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

    # Group specs by similar color
    color_groups = _group_specs_by_color(series_specs)

    # Pre-allocate results in original spec order
    results: list[dict | None] = [None] * len(series_specs)

    for group in color_groups:
        group_specs = [series_specs[i] for i in group]
        representative = group_specs[0]

        if len(group) == 1:
            # Single-color series: use existing trace_curve (no regression)
            mask = make_color_mask(img, representative.rgb, representative.tolerance,
                                   bounds=bounds)
            points = trace_curve(mask, bounds, axis_range, n_points,
                                 representative.monotonic)
            results[group[0]] = {"name": representative.name, "data": points}
        else:
            # Multiple same-color series: use multi-curve tracing
            mask = make_color_mask(img, representative.rgb, representative.tolerance,
                                   bounds=bounds)
            mono = representative.monotonic  # assume same for the group
            curves = trace_multi_curves(mask, bounds, axis_range,
                                        n_curves=len(group),
                                        n_points=n_points,
                                        monotonic=mono)

            # Assign curves to specs using line-style hints if available
            curves = _assign_curves_by_line_style(
                mask, bounds, curves, group_specs, axis_range,
            )

            for idx_in_group, spec_idx in enumerate(group):
                results[spec_idx] = {
                    "name": series_specs[spec_idx].name,
                    "data": curves[idx_in_group] if idx_in_group < len(curves) else [],
                }

    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Pixel-snap correction: fix AI-extracted y-values using actual curve pixels
# ---------------------------------------------------------------------------

def _data_to_pixel(
    x: float, y: float, bounds: PlotBounds, axis_range: AxisRange
) -> tuple[int, int]:
    """Convert data coordinates to pixel coordinates (col, row)."""
    x_frac = (x - axis_range.x_min) / (axis_range.x_max - axis_range.x_min)
    y_frac = (y - axis_range.y_min) / (axis_range.y_max - axis_range.y_min)
    col = int(bounds.left + x_frac * (bounds.right - bounds.left))
    row = int(bounds.bottom - y_frac * (bounds.bottom - bounds.top))
    return (
        max(bounds.left, min(bounds.right, col)),
        max(bounds.top, min(bounds.bottom, row)),
    )


def snap_series_to_pixels(
    image_bytes: bytes,
    data_series: list[dict],
    axis_range: AxisRange,
    search_radius: int = 25,
) -> list[dict]:
    """Snap AI-extracted data points to actual curve pixels in the original image.

    For each data point, finds the nearest dark pixel vertically in the
    original image and corrects the y-value.  This combines the AI's ability
    to identify and separate curves with pixel-level measurement precision.

    When multiple series are present, points are snapped in order from the
    series with the highest y-values to lowest, and already-claimed pixels
    are avoided so nearby curves don't collapse onto each other.

    Args:
        image_bytes: Original chart image bytes.
        data_series: AI-extracted [{"name", "data": [{"x", "y"}, ...]}, ...].
        axis_range: Data-coordinate axis ranges.
        search_radius: Max pixel distance to search vertically.

    Returns:
        Corrected data_series (same structure, more accurate y-values).
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    bounds = detect_plot_bounds(img)
    if bounds is None or bounds.height < 20:
        return data_series

    gray = img.mean(axis=2)

    # Build dark-pixel mask: curve pixels but not background, and exclude
    # axis lines by position.
    dark = gray < 160
    # Exclude near-white
    brightness = img[:, :, 0].astype(int) + img[:, :, 1].astype(int) + img[:, :, 2].astype(int)
    dark = dark & (brightness < 680)
    # Exclude axis lines by position
    axis_margin = 4
    for r in range(max(0, bounds.bottom - axis_margin),
                   min(img.shape[0], bounds.bottom + axis_margin + 1)):
        dark[r, :] = False
    for c in range(max(0, bounds.left - axis_margin),
                   min(img.shape[1], bounds.left + axis_margin + 1)):
        dark[:, c] = False

    # For each column, find all dark-pixel segments (contiguous vertical runs)
    # so we can snap to segment centers rather than individual pixels.
    def _segments_at_col(col: int) -> list[float]:
        """Return sorted list of row-centers of dark segments at this column."""
        col_lo = max(bounds.left, col - 2)
        col_hi = min(bounds.right, col + 2)
        band = dark[bounds.top:bounds.bottom + 1, col_lo:col_hi + 1].any(axis=1)
        rows = np.where(band)[0] + bounds.top
        if len(rows) == 0:
            return []
        diffs = np.diff(rows)
        splits = np.where(diffs > 3)[0] + 1
        segments = np.split(rows, splits)
        return sorted(float(s.mean()) for s in segments if len(s) >= 1)

    # Pre-compute segments for all columns in the plot area
    col_segments: dict[int, list[float]] = {}
    for col in range(bounds.left, bounds.right + 1):
        segs = _segments_at_col(col)
        if segs:
            col_segments[col] = segs

    # Process series: snap each point to the nearest unclaimed segment
    # Process from top curve to bottom so claimed-row tracking works.
    series_order = list(range(len(data_series)))

    # Track which segment rows are "claimed" at each column to prevent
    # multiple series snapping to the same curve.
    claimed: dict[int, set[int]] = {}  # col -> set of segment indices

    corrected_series: list[dict] = [None] * len(data_series)  # type: ignore
    for si in series_order:
        series = data_series[si]
        corrected_points = []
        for pt in series.get("data", []):
            if "x" not in pt or "y" not in pt:
                corrected_points.append(pt)
                continue

            x, y = float(pt["x"]), float(pt["y"])
            col, row = _data_to_pixel(x, y, bounds, axis_range)

            # Find the nearest unclaimed segment
            segs = col_segments.get(col, [])
            col_claimed = claimed.get(col, set())

            best_row: float | None = None
            best_seg_idx: int | None = None
            best_dist = search_radius + 1
            for idx, seg_row in enumerate(segs):
                if idx in col_claimed:
                    continue
                dist = abs(seg_row - row)
                if dist < best_dist:
                    best_dist = dist
                    best_row = seg_row
                    best_seg_idx = idx

            if best_row is not None and best_seg_idx is not None:
                _, corrected_y = pixel_to_data(col, best_row, bounds, axis_range)
                corrected_points.append({"x": round(x, 4), "y": round(corrected_y, 4)})
                # Claim this segment
                if col not in claimed:
                    claimed[col] = set()
                claimed[col].add(best_seg_idx)
            else:
                corrected_points.append(pt)

        # Apply median filter to smooth out noise from snapping
        if len(corrected_points) >= 7:
            ys = np.array([p["y"] for p in corrected_points])
            smooth = median_filter(ys, size=5)
            for i, p in enumerate(corrected_points):
                p["y"] = round(float(smooth[i]), 4)

        corrected_series[si] = {"name": series["name"], "data": corrected_points}

    return corrected_series


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

    h, w = img.shape[:2]

    # Compute image extent in data coordinates so the background aligns
    # with the extracted curves.  When plot-bounds detection succeeds and
    # gives a sensible region we map pixel→data precisely; otherwise we
    # fall back to filling the axis range with the whole image.
    if (bounds is not None
            and bounds.width > 20 and bounds.height > 20
            and bounds.bottom > bounds.top):
        x_span = bounds.right - bounds.left
        y_span = bounds.bottom - bounds.top
        dx = axis_range.x_max - axis_range.x_min
        dy = axis_range.y_max - axis_range.y_min

        img_x_left = axis_range.x_min - (bounds.left / x_span) * dx
        img_x_right = axis_range.x_min + ((w - bounds.left) / x_span) * dx
        img_y_top = axis_range.y_min + ((bounds.bottom) / y_span) * dy
        img_y_bottom = axis_range.y_min + ((bounds.bottom - h) / y_span) * dy
    else:
        # Fallback: stretch the image across the full axis range with a
        # small margin so labels/titles outside the plot area are visible.
        dx = axis_range.x_max - axis_range.x_min
        dy = axis_range.y_max - axis_range.y_min
        margin_x = dx * 0.12
        margin_y = dy * 0.12
        img_x_left = axis_range.x_min - margin_x
        img_x_right = axis_range.x_max + margin_x
        img_y_bottom = axis_range.y_min - margin_y
        img_y_top = axis_range.y_max + margin_y

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(
        img / 255.0,
        extent=[img_x_left, img_x_right, img_y_bottom, img_y_top],
        aspect="auto", alpha=0.75, zorder=0,
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
