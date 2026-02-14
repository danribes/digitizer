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

    @staticmethod
    def from_calibration_points(
        pixel_1: tuple[int, int],  # (col, row) for point 1
        data_1: tuple[float, float],  # (x, y) data coords for point 1
        pixel_2: tuple[int, int],  # (col, row) for point 2
        data_2: tuple[float, float],  # (x, y) data coords for point 2
    ) -> tuple["PlotBounds", "AxisRange"]:
        """Derive PlotBounds and AxisRange from two user-clicked calibration points.

        The two points are treated as opposite corners of the plot area:
        Point 1 = top-left corner (x_min, y_max), Point 2 = bottom-right (x_max, y_min).
        """
        bounds = PlotBounds(
            left=min(pixel_1[0], pixel_2[0]),
            right=max(pixel_1[0], pixel_2[0]),
            top=min(pixel_1[1], pixel_2[1]),
            bottom=max(pixel_1[1], pixel_2[1]),
        )
        axis_range = AxisRange(
            x_min=min(data_1[0], data_2[0]),
            x_max=max(data_1[0], data_2[0]),
            y_min=min(data_1[1], data_2[1]),
            y_max=max(data_1[1], data_2[1]),
        )
        return bounds, axis_range


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


def _refine_bounds_from_ticks(
    bounds: PlotBounds,
    x_tick_cols: list[int],
    y_tick_rows: list[int],
) -> PlotBounds:
    """Refine plot bounds using detected tick positions.

    Tick marks provide precise calibration points.  The base tick spacing
    is derived from the minimum adjacent-tick gap (handling missing ticks
    where a gap is ~2× the base).  The bounds are then snapped so that the
    tick grid covers the full plot area.

    Returns the refined PlotBounds (unchanged axes if < 3 ticks detected).
    """

    def _refine(ticks: list[int], lo: int, hi: int) -> tuple[int, int]:
        if len(ticks) < 3:
            return lo, hi
        ticks_s = sorted(ticks)
        diffs = [ticks_s[i + 1] - ticks_s[i] for i in range(len(ticks_s) - 1)]
        base = min(diffs)
        if base < 3:
            return lo, hi
        # Total base intervals between first and last tick
        n_between = round((ticks_s[-1] - ticks_s[0]) / base)
        if n_between <= 0:
            return lo, hi
        spacing = (ticks_s[-1] - ticks_s[0]) / n_between
        # Verify all gaps are integer multiples of spacing (within 20%).
        # Non-uniform tick spacing (e.g. 0,3,6,12,18,24) would produce
        # wrong results.
        for d in diffs:
            m = d / spacing
            if abs(m - round(m)) > 0.2:
                return lo, hi
        # How many intervals from bound-lo to first tick / last tick to bound-hi
        k_lo = round((ticks_s[0] - lo) / spacing)
        k_hi = round((hi - ticks_s[-1]) / spacing)
        new_lo = int(round(ticks_s[0] - k_lo * spacing))
        new_hi = int(round(ticks_s[-1] + k_hi * spacing))
        return new_lo, new_hi

    new_left, new_right = _refine(x_tick_cols, bounds.left, bounds.right)
    new_top, new_bottom = _refine(y_tick_rows, bounds.top, bounds.bottom)
    return PlotBounds(
        left=new_left, right=new_right, top=new_top, bottom=new_bottom,
    )


def _build_curve_mask(img: np.ndarray, bounds: PlotBounds) -> np.ndarray:
    """Build a binary mask of curve pixels, excluding axes and censoring marks.

    Censoring marks (crosses/ticks on Kaplan-Meier curves) are short
    perpendicular marks typically 1-2 pixels wide.  A morphological
    opening with a horizontal structuring element removes them while
    preserving the main curve lines.
    """
    from scipy.ndimage import binary_opening

    gray = img.mean(axis=2)
    dark = gray < 160
    brightness = (img[:, :, 0].astype(int)
                  + img[:, :, 1].astype(int)
                  + img[:, :, 2].astype(int))
    dark = dark & (brightness < 680)

    # Exclude axis lines by position
    axis_margin = 4
    for r in range(max(0, bounds.bottom - axis_margin),
                   min(img.shape[0], bounds.bottom + axis_margin + 1)):
        dark[r, :] = False
    for c in range(max(0, bounds.left - axis_margin),
                   min(img.shape[1], bounds.left + axis_margin + 1)):
        dark[:, c] = False

    # Remove censoring marks: opening with horizontal kernel removes
    # features narrower than 3 px (crosses are 1-2 px wide).
    h_kernel = np.ones((1, 3), dtype=bool)
    dark = binary_opening(dark, structure=h_kernel)

    return dark


def _detect_curve_origin(
    img: np.ndarray,
    raw_bounds: PlotBounds,
    dark_mask: np.ndarray,
    x_ticks: list[int],
    y_ticks: list[int],
) -> tuple[int, int] | None:
    """Detect the pixel where curves start at (x_min, y_max).

    Uses tick-refined bounds to extrapolate the origin — the first detected
    tick may not be at x_min (e.g. ticks at x=6,12,... with origin at x=0).

    Returns (origin_col, origin_row) or None if detection fails.
    """
    gray = np.mean(img, axis=2) if img.ndim == 3 else img.astype(float)

    # Strategy 1: use tick-refined bounds (extrapolates origin from spacing)
    if x_ticks and y_ticks:
        tick_bounds = _refine_bounds_from_ticks(raw_bounds, x_ticks, y_ticks)
        origin_col = tick_bounds.left   # x_min position
        origin_row = tick_bounds.top    # y_max position
    else:
        # Strategy 2: scan dark_mask from top-left for first dark cluster
        search_rows = range(raw_bounds.top, min(raw_bounds.top + 30, raw_bounds.bottom))
        search_cols = range(raw_bounds.left, min(raw_bounds.left + 30, raw_bounds.right))
        for r in search_rows:
            for c in search_cols:
                if dark_mask[r, c]:
                    origin_col, origin_row = c, r
                    break
            else:
                continue
            break
        else:
            return None

    # Validate: check that the origin is within the image and near the
    # raw plot bounds.  For KM curves the exact origin pixel is often
    # white (axis intersection) so we only do a loose sanity check — the
    # nearby area should have *some* dark content (axis line, curve, or
    # label) within a wider radius.
    if not (0 <= origin_row < gray.shape[0] and 0 <= origin_col < gray.shape[1]):
        return None
    r_lo = max(0, origin_row - 8)
    r_hi = min(gray.shape[0], origin_row + 9)
    c_lo = max(0, origin_col - 3)
    c_hi = min(gray.shape[1], origin_col + 12)
    patch = gray[r_lo:r_hi, c_lo:c_hi]
    if patch.size == 0 or patch.min() > 200:
        return None

    return (origin_col, origin_row)


def _origin_anchored_calibration(
    origin: tuple[int, int],
    x_ticks: list[int],
    y_ticks: list[int],
    raw_bounds: PlotBounds,
    axis_range: AxisRange,
) -> tuple[PlotBounds, dict]:
    """Use origin pixel as calibration anchor combined with tick spacing.

    Returns (refined_bounds, calibration_report).
    """
    # Start with tick-refined bounds
    tick_bounds = _refine_bounds_from_ticks(raw_bounds, x_ticks, y_ticks)
    origin_col, origin_row = origin

    # Origin pixel MUST map to (x_min, y_max).
    # Override bounds.left and bounds.top if within 5px of tick-based values.
    new_left = tick_bounds.left
    new_top = tick_bounds.top
    new_right = tick_bounds.right
    new_bottom = tick_bounds.bottom

    if abs(origin_col - tick_bounds.left) <= 5:
        new_left = origin_col
    if abs(origin_row - tick_bounds.top) <= 5:
        new_top = origin_row

    # Recompute right/bottom from origin + tick spacing scale
    if x_ticks and len(x_ticks) >= 2:
        x_sorted = sorted(x_ticks)
        diffs = [x_sorted[i + 1] - x_sorted[i] for i in range(len(x_sorted) - 1)]
        base_x = min(diffs)
        if base_x >= 3:
            n_total = round((x_sorted[-1] - x_sorted[0]) / base_x)
            if n_total > 0:
                spacing_x = (x_sorted[-1] - x_sorted[0]) / n_total
                # How many intervals from origin to last tick, then to right edge
                k_right = round((tick_bounds.right - x_sorted[-1]) / spacing_x)
                new_right = int(round(x_sorted[-1] + k_right * spacing_x))

    if y_ticks and len(y_ticks) >= 2:
        y_sorted = sorted(y_ticks)
        diffs = [y_sorted[i + 1] - y_sorted[i] for i in range(len(y_sorted) - 1)]
        base_y = min(diffs)
        if base_y >= 3:
            n_total = round((y_sorted[-1] - y_sorted[0]) / base_y)
            if n_total > 0:
                spacing_y = (y_sorted[-1] - y_sorted[0]) / n_total
                k_bottom = round((tick_bounds.bottom - y_sorted[-1]) / spacing_y)
                new_bottom = int(round(y_sorted[-1] + k_bottom * spacing_y))

    refined = PlotBounds(
        left=new_left, right=new_right,
        top=new_top, bottom=new_bottom,
    )

    # Build calibration report
    report: dict = {
        "origin_pixel": (origin_col, origin_row),
        "origin_data": (axis_range.x_min, axis_range.y_max),
        "tick_consistency": "unknown",
        "quality": "poor",
    }

    # Check origin-tick consistency: origin should agree with tick_bounds
    if x_ticks and y_ticks:
        x_sorted = sorted(x_ticks)
        y_sorted = sorted(y_ticks)
        x_err = abs(origin_col - tick_bounds.left)
        y_err = abs(origin_row - tick_bounds.top)
        report["origin_tick_x_err_px"] = x_err
        report["origin_tick_y_err_px"] = y_err
        consistent = x_err <= 3 and y_err <= 3
        report["tick_consistency"] = "consistent" if consistent else "offset"

        # Tick-to-data mappings
        x_range = axis_range.x_max - axis_range.x_min
        y_range = axis_range.y_max - axis_range.y_min
        tick_mappings = []
        for xt in x_sorted:
            x_frac = (xt - refined.left) / max(1, refined.width)
            tick_mappings.append({"pixel_col": xt,
                                  "data_x": round(axis_range.x_min + x_frac * x_range, 3)})
        for yt in y_sorted:
            y_frac = (refined.bottom - yt) / max(1, refined.height)
            tick_mappings.append({"pixel_row": yt,
                                  "data_y": round(axis_range.y_min + y_frac * y_range, 3)})
        report["tick_mappings"] = tick_mappings

        if consistent and len(x_ticks) >= 3 and len(y_ticks) >= 3:
            report["quality"] = "excellent"
        elif x_err <= 5 and y_err <= 5:
            report["quality"] = "good"

    return refined, report


def _detect_series_colors(
    img: np.ndarray,
    data_series: list[dict],
    bounds: PlotBounds,
    axis_range: AxisRange,
    search_radius: int = 50,
) -> list[tuple[int, int, int]]:
    """Detect the dominant color of each AI series from the image.

    For each data point, finds all curve-like segments in the column
    (gray < 220, non-white, non-black), picks the segment closest to
    the AI y-position, and samples its median color.  This is robust
    even when AI y-values are off by 50+ pixels.

    Returns:
        List of (R, G, B) tuples, one per series.
    """
    gray = img.mean(axis=2)

    colors: list[tuple[int, int, int]] = []
    for series in data_series:
        pixels: list[tuple[int, int, int]] = []
        for pt in series.get("data", []):
            if "x" not in pt or "y" not in pt:
                continue
            x, y = float(pt["x"]), float(pt["y"])
            col, row_expected = _data_to_pixel(x, y, bounds, axis_range)
            if col <= bounds.left + 5 or col >= bounds.right - 2:
                continue

            # Find all curve-like pixel segments in this column
            col_lo = max(bounds.left, col - 1)
            col_hi = min(bounds.right, col + 1)
            # Use a lenient threshold to catch grey curves too
            band_gray = gray[bounds.top:bounds.bottom + 1, col_lo:col_hi + 1].min(axis=1)
            curve_rows = np.where(band_gray < 220)[0] + bounds.top

            if len(curve_rows) == 0:
                continue

            # Cluster into contiguous segments
            diffs = np.diff(curve_rows)
            splits = np.where(diffs > 5)[0] + 1
            segments = np.split(curve_rows, splits)

            # Pick the segment closest to AI expected row
            best_seg = None
            best_dist = float("inf")
            for seg in segments:
                if len(seg) < 2:
                    continue
                seg_center = float(seg.mean())
                dist = abs(seg_center - row_expected)
                if dist < best_dist and dist <= search_radius:
                    best_dist = dist
                    best_seg = seg

            if best_seg is not None:
                # Sample the darkest pixel in the segment (curve core)
                seg_pixels = img[best_seg, col]  # (N, 3)
                # Pick the pixel with lowest brightness (most likely the
                # actual curve rather than anti-aliased edge)
                brightness = seg_pixels.sum(axis=1)
                darkest_idx = np.argmin(brightness)
                px = seg_pixels[darkest_idx]
                pixels.append((int(px[0]), int(px[1]), int(px[2])))

        if pixels:
            arr = np.array(pixels)
            median_rgb = (int(np.median(arr[:, 0])),
                          int(np.median(arr[:, 1])),
                          int(np.median(arr[:, 2])))
            colors.append(median_rgb)
        else:
            colors.append((80, 80, 80))  # fallback grey
    return colors


def _colors_all_similar(colors: list[tuple], threshold: float = 50.0) -> bool:
    """Check if all detected series colors are effectively the same.

    Two checks:
      1. All pairwise Euclidean RGB distances < threshold.
      2. All colors are achromatic (R≈G≈B within 30).  This catches
         grayscale/BW charts where sparse-pixel sampling produces
         varying brightness levels (e.g. black vs medium-gray) but the
         curves are really all the same color.
    """
    all_close = True
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            dist = sum((a - b) ** 2 for a, b in zip(colors[i], colors[j])) ** 0.5
            if dist > threshold:
                all_close = False
                break
        if not all_close:
            break
    if all_close:
        return True
    # All achromatic → grayscale chart (different brightness is a sampling artifact)
    for c in colors:
        if max(c) - min(c) > 30:
            return False
    return True


def _infer_line_style(name: str) -> str | None:
    """Parse line-style hints from AI series names."""
    low = name.lower()
    if "solid" in low:
        return "solid"
    if "dash" in low:
        return "dashed"
    if "dot" in low:
        return "dotted"
    return None


def _build_per_series_masks(
    img: np.ndarray,
    data_series: list[dict],
    bounds: PlotBounds,
    axis_range: AxisRange,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build a targeted color mask for each AI series.

    Strategy: detect each series' color, then find all curve-like pixels
    (gray < 220, not white, not black) and assign each to the nearest
    series by RGB distance.  This avoids fixed-tolerance misses.

    For single-series charts, falls back to ``make_color_mask`` with a
    generous tolerance.

    Returns:
        (per_series_masks, union_mask) where each entry is a boolean array
        of the same shape as the image (H, W).
    """
    from scipy.ndimage import binary_opening

    colors = _detect_series_colors(img, data_series, bounds, axis_range)
    # Two-pass opening: horizontal (1,3) removes vertical cross arms
    # (<3px wide), vertical (2,1) removes single-row stray pixels from
    # horizontal cross arms.  Actual curve lines (≥2px tall) survive.
    h_kernel = np.ones((1, 3), dtype=bool)
    v_kernel = np.ones((2, 1), dtype=bool)
    n_series = len(data_series)

    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        mask = binary_opening(mask, structure=h_kernel)
        mask = binary_opening(mask, structure=v_kernel)
        return mask

    if n_series <= 1:
        # Single series — use make_color_mask with generous tolerance
        masks: list[np.ndarray] = []
        union = np.zeros(img.shape[:2], dtype=bool)
        for color in colors:
            mask = make_color_mask(img, color, tolerance=70, bounds=bounds)
            axis_margin = 4
            for r in range(max(0, bounds.bottom - axis_margin),
                           min(img.shape[0], bounds.bottom + axis_margin + 1)):
                mask[r, :] = False
            for c in range(max(0, bounds.left - axis_margin),
                           min(img.shape[1], bounds.left + axis_margin + 1)):
                mask[:, c] = False
            mask = _clean_mask(mask)
            masks.append(mask)
            union |= mask
        return masks, union

    # Same-color charts: separate by line style instead of color
    if _colors_all_similar(colors):
        # Build global curve mask (same filter as below)
        gray = img.mean(axis=2)
        r_ch, g_ch, b_ch = (img[:, :, 0].astype(np.int16),
                             img[:, :, 1].astype(np.int16),
                             img[:, :, 2].astype(np.int16))
        brightness = r_ch + g_ch + b_ch
        curve_pixels = (gray < 220) & (brightness > 100) & (brightness < 680)
        axis_margin = 4
        for row in range(max(0, bounds.bottom - axis_margin),
                         min(img.shape[0], bounds.bottom + axis_margin + 1)):
            curve_pixels[row, :] = False
        for col_px in range(max(0, bounds.left - axis_margin),
                            min(img.shape[1], bounds.left + axis_margin + 1)):
            curve_pixels[:, col_px] = False
        plot_mask = np.zeros(img.shape[:2], dtype=bool)
        plot_mask[max(0, bounds.top - 5):bounds.bottom + 1,
                  bounds.left:bounds.right + 5] = True
        curve_pixels &= plot_mask

        # Classify by line style
        style_masks = classify_line_style(curve_pixels, bounds)

        # Infer line styles from AI series names
        hints = [_infer_line_style(s.get("name", "")) for s in data_series]
        distinct_hints = [h for h in hints if h is not None]

        if (len(distinct_hints) == n_series
                and len(set(distinct_hints)) == n_series):
            # All series have distinct style hints -- assign style sub-masks
            # but only if every sub-mask has enough pixels for assessment
            masks = []
            for h in hints:
                m = _clean_mask(style_masks.get(h, np.zeros(img.shape[:2], dtype=bool)))
                masks.append(m)
            if all(m.sum() >= 30 for m in masks):
                union = np.zeros(img.shape[:2], dtype=bool)
                for m in masks:
                    union |= m
                return masks, union
            # else fall through to global mask path

        # No hints, ambiguous, or style sub-masks too sparse --
        # skip _clean_mask so dashed/dotted segments survive
        masks = [curve_pixels.copy() for _ in range(n_series)]
        union = curve_pixels.copy()
        return masks, union

    # Multi-series: find all "curve-like" pixels, assign by nearest color
    gray = img.mean(axis=2)
    r, g, b = img[:, :, 0].astype(np.int16), img[:, :, 1].astype(np.int16), img[:, :, 2].astype(np.int16)
    brightness = r + g + b

    # Curve-like: darker than background, not black axes/text
    curve_pixels = (gray < 220) & (brightness > 100) & (brightness < 680)

    # Exclude axis lines
    axis_margin = 4
    for row in range(max(0, bounds.bottom - axis_margin),
                     min(img.shape[0], bounds.bottom + axis_margin + 1)):
        curve_pixels[row, :] = False
    for col_px in range(max(0, bounds.left - axis_margin),
                        min(img.shape[1], bounds.left + axis_margin + 1)):
        curve_pixels[:, col_px] = False

    # Restrict to plot area with small margin
    plot_mask = np.zeros(img.shape[:2], dtype=bool)
    plot_mask[max(0, bounds.top - 5):bounds.bottom + 1,
              bounds.left:bounds.right + 5] = True
    curve_pixels &= plot_mask

    # Assign each curve pixel to the nearest series by Euclidean RGB
    # distance.  This correctly separates grey vs teal curves.  Anti-aliased
    # edges may be assigned to the wrong series (fragmenting masks to 2–3px
    # slivers), but trace_curve handles this via a relaxed segment gap.
    color_arr = np.array(colors, dtype=np.float64)  # (N, 3)

    masks = [np.zeros(img.shape[:2], dtype=bool) for _ in range(n_series)]
    union = np.zeros(img.shape[:2], dtype=bool)

    rows, cols = np.where(curve_pixels)
    if len(rows) > 0:
        px_rgb = img[rows, cols].astype(np.float64)  # (M, 3)
        dists = np.zeros((len(rows), n_series), dtype=np.float64)
        for si in range(n_series):
            diff = px_rgb - color_arr[si]
            dists[:, si] = (diff ** 2).sum(axis=1)
        nearest = np.argmin(dists, axis=1)

        for si in range(n_series):
            si_mask = nearest == si
            masks[si][rows[si_mask], cols[si_mask]] = True
            masks[si] = _clean_mask(masks[si])
        union = np.zeros(img.shape[:2], dtype=bool)
        for si in range(n_series):
            union |= masks[si]

    return masks, union


def assess_extraction_accuracy(
    img: np.ndarray,
    data_series: list[dict],
    bounds: PlotBounds,
    axis_range: AxisRange,
    search_radius: int = 30,
    per_series_masks: list[np.ndarray] | None = None,
) -> dict:
    """CV-based extraction accuracy assessment.

    For each extracted data point, finds the nearest dark curve pixel
    vertically and measures the discrepancy.

    Args:
        per_series_masks: Optional list of boolean masks, one per series.
            When provided, ``per_series_masks[si]`` is used instead of the
            shared dark mask for series *si*.  Backward compatible — existing
            callers pass nothing.

    Returns dict with per-series metrics, overall metrics, and
    a feedback_text string describing specific errors for AI correction.
    """
    dark = _build_curve_mask(img, bounds)

    y_full_scale = axis_range.y_max - axis_range.y_min

    series_results = []
    all_errors: list[float] = []

    for si, series in enumerate(data_series):
        series_mask = (
            per_series_masks[si]
            if per_series_masks is not None and si < len(per_series_masks)
            else dark
        )
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

            # Find nearest curve pixel vertically
            col_lo = max(bounds.left, col - 2)
            col_hi = min(bounds.right, col + 2)
            actual_row = None
            for dist in range(search_radius + 1):
                for dr in ([0] if dist == 0 else [-dist, dist]):
                    r = row_expected + dr
                    if bounds.top <= r <= bounds.bottom and series_mask[r, col_lo:col_hi + 1].any():
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
    ai_guide: list[dict] | None = None,
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
        ai_guide: Optional AI data points [{"x", "y"}, ...] used for
            initialization and jump recovery.  When provided, the first
            segment is chosen near the AI-expected position (instead of
            topmost), and large jumps (>30px) are checked against the
            AI guide before committing.

    Returns:
        List of {"x": float, "y": float} dicts.
    """
    row_min = bounds.top + 5
    row_max = bounds.bottom - 10

    if row_min >= row_max:
        row_min = bounds.top
        row_max = bounds.bottom

    # Build AI-guide interpolation (data-y → pixel-row at each column)
    _ai_row_at_col: dict[int, float] | None = None
    if ai_guide and len(ai_guide) >= 2:
        ai_xs = [p["x"] for p in ai_guide if "x" in p and "y" in p]
        ai_ys = [p["y"] for p in ai_guide if "x" in p and "y" in p]
        if len(ai_xs) >= 2:
            _ai_row_at_col = {}
            for col in range(bounds.left, bounds.right + 1):
                x_data = pixel_to_data(col, 0, bounds, axis_range)[0]
                # Linear interpolation of AI y
                y_interp = np.interp(x_data, ai_xs, ai_ys)
                _, row_interp = _data_to_pixel(x_data, y_interp, bounds, axis_range)
                _ai_row_at_col[col] = float(row_interp)

    col_to_row: dict[int, float] = {}
    prev_row: float | None = None

    for col in range(bounds.left, bounds.right + 1):
        rows = np.where(mask[row_min:row_max + 1, col])[0] + row_min
        if len(rows) == 0:
            continue

        # Split into contiguous segments (gap > 5 merges fragments
        # separated by anti-aliasing gaps of 3–5 pixels)
        diffs = np.diff(rows)
        splits = np.where(diffs > 5)[0] + 1
        segments = np.split(rows, splits)

        candidates = [s for s in segments if len(s) >= 2]
        if not candidates:
            continue

        if prev_row is not None:
            best = min(candidates, key=lambda s: abs(s.mean() - prev_row))
            # Skip column if nearest segment is too far — likely a
            # censoring mark, not the actual curve.  Gap interpolation
            # will fill in the skipped columns later.
            max_jump = bounds.height * 0.12
            if abs(best.mean() - prev_row) > max_jump:
                continue
        else:
            # First detection: use AI guide if available, else largest
            if _ai_row_at_col is not None and col in _ai_row_at_col:
                ai_row = _ai_row_at_col[col]
                best = min(candidates, key=lambda s: abs(s.mean() - ai_row))
            else:
                best = max(candidates, key=len)

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
    bounds_override: PlotBounds | None = None,
) -> list[dict]:
    """Extract curves from a chart image using pixel-level color detection.

    Top-level entry point. Auto-detects plot bounds (or uses bounds_override),
    builds color masks per series, traces each curve, and returns data in the
    standard format.

    When multiple series share the same color, switches to multi-curve tracing
    using trace_multi_curves and optionally uses line-style classification to
    assign curves to the correct series.

    Args:
        image_bytes: Raw image bytes (PNG/JPEG/WebP).
        series_specs: List of SeriesSpec for each series to trace.
        axis_range: Data-coordinate axis ranges from metadata.
        n_points: Number of output points per series.
        bounds_override: Pre-computed plot bounds (skips auto-detection).

    Returns:
        List of {"name": str, "data": [{"x": float, "y": float}, ...]} dicts.
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    bounds = bounds_override or detect_plot_bounds(img)
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

    dark = _build_curve_mask(img, bounds)

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


def snap_series_to_pixels_guided(
    image_bytes: bytes,
    data_series: list[dict],
    axis_range: AxisRange,
    series_info: list | None = None,
    search_radius: int = 25,
) -> list[dict]:
    """BW-aware pixel-snap using per-series guided masks.

    For each series, builds a guided mask (band around the AI trajectory)
    with optional morphological closing for dashed/dotted lines, then snaps
    points to curve pixels on that mask.  This prevents text pixels and
    other-series pixels from corrupting the snap.

    Args:
        image_bytes: Original chart image bytes.
        data_series: AI-extracted [{"name", "data": [{"x", "y"}, ...]}, ...].
        axis_range: Data-coordinate axis ranges.
        series_info: AI-provided series metadata with ``line_style`` etc.
        search_radius: Max pixel distance to search vertically.

    Returns:
        Corrected data_series (same structure, more accurate y-values).
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    bounds = detect_plot_bounds(img)
    if bounds is None or bounds.height < 20:
        return data_series

    dark_mask = _build_curve_mask(img, bounds)

    corrected_series: list[dict] = []
    for si, series in enumerate(data_series):
        ai_pts = series.get("data", [])
        series_name = series.get("name", f"Series {si}")

        # Extract line_style from series_info
        ls = None
        if series_info and si < len(series_info):
            ls = series_info[si].get("line_style")
            if ls == "unknown":
                ls = None

        # Build guided mask with gap-bridging for dashed/dotted
        guided = _build_guided_mask(
            dark_mask, ai_pts, bounds, axis_range, line_style=ls,
        )

        # Snap on the guided mask
        snapped = _snap_series_on_mask(
            guided, ai_pts, bounds, axis_range,
            search_radius=search_radius,
        )
        corrected_series.append({"name": series_name, "data": snapped})

    return corrected_series


# ---------------------------------------------------------------------------
# Per-series snap helpers
# ---------------------------------------------------------------------------

def _snap_series_on_mask(
    mask: np.ndarray,
    series_data: list[dict],
    bounds: PlotBounds,
    axis_range: AxisRange,
    search_radius: int = 25,
    apply_median: bool = True,
) -> list[dict]:
    """Snap a single series' AI points to the nearest curve pixel on *mask*.

    Like ``snap_series_to_pixels`` but operates on a single series with a
    caller-provided mask (no claimed-pixel tracking needed).

    Args:
        mask: Boolean mask (H, W) of curve pixels for this series.
        series_data: List of {"x", "y"} dicts (AI points for one series).
        bounds: Plot bounds.
        axis_range: Data-coordinate axis ranges.
        search_radius: Max vertical pixel distance to search.
        apply_median: If True (default), apply median filter to smooth snap
            noise.  Set False to preserve step-function edges.

    Returns:
        List of {"x", "y"} dicts with corrected y-values.
    """
    corrected: list[dict] = []
    for pt in series_data:
        if "x" not in pt or "y" not in pt:
            corrected.append(pt)
            continue
        x, y = float(pt["x"]), float(pt["y"])
        col, row = _data_to_pixel(x, y, bounds, axis_range)
        col = max(bounds.left, min(bounds.right, col))

        col_lo = max(bounds.left, col - 2)
        col_hi = min(bounds.right, col + 2)

        best_row: float | None = None
        for dist in range(search_radius + 1):
            for dr in ([0] if dist == 0 else [-dist, dist]):
                r = row + dr
                if bounds.top <= r <= bounds.bottom and mask[r, col_lo:col_hi + 1].any():
                    # Refine: find segment center in the band
                    band = mask[max(bounds.top, r - 2):min(bounds.bottom, r + 3),
                                col_lo:col_hi + 1]
                    rows_hit = np.where(band.any(axis=1))[0]
                    if len(rows_hit):
                        best_row = float(rows_hit.mean()) + max(bounds.top, r - 2)
                    else:
                        best_row = float(r)
                    break
            if best_row is not None:
                break

        if best_row is not None:
            _, corrected_y = pixel_to_data(col, best_row, bounds, axis_range)
            corrected.append({"x": round(x, 4), "y": round(corrected_y, 4)})
        else:
            corrected.append({"x": round(x, 4), "y": round(y, 4)})

    # Median filter to smooth snap noise
    if apply_median and len(corrected) >= 5:
        ys = np.array([p["y"] for p in corrected])
        smooth = median_filter(ys, size=5)
        for i, p in enumerate(corrected):
            p["y"] = round(float(smooth[i]), 4)

    return corrected


def _dense_snap_on_mask(
    mask: np.ndarray,
    raw_pts: list[dict],
    ai_pts: list[dict],
    bounds: PlotBounds,
    axis_range: AxisRange,
) -> list[dict]:
    """Two-pass dense snap: snap trajectory points to curve segments on *mask*.

    Pass 1: For each trajectory point, find the nearest segment in *mask*
    guided by AI-expected y-values.
    Pass 2: Interpolate across gaps from successfully snapped points.

    Args:
        mask: Boolean mask (H, W) of curve pixels for this series.
        raw_pts: Dense trajectory points [{"x", "y"}, ...] from
            ``find_trajectory``.
        ai_pts: AI data points [{"x", "y"}, ...] for this series (used
            to guide which segment to snap to).
        bounds: Plot bounds.
        axis_range: Data-coordinate axis ranges.

    Returns:
        List of {"x", "y"} dicts — dense, pixel-accurate points.
    """
    from scipy.interpolate import interp1d

    # Build AI interpolation for guiding segment selection
    ai_xs = [p["x"] for p in ai_pts if "x" in p and "y" in p]
    ai_ys = [p["y"] for p in ai_pts if "x" in p and "y" in p]
    ai_interp = None
    if len(ai_xs) >= 2:
        ai_interp = interp1d(ai_xs, ai_ys, kind="linear",
                             fill_value="extrapolate", bounds_error=False)

    max_snap_dist = max(8, bounds.height // 15)

    def _snap_col(col: int, expected_row: int) -> float | None:
        col = max(bounds.left, min(bounds.right, col))
        col_pixels = mask[bounds.top:bounds.bottom + 1, col]
        segments = []
        in_seg = False
        seg_start = 0
        for r_off, v in enumerate(col_pixels):
            if v and not in_seg:
                seg_start = r_off
                in_seg = True
            elif not v and in_seg:
                segments.append((seg_start + bounds.top, r_off - 1 + bounds.top))
                in_seg = False
        if in_seg:
            segments.append((seg_start + bounds.top, len(col_pixels) - 1 + bounds.top))

        best_row = None
        best_dist = float("inf")
        for seg_top, seg_bot in segments:
            seg_center = (seg_top + seg_bot) // 2
            dist = abs(seg_center - expected_row)
            if dist < best_dist:
                best_dist = dist
                best_row = seg_center

        if best_row is not None and best_dist <= max_snap_dist:
            _, snapped_y = pixel_to_data(col, best_row, bounds, axis_range)
            return round(snapped_y, 4)
        return None

    # Pass 1: snap to curve segments
    pass1_pts: list[dict] = []
    snapped_xs: list[float] = []
    snapped_ys: list[float] = []
    for pt in raw_pts:
        col, _ = _data_to_pixel(pt["x"], pt["y"], bounds, axis_range)
        if ai_interp is not None:
            expected_y = float(ai_interp(pt["x"]))
            _, expected_row = _data_to_pixel(pt["x"], expected_y, bounds, axis_range)
        else:
            _, expected_row = _data_to_pixel(pt["x"], pt["y"], bounds, axis_range)

        y_val = _snap_col(col, expected_row)
        if y_val is not None:
            pass1_pts.append({"x": pt["x"], "y": y_val, "_snapped": True})
            snapped_xs.append(pt["x"])
            snapped_ys.append(y_val)
        else:
            pass1_pts.append({"x": pt["x"], "y": None, "_snapped": False})

    # Pass 2: interpolate gaps
    if len(snapped_xs) >= 2:
        gap_interp = interp1d(snapped_xs, snapped_ys, kind="linear",
                              fill_value="extrapolate", bounds_error=False)
        for p in pass1_pts:
            if not p["_snapped"]:
                p["y"] = round(float(gap_interp(p["x"])), 4)

    dense_pts = [{"x": p["x"], "y": p["y"]}
                 for p in pass1_pts if p["y"] is not None
                 and axis_range.x_min <= p["x"] <= axis_range.x_max
                 and axis_range.y_min - 1 <= p["y"] <= axis_range.y_max + 1]

    # Light median filter
    if len(dense_pts) >= 3:
        xs = np.array([p["x"] for p in dense_pts])
        ys = np.array([p["y"] for p in dense_pts])
        ys = median_filter(ys, size=3)
        dense_pts = [{"x": round(float(x), 4), "y": round(float(y), 4)}
                     for x, y in zip(xs, ys)]

    return dense_pts


# ---------------------------------------------------------------------------
# BW guided mask for text-free per-series extraction
# ---------------------------------------------------------------------------


def _build_guided_mask(
    dark_mask: np.ndarray,
    ai_pts: list[dict],
    bounds: PlotBounds,
    axis_range: AxisRange,
    guide_radius: int = 30,
    line_style: str | None = None,
) -> np.ndarray:
    """Build a per-series mask by keeping only dark pixels near the AI guide.

    For BW charts where all curves share the same color, this creates a
    "band mask" around each series' expected trajectory.  Pixels far from
    the guide (e.g. title text, legend text) are excluded.

    Algorithm: For each column in the plot, interpolate the AI guide to get
    an expected row.  Keep only ``dark_mask`` pixels within ±guide_radius
    rows of that expected position.  Zero everything else.

    For dashed/dotted lines, a horizontal morphological closing is applied
    after band filtering to bridge dash gaps (up to 14px for dashed, 8px
    for dotted).  The kernel is ``(1, N)`` so it only fills horizontally
    without expanding vertically (won't merge separate curves).

    Args:
        dark_mask: Boolean mask of all dark pixels (curves + text + axes).
        ai_pts: AI-extracted data points for this series [{"x", "y"}, ...].
        bounds: Plot area pixel boundaries.
        axis_range: Data-coordinate axis ranges.
        guide_radius: Max pixel distance from guide to keep (default 30).
        line_style: One of "solid", "dashed", "dotted", or None.
            Dashed/dotted triggers horizontal closing to bridge gaps.

    Returns:
        Boolean mask same shape as dark_mask, with only guided pixels set.
    """
    guided = np.zeros_like(dark_mask)

    # Build sorted (col, row) guide from AI points
    guide_pairs = []
    for pt in ai_pts:
        if "x" not in pt or "y" not in pt:
            continue
        col, row = _data_to_pixel(float(pt["x"]), float(pt["y"]), bounds, axis_range)
        guide_pairs.append((col, row))
    if len(guide_pairs) < 2:
        # Not enough guide points — return the raw dark_mask clipped to plot
        guided[bounds.top:bounds.bottom + 1, bounds.left:bounds.right + 1] = \
            dark_mask[bounds.top:bounds.bottom + 1, bounds.left:bounds.right + 1]
        return guided

    # Sort by column for interpolation
    guide_pairs.sort(key=lambda p: p[0])
    guide_cols = np.array([p[0] for p in guide_pairs], dtype=float)
    guide_rows = np.array([p[1] for p in guide_pairs], dtype=float)

    # For each plot column, interpolate expected row and apply band filter
    for col in range(bounds.left, bounds.right + 1):
        expected_row = np.interp(float(col), guide_cols, guide_rows)
        row_lo = max(bounds.top, int(expected_row - guide_radius))
        row_hi = min(bounds.bottom, int(expected_row + guide_radius))
        guided[row_lo:row_hi + 1, col] = dark_mask[row_lo:row_hi + 1, col]

    # Bridge dash/dot gaps with horizontal morphological closing.
    # Kernel (1, N) fills gaps up to N-1 px wide without vertical expansion.
    if line_style in ("dashed", "dotted"):
        from scipy.ndimage import binary_closing
        close_width = 15 if line_style == "dashed" else 9
        h_kernel = np.ones((1, close_width), dtype=bool)
        guided = binary_closing(guided, structure=h_kernel)

    return guided


# ---------------------------------------------------------------------------
# PlotDigitizer ensemble extraction
# ---------------------------------------------------------------------------


def extract_with_plotdigitizer(
    image_bytes: bytes,
    data_series: list[dict],
    axis_range: AxisRange,
    *,
    is_bw: bool | None = None,
    series_info: list | None = None,
) -> list[dict]:
    """Per-series adaptive extraction using color-aware or guided masks.

    For color charts, each series independently:
      A) Sparse snap on per-series color mask
      B) Dense snap (plotdigitizer trajectory) on per-series color mask
      C) Sparse snap on standard dark mask (fallback)
      D) Column-by-column trace on per-series color mask
    For BW charts, builds per-series guided masks from AI trajectory,
    then runs candidates A-C on the guided mask (no multi-trace).

    Args:
        image_bytes: Original chart image bytes.
        data_series: AI-extracted series (used for curve count and matching).
        axis_range: Data-coordinate axis ranges.
        is_bw: If True, use BW guided-mask path. If False, use color path.
            If None, auto-detect via _colors_all_similar.
        series_info: AI-provided series metadata (line styles, colors).

    Returns:
        data_series in standard format with best-method y-values per series.
        Raises on failure (caller should catch and fall back).
    """
    from plotdigitizer.plotdigitizer import axis_transformation
    from plotdigitizer.trajectory import find_trajectory

    img_rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    raw_bounds = detect_plot_bounds(img_rgb)
    if raw_bounds is None or raw_bounds.height < 20:
        raise ValueError("Plot bounds detection failed for plotdigitizer.")

    # Refine bounds using tick marks for better axis calibration.
    # Use refined bounds for coordinate conversion, raw bounds for mask
    # building (to include curve pixels near axes).
    x_ticks, y_ticks = detect_axis_ticks(img_rgb, raw_bounds)

    # Build dark mask early — needed for origin detection and later steps
    dark_mask = _build_curve_mask(img_rgb, raw_bounds)

    # Detect curve origin and use origin-anchored calibration
    calibration_report: dict | None = None
    origin = _detect_curve_origin(img_rgb, raw_bounds, dark_mask, x_ticks, y_ticks)
    if origin is not None:
        bounds, calibration_report = _origin_anchored_calibration(
            origin, x_ticks, y_ticks, raw_bounds, axis_range,
        )
        print(f"[origin] Detected at pixel {origin} → "
              f"data ({axis_range.x_min}, {axis_range.y_max})")
        if calibration_report:
            print(f"[origin] Calibration quality: {calibration_report.get('quality')}")
            if "tick_mappings" in calibration_report:
                for tm in calibration_report["tick_mappings"]:
                    print(f"  tick → {tm}")
    else:
        bounds = _refine_bounds_from_ticks(raw_bounds, x_ticks, y_ticks)

    n_series = len(data_series)

    # --- Step 1: Build per-series color masks ---
    # Use raw_bounds for mask building to capture pixels near axes
    per_series_masks, union_mask = _build_per_series_masks(
        img_rgb, data_series, raw_bounds, axis_range,
    )

    # plotdigitizer calibration (uses refined bounds for accuracy)
    YROWS = img_rgb.shape[0]
    data_points = [
        (axis_range.x_min, axis_range.y_min),
        (axis_range.x_max, axis_range.y_min),
        (axis_range.x_min, axis_range.y_max),
    ]
    pixel_locations = [
        (bounds.left, YROWS - bounds.bottom),
        (bounds.right, YROWS - bounds.bottom),
        (bounds.left, YROWS - bounds.top),
    ]
    T = axis_transformation(data_points, pixel_locations)

    # Remove gridline remnants from the dark mask (use raw_bounds for pixel range)
    max_col_density = max(15, raw_bounds.height // 8)
    for col in range(raw_bounds.left, raw_bounds.right + 1):
        col_count = dark_mask[raw_bounds.top:raw_bounds.bottom + 1, col].sum()
        if col_count > max_col_density:
            dark_mask[:, col] = False

    # --- Step 1b: BW vs color routing ---
    # Determine if chart is BW (auto-detect if is_bw not provided)
    multi_trace_curves = None
    _detected_bw = False
    if is_bw is None and n_series >= 2:
        colors = _detect_series_colors(img_rgb, data_series, raw_bounds, axis_range)
        _detected_bw = _colors_all_similar(colors)
    use_bw_path = is_bw if is_bw is not None else _detected_bw

    if use_bw_path:
        # BW path: build per-series guided masks from AI trajectory.
        # This excludes title text/legend text that would confuse tracing.
        print("[BW path] Building guided masks for %d series" % n_series)
        for si in range(n_series):
            ai_pts = data_series[si].get("data", [])
            # Extract line_style from series_info if available
            ls = None
            if series_info and si < len(series_info):
                ls = series_info[si].get("line_style")
                if ls == "unknown":
                    ls = None
            guided = _build_guided_mask(
                dark_mask, ai_pts, bounds, axis_range, line_style=ls,
            )
            per_series_masks[si] = guided
        # Disable multi-trace entirely for BW charts — it operates on raw
        # dark_mask and can't distinguish text from curves, leading to
        # title-text tracking failures (e.g. lung chart).
        multi_trace_curves = None

    elif n_series >= 2:
        # Color path: same-color multi-curve tracing (existing logic)
        colors = _detect_series_colors(img_rgb, data_series, raw_bounds, axis_range)
        if _colors_all_similar(colors):
            # Infer monotonicity from all AI data combined
            mono = None
            ai_ys_all = [p["y"] for s in data_series
                         for p in s.get("data", []) if "y" in p]
            if (ai_ys_all and len(ai_ys_all) >= 3
                    and all(ai_ys_all[i] >= ai_ys_all[i + 1]
                            for i in range(len(ai_ys_all) - 1))):
                mono = "decreasing"
            multi_trace_curves = trace_multi_curves(
                dark_mask, bounds, axis_range,
                n_curves=n_series, n_points=300, monotonic=mono,
            )
            # Match traced curves to AI series by rank order of mean y.
            # Rank-based matching is more robust than Hungarian here because
            # AI means are systematically higher (they include the y=1.0
            # starting point that traces miss near the axis).
            if multi_trace_curves and any(multi_trace_curves):
                ai_means = []
                for s in data_series:
                    ys = [p["y"] for p in s.get("data", []) if "y" in p]
                    ai_means.append(np.mean(ys) if ys else 0.0)
                trace_means = []
                for curve in multi_trace_curves:
                    ys = [p["y"] for p in curve if "y" in p]
                    trace_means.append(np.mean(ys) if ys else 0.0)
                # Sort both by mean-y descending, pair by rank
                ai_order = sorted(range(len(ai_means)),
                                  key=lambda i: ai_means[i], reverse=True)
                tr_order = sorted(range(len(trace_means)),
                                  key=lambda i: trace_means[i], reverse=True)
                reordered = [[] for _ in range(len(ai_means))]
                for rank in range(min(len(ai_order), len(tr_order))):
                    reordered[ai_order[rank]] = multi_trace_curves[tr_order[rank]]
                multi_trace_curves = reordered

    # --- Step 2: For each series, generate candidates and pick best ---
    result: list[dict] = []
    for si in range(n_series):
        series = data_series[si]
        ai_pts = series.get("data", [])
        color_mask = per_series_masks[si]
        series_name = series.get("name", f"Series {si}")

        candidates: list[tuple[str, list[dict]]] = []

        # For same-color charts, Candidates A–D operate on a shared mask
        # containing ALL curves, so they can't distinguish which curve to
        # follow and will jump between curves.  Only Candidate E (multi-
        # trace) properly separates same-color curves, so skip B/D when
        # multi-trace is available.
        same_color_mode = multi_trace_curves is not None

        # Candidate A: sparse snap on color mask
        try:
            snap_color = _snap_series_on_mask(
                color_mask, ai_pts, bounds, axis_range,
            )
            if snap_color:
                candidates.append(("snap-color", snap_color))
        except Exception:
            pass

        # Candidate B: dense snap on color mask via plotdigitizer trajectory
        if not same_color_mode:
            try:
                # Run trajectory on this series' color mask
                img_single = np.where(color_mask, np.uint8(0), np.uint8(255))
                res, _ = find_trajectory(img_single, 0, T)
                if res and len(res) >= 5:
                    raw_pts = [{"x": round(float(x), 4), "y": round(float(y), 4)}
                               for x, y in res]
                    raw_pts = [p for p in raw_pts
                               if axis_range.x_min - 0.5 <= p["x"] <= axis_range.x_max + 0.5
                               and axis_range.y_min - 0.5 <= p["y"] <= axis_range.y_max + 0.5]
                    if len(raw_pts) >= 5:
                        dense = _dense_snap_on_mask(
                            color_mask, raw_pts, ai_pts, bounds, axis_range,
                        )
                        if dense:
                            candidates.append(("dense-color", dense))
            except Exception:
                pass

        # Candidate C: sparse snap on standard dark mask
        # Skip for BW charts — raw dark_mask includes text pixels that
        # confuse tracing.  The guided mask (Candidate A) already handles it.
        if not use_bw_path:
            try:
                snap_dark = _snap_series_on_mask(
                    dark_mask, ai_pts, bounds, axis_range,
                )
                if snap_dark:
                    candidates.append(("snap-dark", snap_dark))
            except Exception:
                pass

        # Candidate D: trace_curve on color mask (column-by-column tracking)
        # This follows the actual pixel segments step-by-step, which is
        # superior for step functions where the AI-guided dense snap
        # cuts through steps with linear interpolation.
        if not same_color_mode:
            try:
                # Infer monotonicity from AI data
                ai_ys = [p["y"] for p in ai_pts if "y" in p]
                mono = None
                if len(ai_ys) >= 3:
                    if all(ai_ys[i] >= ai_ys[i + 1] for i in range(len(ai_ys) - 1)):
                        mono = "decreasing"
                    elif all(ai_ys[i] <= ai_ys[i + 1] for i in range(len(ai_ys) - 1)):
                        mono = "increasing"
                traced = trace_curve(
                    color_mask, bounds, axis_range,
                    n_points=300, monotonic=mono,
                    ai_guide=ai_pts,
                )
                if traced and len(traced) >= 5:
                    candidates.append(("trace-color", traced))
            except Exception:
                pass

        # Candidate E: multi-curve trace (same-color spatial separation)
        if multi_trace_curves is not None and si < len(multi_trace_curves):
            mt_traced = multi_trace_curves[si]
            if mt_traced and len(mt_traced) >= 5:
                candidates.append(("multi-trace", mt_traced))

        # --- Step 3: Assess each candidate using this series' color mask ---
        if not candidates:
            # No candidates succeeded — keep AI data as-is
            result.append({"name": series_name, "data": ai_pts})
            continue

        y_full_scale = axis_range.y_max - axis_range.y_min
        mae_tie_threshold = 0.005 * y_full_scale  # 0.5% of y-range

        # Use dark_mask for assessment when per-series mask is too sparse
        assess_mask = color_mask if color_mask.sum() >= 30 else dark_mask

        scored: list[tuple[str, list[dict], float, int, int]] = []
        for method_name, cand_pts in candidates:
            wrapper = [{"name": series_name, "data": cand_pts}]
            acc = assess_extraction_accuracy(
                img_rgb, wrapper, bounds, axis_range,
                per_series_masks=[assess_mask],
            )
            sr = acc["series"][0] if acc["series"] else None
            if sr is None:
                continue
            mae = sr["mae"]
            n_assessed = sr["n_points"]
            if n_assessed < 2:
                mae = float("inf")
            scored.append((method_name, cand_pts, mae, n_assessed, len(cand_pts)))

        if not scored:
            result.append({"name": series_name, "data": ai_pts})
            continue

        # Pick best MAE, then prefer dense candidates within tie threshold
        scored.sort(key=lambda t: t[2])  # sort by MAE
        best_method, best_data, best_mae, _, _ = scored[0]
        for method_name, cand_pts, mae, n_assessed, n_data in scored[1:]:
            if (mae - best_mae <= mae_tie_threshold
                    and n_data >= 5 * len(best_data)):
                best_method = method_name
                best_data = cand_pts
                best_mae = mae
                break

        # Ensure curves start correctly near (x_min, y_max).
        # Multi-trace can't separate converging curves near x=0, so the
        # initial y-values are often wrong.  Find the first trace point
        # that roughly matches the AI guide, trim everything before it,
        # and prepend a dense pixel-snapped prefix for the initial segment.
        if best_data and len(ai_pts) >= 2:
            ai_first = ai_pts[0]
            if "x" in ai_first and "y" in ai_first:
                ai_y0 = float(ai_first["y"])
                ext_y0 = float(best_data[0].get("y", 0))
                y_gap = abs(ext_y0 - ai_y0)

                if y_gap > 0.05 * y_full_scale:
                    # Build piecewise-linear AI guide for interpolation
                    ai_xy = [(float(p["x"]), float(p["y"]))
                             for p in ai_pts if "x" in p and "y" in p]
                    # Find first trace point matching AI guide within 15%
                    reliable_idx = 0
                    tol = 0.15 * y_full_scale
                    for idx, pt in enumerate(best_data):
                        px = float(pt.get("x", 0))
                        py = float(pt.get("y", 0))
                        # Interpolate AI y at this x
                        ai_y = None
                        for k in range(len(ai_xy) - 1):
                            if ai_xy[k][0] <= px <= ai_xy[k + 1][0]:
                                dx = ai_xy[k + 1][0] - ai_xy[k][0]
                                frac = (px - ai_xy[k][0]) / dx if dx > 0 else 0
                                ai_y = ai_xy[k][1] * (1 - frac) + ai_xy[k + 1][1] * frac
                                break
                        if ai_y is not None and abs(py - ai_y) < tol:
                            reliable_idx = idx
                            break
                    # Trim unreliable prefix
                    best_data = best_data[reliable_idx:]
                    if best_data:
                        ext_x0 = float(best_data[0].get("x", 0))
                        # Dense initial segment via pixel snapping
                        x_step = ((axis_range.x_max - axis_range.x_min)
                                  / max(1, bounds.width))
                        guide_xs = np.arange(axis_range.x_min, ext_x0, x_step)
                        if len(guide_xs) == 0:
                            guide_xs = np.array([axis_range.x_min])
                        ai_x_arr = np.array([p[0] for p in ai_xy])
                        ai_y_arr = np.array([p[1] for p in ai_xy])
                        guide_ys = np.interp(guide_xs, ai_x_arr, ai_y_arr)
                        dense_guide = [
                            {"x": round(float(x), 4), "y": round(float(y), 4)}
                            for x, y in zip(guide_xs, guide_ys)
                        ]
                        prefix_mask = color_mask if use_bw_path else dark_mask
                        snapped_prefix = _snap_series_on_mask(
                            prefix_mask, dense_guide, bounds, axis_range,
                            search_radius=20, apply_median=False,
                        )
                        # Ensure origin included
                        if (not snapped_prefix
                                or abs(float(snapped_prefix[0]["x"])
                                       - axis_range.x_min) > 0.01):
                            snapped_prefix = [
                                {"x": round(axis_range.x_min, 4),
                                 "y": round(axis_range.y_max, 4)}
                            ] + snapped_prefix
                        best_data = snapped_prefix + best_data

        result.append({"name": series_name, "data": best_data})

    return result



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
    bounds_override: PlotBounds | None = None,
) -> bytes:
    """Render extracted curves overlaid on the original chart image.

    Args:
        image_bytes: Original chart image bytes.
        data_series: List of {"name": str, "data": [{"x", "y"}, ...]} dicts.
        axis_range: Data-coordinate axis ranges.
        bounds_override: Pre-computed plot bounds (skips auto-detection).

    Returns:
        PNG image bytes of the overlay figure.
    """
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    bounds = bounds_override or detect_plot_bounds(img)

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
