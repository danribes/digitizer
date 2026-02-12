# Plot Digitizer

Extract structured data from chart images using AI vision and pixel-level curve tracing.

Upload a chart screenshot, get back clean data points you can export as CSV, JSON, Excel, or Python code. A chat interface lets you iteratively refine the extraction by describing what needs correction.

## Features

- **Two extraction methods**
  - **AI-only** — Claude Vision analyzes the chart and returns data points directly
  - **Pixel-enhanced (hybrid)** — Claude extracts chart metadata (axis ranges, series colors), then pixel-level color matching traces curves with sub-pixel precision. Falls back to AI-only automatically when pixel tracing isn't viable.

- **Supported chart types** — Line, bar, scatter, pie, polar, heatmap, ternary, and more

- **Multi-curve tracing** — Handles multiple same-color curves differentiated by line style (solid, dashed, dotted) using Hungarian algorithm assignment

- **Interactive refinement** — Chat with Claude to correct extraction errors. Both the original chart and current overlay are sent so Claude can see exactly where the extraction diverges.

- **Export formats** — CSV, JSON, Excel (one sheet per series), Python (runnable pandas code)

## Quick Start

Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### Docker (recommended)

```bash
docker build -t digitizer .
docker run -p 8501:8501 --env-file .env -v ./overlays:/app/overlays digitizer
```

Open http://localhost:8501

The `-v ./overlays:/app/overlays` mount persists overlay images to your host.

### Local

```bash
pip install -r requirements.txt
source .env
streamlit run app.py
```

## Usage

1. Upload a chart image (PNG, JPG, GIF, or WebP)
2. Choose an extraction method and optional chart type hint in the sidebar
3. Click **Extract Data**
4. Review the overlay visualization comparing extracted curves against the original
5. Use the chat input to describe corrections (e.g., "the blue curve should be higher between x=5 and x=10")
6. Export the refined data in your preferred format

## How It Works

### AI-Only Extraction

The chart image is sent to Claude with a structured tool call that returns chart type, axis labels, and data points for each series.

### Pixel-Enhanced Extraction

1. **Metadata extraction** — Claude identifies axis ranges, series names, RGB colors, line styles, and whether the chart is pixel-traceable
2. **Plot bounds detection** — Axis lines are found by scanning for long horizontal/vertical dark pixel runs
3. **Color masking** — Per-series boolean masks isolate pixels matching each series color
4. **Curve tracing** — Column-by-column segment detection with continuity tracking, median filtering, and optional monotonicity enforcement
5. **Multi-curve separation** — When multiple series share a color, bidirectional tracking with Hungarian algorithm assignment separates them, with line-style classification for correct ordering

### Chat Refinement

Each refinement round sends Claude both the original chart image and the current overlay visualization, along with the full conversation history. Claude compares where the extraction diverges from the original and returns corrected data points.

## Project Structure

```
app.py            Streamlit UI
extractor.py      Claude API extraction and refinement
pixel_tracer.py   Pixel-level curve tracing and overlay generation
export.py         CSV, JSON, Excel, and Python export
overlays/         Saved overlay images (gitignored)
Dockerfile        Container configuration
.env              API key (gitignored, create manually)
```

## Requirements

- Python 3.12+
- [Anthropic API key](https://console.anthropic.com/)
- Dependencies: streamlit, anthropic, pandas, Pillow, scipy, matplotlib, openpyxl
