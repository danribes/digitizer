import io
import json

import pandas as pd


def _series_to_dataframe(series: dict) -> pd.DataFrame:
    """Convert a single data series dict to a DataFrame."""
    df = pd.DataFrame(series["data"])
    df.insert(0, "series", series["name"])
    return df


def _build_combined_df(data: dict) -> pd.DataFrame:
    """Combine all series into one DataFrame."""
    frames = [_series_to_dataframe(s) for s in data["data_series"]]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def to_csv(data: dict) -> str:
    """Return extracted data as CSV text."""
    return _build_combined_df(data).to_csv(index=False)


def to_json(data: dict) -> str:
    """Return extracted data as pretty-printed JSON."""
    return json.dumps(data, indent=2)


def to_excel(data: dict) -> bytes:
    """Return an Excel workbook with one sheet per series."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for series in data["data_series"]:
            sheet_name = series["name"][:31]  # Excel sheet names max 31 chars
            df = pd.DataFrame(series["data"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


def to_python(data: dict) -> str:
    """Return a Python/pandas code snippet that recreates the data."""
    lines = ["import pandas as pd", ""]

    for i, series in enumerate(data["data_series"]):
        var = f"df_{i}" if len(data["data_series"]) > 1 else "df"
        lines.append(f"# {series['name']}")
        lines.append(f"{var} = pd.DataFrame({json.dumps(series['data'], indent=2)})")
        lines.append("")

    if len(data["data_series"]) > 1:
        names = [f"df_{i}" for i in range(len(data["data_series"]))]
        lines.append("# Combined")
        lines.append(f"df = pd.concat([{', '.join(names)}], keys={json.dumps([s['name'] for s in data['data_series']])})")
        lines.append("")

    return "\n".join(lines)
