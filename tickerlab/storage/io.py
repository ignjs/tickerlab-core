from __future__ import annotations

from pathlib import Path

import polars as pl


EXTENSIONS = {"parquet": "parquet", "csv": "csv", "json": "json"}


def dataset_path(base: Path, name: str, symbol: str, timeframe: str, export: str = "parquet") -> Path:
    ext = EXTENSIONS[export]
    return base / name / f"symbol={symbol}" / f"timeframe={timeframe}" / f"data.{ext}"


def save_frame(df: pl.DataFrame, path: Path, export: str = "parquet") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if export == "parquet":
        df.write_parquet(path)
    elif export == "csv":
        df.write_csv(path)
    elif export == "json":
        df.write_ndjson(path)
    else:
        raise ValueError(f"Unsupported export format: {export}")
    return path


def save_dataset(base: Path, name: str, symbol: str, timeframe: str, df: pl.DataFrame, export: str = "parquet") -> Path:
    path = dataset_path(base, name, symbol, timeframe, export)
    return save_frame(df, path, export=export)


def load_frame(path: Path) -> pl.DataFrame:

    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if path.suffix == ".json":
        return pl.read_ndjson(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")
