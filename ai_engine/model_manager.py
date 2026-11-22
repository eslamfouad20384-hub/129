"""Model-only persistence for EGX Market Intelligence.

Only fitted estimators and compact metadata are written. OHLCV candles and
training rows are never persisted by this module.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import joblib
import pandas as pd

DEFAULT_MODEL_DIR = os.environ.get("EGX_AI_MODEL_DIR", "models/market_intelligence")
MODEL_SCHEMA_VERSION = 2


def _safe_frame_hash(frame: pd.DataFrame | None) -> str:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return "empty"
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    if not cols:
        return "no_ohlcv"
    d = frame[cols].copy()
    d = d.apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    # Hash index and values; only the digest is retained, never the candles.
    try:
        values = pd.util.hash_pandas_object(d, index=True).values.tobytes()
    except Exception:
        values = repr((list(d.index.astype(str)), d.to_dict("list"))).encode("utf-8")
    return hashlib.sha256(values).hexdigest()


def data_fingerprint(symbols, daily_raw, weekly_raw, monthly_raw) -> str:
    h = hashlib.sha256()
    for symbol in sorted(map(str, symbols)):
        h.update(symbol.encode("utf-8"))
        for raw in (daily_raw, weekly_raw, monthly_raw):
            frame = None
            if isinstance(raw, dict):
                frame = raw.get(symbol)
            elif isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                try:
                    if symbol in raw.columns.get_level_values(0):
                        frame = raw[symbol]
                    elif symbol in raw.columns.get_level_values(1):
                        frame = raw.xs(symbol, axis=1, level=1)
                except Exception:
                    frame = None
            elif isinstance(raw, pd.DataFrame):
                frame = raw
            h.update(_safe_frame_hash(frame).encode("ascii"))
    return h.hexdigest()


def model_path(horizon: int, model_dir: str | None = None) -> Path:
    return Path(model_dir or DEFAULT_MODEL_DIR) / f"market_model_{int(horizon)}d.joblib"


def load_model(horizon: int, fingerprint: str, feature_version: str, model_dir: str | None = None):
    path = model_path(horizon, model_dir)
    if not path.exists():
        return None, None
    try:
        payload = joblib.load(path)
        meta = payload.get("metadata", {})
        if meta.get("schema_version") != MODEL_SCHEMA_VERSION:
            return None, None
        if meta.get("fingerprint") != fingerprint or meta.get("feature_version") != feature_version:
            return None, None
        return payload.get("model"), meta
    except Exception:
        return None, None


def save_model(horizon: int, model: Any, fingerprint: str, feature_version: str,
               metrics: dict, model_dir: str | None = None) -> str | None:
    if model is None:
        return None
    path = model_path(horizon, model_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "feature_version": feature_version,
            "horizon": int(horizon),
            "saved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (str, int, float, bool)) or v is None},
            "storage_policy": "model_and_metadata_only_no_ohlcv_or_training_rows",
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump({"model": model, "metadata": metadata}, tmp, compress=3)
        tmp.replace(path)
        return str(path)
    except Exception:
        return None
