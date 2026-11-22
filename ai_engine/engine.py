"""Leakage-aware multi-timeframe ML engine.

Consumes already-downloaded OHLCV frames from the technical engine. It never
fetches market data and never changes technical/financial/portfolio decisions.
"""
from __future__ import annotations

import warnings
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .model_manager import data_fingerprint, load_model, save_model

HORIZONS = (5, 10, 20)
COST_DEFAULTS = {"commission_pct": 0.15, "slippage_pct": 0.10}
FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "volatility_10d", "volatility_20d",
    "rsi_14", "ema20_gap", "ema50_gap", "ema200_gap", "atr_pct", "volume_ratio",
    "turnover_proxy_z20", "range_pct", "distance_high20", "distance_low20",
    "weekly_ret_1", "weekly_ret_4", "weekly_ret_12", "weekly_ema20_gap",
    "weekly_volatility_8", "monthly_ret_1", "monthly_ret_3", "monthly_ret_6",
    "monthly_ema12_gap", "monthly_volatility_6",
]
CLASS_NAMES = {0: "هبوط", 1: "عرضي", 2: "صعود"}
FEATURE_VERSION = "egx_market_features_v2"


def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(frame.columns):
        return pd.DataFrame()
    out = frame[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Open", "High", "Low", "Close"])
    out = out[(out["Close"] > 0) & (out["High"] >= out["Low"])].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[~out.index.isna()]
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = _clean_ohlcv(frame)
    if d.empty:
        return d
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    f = pd.DataFrame(index=d.index)
    for n in (1, 5, 10, 20):
        f[f"ret_{n}d"] = c.pct_change(n)
    daily_ret = c.pct_change()
    f["volatility_10d"] = daily_ret.rolling(10, min_periods=8).std() * np.sqrt(252)
    f["volatility_20d"] = daily_ret.rolling(20, min_periods=15).std() * np.sqrt(252)
    f["rsi_14"] = _rsi(c)
    for n in (20, 50, 200):
        ema = c.ewm(span=n, adjust=False, min_periods=min(n, 30 if n == 200 else n)).mean()
        f[f"ema{n}_gap"] = c / ema - 1
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / c
    vol_ma = v.rolling(20, min_periods=10).mean()
    f["volume_ratio"] = v / vol_ma.replace(0, np.nan)
    turnover = c * v
    turnover_mean = turnover.rolling(20, min_periods=10).mean()
    turnover_std = turnover.rolling(20, min_periods=10).std()
    f["turnover_proxy_z20"] = (turnover - turnover_mean) / turnover_std.replace(0, np.nan)
    f["range_pct"] = (h-l) / c
    f["distance_high20"] = c / h.rolling(20, min_periods=10).max() - 1
    f["distance_low20"] = c / l.rolling(20, min_periods=10).min() - 1
    return f.replace([np.inf, -np.inf], np.nan)


def _period_features(frame: pd.DataFrame, prefix: str, periods: Iterable[int], ema_period: int,
                     vol_period: int, availability_days: int) -> pd.DataFrame:
    d = _clean_ohlcv(frame)
    if d.empty:
        return pd.DataFrame()
    c = d.Close
    f = pd.DataFrame(index=d.index)
    for n in periods:
        f[f"{prefix}_ret_{n}"] = c.pct_change(n)
    ema = c.ewm(span=ema_period, adjust=False, min_periods=min(ema_period, 6)).mean()
    f[f"{prefix}_ema{ema_period}_gap"] = c / ema - 1
    f[f"{prefix}_volatility_{vol_period}"] = c.pct_change().rolling(vol_period, min_periods=max(3, vol_period//2)).std()
    # Yahoo weekly/monthly index labels may identify the start of the period.
    # Delay availability until the period is safely complete to prevent look-ahead.
    f = f.copy()
    f.index = pd.DatetimeIndex(f.index) + pd.to_timedelta(availability_days, unit="D")
    return f.replace([np.inf, -np.inf], np.nan)


def build_symbol_features(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    d = _clean_ohlcv(daily)
    if len(d) < 260:
        return pd.DataFrame()
    f = _daily_features(d)
    w = _period_features(weekly, "weekly", (1, 4, 12), 20, 8, 6)
    m = _period_features(monthly, "monthly", (1, 3, 6), 12, 6, 32)
    if not w.empty:
        f = pd.merge_asof(f.sort_index(), w.sort_index(), left_index=True, right_index=True, direction="backward")
    else:
        for c in ("weekly_ret_1", "weekly_ret_4", "weekly_ret_12", "weekly_ema20_gap", "weekly_volatility_8"):
            f[c] = np.nan
    if not m.empty:
        f = pd.merge_asof(f.sort_index(), m.sort_index(), left_index=True, right_index=True, direction="backward")
    else:
        for c in ("monthly_ret_1", "monthly_ret_3", "monthly_ret_6", "monthly_ema12_gap", "monthly_volatility_6"):
            f[c] = np.nan
    return f.reindex(columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)


def make_labeled_rows(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame,
                      horizon: int, round_trip_cost_pct: float) -> pd.DataFrame:
    d = _clean_ohlcv(daily)
    f = build_symbol_features(daily, weekly, monthly)
    if d.empty or f.empty:
        return pd.DataFrame()
    close = d.Close.reindex(f.index)
    future_return = close.shift(-horizon) / close - 1.0
    net_return = future_return - round_trip_cost_pct / 100.0
    # Neutral band is deliberately wider than estimated trading costs.
    y = pd.Series(1, index=f.index, dtype="int64")
    y.loc[net_return < -0.01] = 0
    y.loc[net_return > 0.01] = 2
    out = f.copy()
    out["target"] = y
    out["forward_net_return"] = net_return
    out["symbol"] = str(symbol)
    out["sample_date"] = out.index
    # The last horizon observations have no known outcome and are excluded.
    out = out.loc[net_return.notna()]
    # Do not train on rows with no usable daily features or no multi-timeframe context.
    out = out.dropna(subset=["ret_1d", "ret_5d", "rsi_14"])
    return out


def _make_model(kind: str):
    if kind == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500, class_weight="balanced", C=0.5, random_state=42)),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestClassifier(
            n_estimators=70, max_depth=8, min_samples_leaf=18,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42
        )),
    ])


def _fit_horizon(dataset: pd.DataFrame, horizon: int) -> Tuple[Optional[Pipeline], dict, pd.DataFrame]:
    result = {"horizon": horizon, "status": "insufficient", "samples": int(len(dataset))}
    if len(dataset) < 900 or dataset["target"].nunique() < 3:
        result["message"] = "العينات أو الفئات المتاحة غير كافية لتدريب موثوق."
        return None, result, pd.DataFrame()
    data = dataset.sort_values("sample_date").copy()
    unique_dates = np.array(sorted(pd.to_datetime(data.sample_date).unique()))
    if len(unique_dates) < 80:
        result["message"] = "عدد التواريخ المستقلة غير كافٍ للتقسيم الزمني."
        return None, result, pd.DataFrame()
    split1 = int(len(unique_dates) * 0.70)
    split2 = int(len(unique_dates) * 0.85)
    cut1 = unique_dates[split1]
    cut2 = unique_dates[split2]
    # Purge the final horizon observations before each boundary so their
    # forward-return labels cannot overlap the next chronological segment.
    train_end = unique_dates[max(0, split1 - horizon)]
    valid_end = unique_dates[max(split1, split2 - horizon)]
    train = data[data.sample_date < train_end]
    valid = data[(data.sample_date >= cut1) & (data.sample_date < valid_end)]
    test = data[data.sample_date >= cut2]
    if min(len(train), len(valid), len(test)) < 120 or train.target.nunique() < 3:
        result["message"] = "تعذر تكوين مجموعات Train/Validation/Test زمنية كافية."
        return None, result, pd.DataFrame()
    Xtr, ytr = train[FEATURE_COLUMNS], train.target
    Xv, yv = valid[FEATURE_COLUMNS], valid.target
    candidates = []
    for kind in ("logistic", "forest"):
        model = _make_model(kind)
        try:
            model.fit(Xtr, ytr)
            pred = model.predict(Xv)
            score = balanced_accuracy_score(yv, pred)
            candidates.append((score, kind, model))
        except Exception as exc:
            warnings.warn(f"{kind} model failed: {exc}")
    if not candidates:
        result["message"] = "فشل تدريب النماذج المرجعية."
        return None, result, pd.DataFrame()
    _, chosen, _ = max(candidates, key=lambda x: x[0])
    final_model = _make_model(chosen)
    train_valid = pd.concat([train, valid], axis=0)
    final_model.fit(train_valid[FEATURE_COLUMNS], train_valid.target)
    test_pred = final_model.predict(test[FEATURE_COLUMNS])
    test_prob = final_model.predict_proba(test[FEATURE_COLUMNS])
    classes = list(final_model.named_steps["model"].classes_)
    aligned = np.zeros((len(test), 3), dtype=float)
    for j, cls in enumerate(classes):
        aligned[:, int(cls)] = test_prob[:, j]
    try:
        ll = float(log_loss(test.target, aligned, labels=[0, 1, 2]))
    except Exception:
        ll = np.nan
    result.update({
        "status": "trained", "model": chosen, "train_samples": int(len(train)),
        "validation_samples": int(len(valid)), "test_samples": int(len(test)),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(test.target, test_pred)), 4),
        "test_accuracy": round(float(accuracy_score(test.target, test_pred)), 4),
        "test_macro_f1": round(float(f1_score(test.target, test_pred, average="macro", zero_division=0)), 4),
        "test_log_loss": round(ll, 4) if np.isfinite(ll) else None,
        "train_end": str(pd.Timestamp(train.sample_date.max()).date()),
        "validation_end": str(pd.Timestamp(valid.sample_date.max()).date()),
        "test_end": str(pd.Timestamp(test.sample_date.max()).date()),
        "message": "تقييم زمني خارج العينة؛ لا يمثل ضمانًا للأداء المستقبلي.",
    })
    return final_model, result, test[["symbol", "sample_date", "target", "forward_net_return"]].assign(predicted=test_pred)


def _extract_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol in raw.columns.get_level_values(0):
                return raw[symbol].copy()
            if symbol in raw.columns.get_level_values(1):
                return raw.xs(symbol, axis=1, level=1).copy()
            return pd.DataFrame()
        return raw.copy()
    except Exception:
        return pd.DataFrame()


def run_ai_analysis(symbols: Iterable[str], daily_raw: pd.DataFrame, weekly_raw: pd.DataFrame,
                    monthly_raw: pd.DataFrame, commission_pct: float = 0.15,
                    slippage_pct: float = 0.10, model_dir: str | None = None) -> dict:
    """Hybrid ML/statistical learning using existing OHLCV only; persist fitted models, never candles."""
    round_trip_cost = 2.0 * (float(commission_pct) + float(slippage_pct))
    per_symbol: Dict[str, dict] = {}
    datasets = {h: [] for h in HORIZONS}
    latest_features = []
    for symbol in symbols:
        d = _extract_frame(daily_raw, symbol)
        w = _extract_frame(weekly_raw, symbol)
        m = _extract_frame(monthly_raw, symbol)
        clean_d = _clean_ohlcv(d)
        # Cairo session safety: never use today's still-forming daily candle before 15:00.
        try:
            cairo_now = datetime.now(ZoneInfo("Africa/Cairo"))
            if not clean_d.empty and pd.Timestamp(clean_d.index[-1]).date() == cairo_now.date() and cairo_now.time() < dt_time(15, 0):
                clean_d = clean_d.iloc[:-1]
        except Exception:
            pass
        if len(clean_d) < 260:
            continue
        feat = build_symbol_features(clean_d, w, m)
        if feat.empty:
            continue
        latest = feat.iloc[[-1]].copy()
        latest["symbol"] = symbol
        latest["sample_date"] = latest.index
        latest_features.append(latest)
        per_symbol[symbol] = {"rows": len(clean_d), "last_date": str(pd.Timestamp(clean_d.index[-1]).date())}
        for h in HORIZONS:
            rows = make_labeled_rows(symbol, clean_d, w, m, h, round_trip_cost)
            if not rows.empty:
                datasets[h].append(rows)
    models, metrics, tests, effective_training_rows = {}, {}, {}, {}
    fingerprint = data_fingerprint(symbols, daily_raw, weekly_raw, monthly_raw)
    model_sources = {}
    for h in HORIZONS:
        ds = pd.concat(datasets[h], ignore_index=False) if datasets[h] else pd.DataFrame()
        if len(ds) > 140000:
            # Deterministic cap preserves broad date coverage and bounded memory.
            ds = ds.sort_values("sample_date").groupby("symbol", group_keys=False).tail(500)
        effective_training_rows[h] = int(len(ds))
        cached_model, cached_meta = load_model(h, fingerprint, FEATURE_VERSION, model_dir=model_dir)
        if cached_model is not None:
            model = cached_model
            metric = dict(cached_meta.get("metrics", {}))
            metric.update({"horizon": h, "status": "loaded", "samples": int(len(ds)),
                           "message": "تم تحميل النموذج المحفوظ؛ بصمة الشموع لم تتغير.",
                           "model": metric.get("model", "cached")})
            test_rows = pd.DataFrame()
            model_sources[h] = "cached_model"
        else:
            model, metric, test_rows = _fit_horizon(ds, h)
            model_sources[h] = "retrained" if model is not None else "not_trained"
            if model is not None:
                saved = save_model(h, model, fingerprint, FEATURE_VERSION, metric, model_dir=model_dir)
                metric["model_saved"] = bool(saved)
        models[h], metrics[h], tests[h] = model, metric, test_rows
    prediction_rows = []
    latest_df = pd.concat(latest_features, axis=0) if latest_features else pd.DataFrame()
    for _, row in latest_df.iterrows():
        rec = {"السهم": str(row["symbol"]), "تاريخ آخر شمعة": str(pd.Timestamp(row["sample_date"]).date())}
        valid_horizons = 0
        for h in HORIZONS:
            model = models.get(h)
            metric = metrics.get(h, {})
            if model is None or metric.get("status") != "trained":
                rec[f"توقع {h} جلسة"] = "غير متاح"
                rec[f"صعود {h} جلسة %"] = np.nan
                rec[f"عرضي {h} جلسة %"] = np.nan
                rec[f"هبوط {h} جلسة %"] = np.nan
                continue
            try:
                p = model.predict_proba(pd.DataFrame([row[FEATURE_COLUMNS]], columns=FEATURE_COLUMNS))[0]
                cls = list(model.named_steps["model"].classes_)
                ml_prob = {int(c): float(p[i]) for i, c in enumerate(cls)}
                # Statistical layer: smoothed historical class prior, blended conservatively
                # with ML probabilities. It is computed from the current training sample only.
                ds_for_prior = pd.concat(datasets[h], ignore_index=False) if datasets[h] else pd.DataFrame()
                if not ds_for_prior.empty and "target" in ds_for_prior:
                    counts = ds_for_prior.target.value_counts().reindex([0, 1, 2], fill_value=0).astype(float)
                    prior = ((counts + 2.0) / (counts.sum() + 6.0)).to_dict()
                else:
                    prior = {0: 1/3, 1: 1/3, 2: 1/3}
                prob = {c: 0.85 * ml_prob.get(c, 0.0) + 0.15 * float(prior.get(c, 1/3)) for c in (0, 1, 2)}
                total_prob = sum(prob.values()) or 1.0
                prob = {c: v / total_prob for c, v in prob.items()}
                up, side, down = prob.get(2, 0.0), prob.get(1, 0.0), prob.get(0, 0.0)
                rec[f"توقع {h} جلسة"] = CLASS_NAMES[max(prob, key=prob.get)]
                rec[f"صعود {h} جلسة %"] = round(up * 100, 1)
                rec[f"عرضي {h} جلسة %"] = round(side * 100, 1)
                rec[f"هبوط {h} جلسة %"] = round(down * 100, 1)
                valid_horizons += 1
            except Exception:
                rec[f"توقع {h} جلسة"] = "خطأ توقع"
                rec[f"صعود {h} جلسة %"] = np.nan
                rec[f"عرضي {h} جلسة %"] = np.nan
                rec[f"هبوط {h} جلسة %"] = np.nan
        rec["عدد الآفاق المدربة"] = valid_horizons
        prediction_rows.append(rec)
    predictions = pd.DataFrame(prediction_rows)
    if not predictions.empty:
        predictions["متوسط احتمال الصعود %"] = predictions[[f"صعود {h} جلسة %" for h in HORIZONS]].mean(axis=1)
        predictions = predictions.sort_values("متوسط احتمال الصعود %", ascending=False, na_position="last")
    metric_rows = []
    for h in HORIZONS:
        m = metrics[h].copy()
        m.pop("message", None)
        m["الأفق"] = f"{h} جلسة"
        metric_rows.append(m)
    return {
        "predictions": predictions,
        "metrics": metrics,
        "metrics_table": pd.DataFrame(metric_rows),
        "models": models,
        "test_rows": tests,
        "symbol_coverage": pd.DataFrame([{"السهم": s, **v} for s, v in per_symbol.items()]),
        "training_rows": effective_training_rows,
        "round_trip_cost_pct": round_trip_cost,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_sources": model_sources,
        "model_fingerprint": fingerprint,
        "learning_mode": "hybrid_ml_statistical",
        "persistence_policy": "models_only_no_candle_database",
    }
