"""Rule/statistics-only EGX market regime and next-session outlook.

Consumes already-downloaded Yahoo OHLCV frames. No network calls, model fitting,
model artifacts, or persistent price history.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Callable, Iterable
import math
import numpy as np
import pandas as pd

CAIRO = ZoneInfo("Africa/Cairo")


def _symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            a, b = raw.columns.get_level_values(0), raw.columns.get_level_values(1)
            if symbol in a:
                out = raw[symbol].copy()
            elif symbol in b:
                out = raw.xs(symbol, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            out = raw.copy()
        out.columns = [str(c).title() for c in out.columns]
        needed = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in out.columns]
        if "Close" not in needed:
            return pd.DataFrame()
        out = out[needed].copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].sort_index()
        for c in needed:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["Close"])
        return out[~out.index.duplicated(keep="last")]
    except Exception:
        return pd.DataFrame()


def _complete_daily(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    # Yahoo daily bars are date-labelled. Before 15:00 Cairo, today's bar is not final.
    today = now.date()
    if now.time() < time(15, 0):
        df = df.loc[[x.date() != today for x in idx]]
    return df


def _completed_period_frame(raw: pd.DataFrame, symbol: str, now: datetime, period: str) -> pd.DataFrame:
    """Drop the active weekly/monthly candle; only closed periods confirm trend."""
    frame = _symbol_frame(raw, symbol)
    if frame.empty:
        return frame
    today = now.date()
    idx = pd.DatetimeIndex(frame.index)
    if period == "weekly":
        # EGX trading week starts Sunday; exclude any candle belonging to this week.
        week_start = today - timedelta(days=(today.weekday() + 1) % 7)
        frame = frame.loc[[x.date() < week_start for x in idx]]
    elif period == "monthly":
        frame = frame.loc[[not (x.year == today.year and x.month == today.month) for x in idx]]
    return frame


def _breadth_snapshot(frames: dict[str, pd.DataFrame], date_value=None) -> dict:
    rows = []
    for sym, frame in frames.items():
        d = frame if date_value is None else frame.loc[:date_value]
        if len(d) < 22:
            continue
        c = d["Close"].astype(float)
        last = float(c.iloc[-1])
        prev = float(c.iloc[-2])
        if not np.isfinite(last) or last <= 0 or not np.isfinite(prev) or prev <= 0:
            continue
        e20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        r1 = last / prev - 1
        r5 = last / c.iloc[-6] - 1 if len(c) >= 6 and c.iloc[-6] else np.nan
        r20 = last / c.iloc[-21] - 1 if len(c) >= 21 and c.iloc[-21] else np.nan
        vol = d["Volume"].astype(float) if "Volume" in d else pd.Series(dtype=float)
        relvol = np.nan
        if len(vol) >= 21 and vol.iloc[-21:-1].mean() > 0:
            relvol = float(vol.iloc[-1] / vol.iloc[-21:-1].mean())
        realized_vol20 = float(c.pct_change().tail(20).std() * np.sqrt(20)) if len(c) >= 22 else np.nan
        atr_pct = np.nan
        if all(col in d.columns for col in ("High", "Low")):
            high = d["High"].astype(float); low = d["Low"].astype(float)
            prev_close = c.shift(1)
            true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            atr_pct = float(atr / last * 100) if pd.notna(atr) and last > 0 else np.nan
        rows.append({"symbol": sym, "r1": r1, "r5": r5, "r20": r20,
                     "above20": last > e20, "above50": last > e50,
                     "relvol": relvol, "realized_vol20": realized_vol20, "atr_pct": atr_pct})
    if not rows:
        return {"count": 0}
    x = pd.DataFrame(rows)
    adv = int((x.r1 > 0).sum()); dec = int((x.r1 < 0).sum())
    pulse = 100 * (0.30 * (x.r1.clip(-.05, .05).mean() / .05 + 1) / 2
                   + 0.30 * x.above20.mean()
                   + 0.20 * x.above50.mean()
                   + 0.20 * ((x.r5 > 0).mean()))
    pulse = float(np.clip(pulse, 0, 100))
    return {"count": int(len(x)), "adv": adv, "dec": dec,
            "unchanged": int(len(x) - adv - dec),
            "adv_pct": float((x.r1 > 0).mean() * 100),
            "above20_pct": float(x.above20.mean() * 100),
            "above50_pct": float(x.above50.mean() * 100),
            "r1_mean": float(x.r1.mean()), "r5_mean": float(x.r5.mean()),
            "r20_mean": float(x.r20.mean()), "relvol_median": float(x.relvol.median()) if x.relvol.notna().any() else np.nan,
            "realized_vol20_median": float(x.realized_vol20.median()) if x.realized_vol20.notna().any() else np.nan,
            "atr_pct_median": float(x.atr_pct.median()) if x.atr_pct.notna().any() else np.nan,
            "pulse": pulse}


def _regime(s: dict) -> str:
    if not s.get("count"):
        return "بيانات غير كافية"
    p = s["pulse"]
    if p >= 75: return "صاعد قوي"
    if p >= 60: return "صاعد"
    if p <= 25: return "هابط قوي"
    if p <= 40: return "هابط"
    if s.get("relvol_median", 1) and s.get("relvol_median", 1) > 1.8 and abs(s.get("r1_mean", 0)) > .025:
        return "متذبذب عالي المخاطر"
    return "عرضي"


def _historical_forecast(frames: dict[str, pd.DataFrame], current_regime: str, horizons=(1, 3, 5)) -> dict:
    # Build equal-weight market close series on dates with enough constituents.
    closes = {}
    for sym, df in frames.items():
        if len(df) >= 30:
            closes[sym] = df["Close"].astype(float)
    if not closes:
        return {}
    panel = pd.DataFrame(closes).sort_index()
    returns = panel.pct_change()
    market_ret = returns.mean(axis=1, skipna=True)
    breadth20 = pd.DataFrame({s: (d.Close > d.Close.ewm(span=20, adjust=False).mean()).astype(float)
                              for s, d in frames.items() if len(d) >= 30}).reindex(panel.index)
    breadth50 = pd.DataFrame({s: (d.Close > d.Close.ewm(span=50, adjust=False).mean()).astype(float)
                              for s, d in frames.items() if len(d) >= 55}).reindex(panel.index)
    adv = (returns > 0).mean(axis=1)
    b20 = breadth20.mean(axis=1)
    b50 = breadth50.mean(axis=1)
    # Historical regime labels use only information available at each date.
    hist_pulse = 100 * (0.30 * ((market_ret.clip(-.05, .05) / .05 + 1) / 2)
                        + 0.30 * b20 + 0.20 * b50 + 0.20 * adv)
    labels = pd.Series("عرضي", index=panel.index)
    labels[hist_pulse >= 75] = "صاعد قوي"; labels[(hist_pulse >= 60) & (hist_pulse < 75)] = "صاعد"
    labels[hist_pulse <= 25] = "هابط قوي"; labels[(hist_pulse <= 40) & (hist_pulse > 25)] = "هابط"
    result = {}
    for h in horizons:
        fwd = (1 + market_ret).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        sample = pd.DataFrame({"regime": labels, "fwd": fwd}).dropna()
        # Exclude last h bars, which have no fully observed forward outcome.
        matched = sample[sample.regime == current_regime]
        # Conservative fallback to all historical dates when regime sample is too small.
        used = matched if len(matched) >= 8 else sample
        vals = used.fwd.dropna()
        if vals.empty:
            result[h] = {"up": np.nan, "flat": np.nan, "down": np.nan, "n": 0, "confidence": 0}
            continue
        up = float((vals > .005).mean()); down = float((vals < -.005).mean()); flat = max(0.0, 1-up-down)
        # Confidence is sample-size sufficiency, not forecast accuracy.
        confidence = min(90.0, 20.0 + 70.0 * (1 - math.exp(-len(vals) / 35.0)))
        if len(matched) < 8: confidence *= 0.70
        result[h] = {"up": up, "flat": flat, "down": down, "n": int(len(vals)),
                     "matched_n": int(len(matched)), "mean_return": float(vals.mean()),
                     "median_return": float(vals.median()), "confidence": round(confidence, 1)}
    return result


def analyze_market(daily_raw: pd.DataFrame, symbols: Iterable[str],
                   weekly_raw: pd.DataFrame | None = None,
                   monthly_raw: pd.DataFrame | None = None,
                   now: datetime | None = None) -> dict:
    now = now or datetime.now(CAIRO)
    if now.tzinfo is None:
        now = now.replace(tzinfo=CAIRO)
    frames = {}
    for symbol in symbols:
        d = _complete_daily(_symbol_frame(daily_raw, symbol), now)
        if len(d) >= 22:
            frames[symbol] = d
    snapshot = _breadth_snapshot(frames)
    regime = _regime(snapshot)
    forecasts = _historical_forecast(frames, regime)
    # Multi-timeframe trend alignment is a descriptive cross-check only.
    symbols = list(symbols)
    def tf_breadth(raw, period):
        if raw is None or raw.empty: return None
        above = total = 0
        for sym in symbols:
            x = _completed_period_frame(raw, sym, now, period)
            if len(x) < 22: continue
            c = x.Close.astype(float); ema = c.ewm(span=20, adjust=False).mean().iloc[-1]
            above += int(c.iloc[-1] > ema); total += 1
        return {"above20": above, "count": total, "pct": above / total * 100 if total else np.nan}
    weekly = tf_breadth(weekly_raw, "weekly"); monthly = tf_breadth(monthly_raw, "monthly")
    last_market_date = max((d.index[-1].date() for d in frames.values() if not d.empty), default=None)
    next_session_label = "الجلسة القادمة"
    if last_market_date is not None:
        # EGX's regular trading days are Sunday through Thursday; holidays are not embedded.
        next_day = (last_market_date.weekday() + 1) % 7
        if next_day == 4:  # Friday -> Sunday
            next_day = 6
        elif next_day == 5:  # Saturday -> Sunday
            next_day = 6
        day_names = {6: "الأحد", 0: "الاثنين", 1: "الثلاثاء", 2: "الأربعاء", 3: "الخميس"}
        next_session_label = f"{day_names.get(next_day, 'الجلسة القادمة')} المتوقع (ما لم توجد عطلة)"
    reasons = []
    if snapshot.get("count"):
        if snapshot["above20_pct"] >= 60: reasons.append("أغلب الأسهم أعلى EMA20")
        elif snapshot["above20_pct"] <= 40: reasons.append("أغلب الأسهم أسفل EMA20")
        if snapshot["adv_pct"] >= 60: reasons.append("اتساع إيجابي: الصاعدين أكثر من الهابطين")
        elif snapshot["adv_pct"] <= 40: reasons.append("اتساع سلبي: الهابطين أكثر من الصاعدين")
        if snapshot.get("r20_mean", 0) > 0: reasons.append("متوسط عائد 20 جلسة موجب")
        elif snapshot.get("r20_mean", 0) < 0: reasons.append("متوسط عائد 20 جلسة سالب")
    return {"as_of": now.isoformat(), "last_market_date": str(last_market_date) if last_market_date else None,
            "next_session_label": next_session_label, "regime": regime, "snapshot": snapshot,
            "forecasts": forecasts, "weekly": weekly, "monthly": monthly,
            "reasons": reasons, "symbols_used": len(frames),
            "symbols_requested": len(list(symbols)),
            "note": "التوقع إحصائي مشروط بحالات تاريخية مشابهة، وليس ضمانًا. أفق 1 يعني الجلسة التالية المتاحة؛ لا يوجد تقويم إجازات رسمي مدمج."}
