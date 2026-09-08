from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

NY = ZoneInfo("America/New_York")

MODEL_FEATURES = [
    "daily_ret20",
    "daily_ret60",
    "price_vs_ma20",
    "ma20_vs_ma50",
    "rsi14",
    "daily_volume_ratio",
    "ret15_volnorm",
    "ret60_volnorm",
    "acceleration_volnorm",
    "log_intraday_volume_ratio",
    "volume_acceleration",
    "breakout_distance",
    "vwap_distance",
    "opening_range_position",
    "opening_range_breakout_distance",
    "opening_range_ready",
    "session_return_volnorm",
    "market_ret15",
    "market_ret60m",
    "relative_ret15_volnorm",
    "relative_ret60_volnorm",
    "same_slot_ret15_mean_volnorm",
    "market_range_pct",
    "range_pct",
    "early_session",
    "late_session",
]


def load_config(path: str | Path = "cloud_config.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_calibration(path: str | Path = "clarity_calibration.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {"status": "missing", "ready_for_alerts": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Calibration is not a JSON object")
        return data
    except Exception as exc:
        return {"status": "invalid", "ready_for_alerts": False, "error": str(exc)}


def get_many(tickers: list[str], period: str | None, interval: str, start=None, end=None) -> dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys([str(t).upper().strip() for t in tickers if str(t).strip()]))
    if not tickers:
        return {}

    try:
        intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        kwargs = {
            "interval": interval,
            "auto_adjust": True,
            # Repair can trigger many slower fine-grained follow-up downloads (for
            # example 2-minute requests while calibrating 5-minute history). It is
            # useful for daily history, but counterproductive for the large intraday
            # calibration pull.
            "repair": interval not in intraday_intervals,
            "progress": False,
            "threads": True,
            "group_by": "ticker",
            "multi_level_index": True,
            "prepost": False,
        }
        if start is not None or end is not None:
            if start is not None:
                kwargs["start"] = start
            if end is not None:
                kwargs["end"] = end
        else:
            kwargs["period"] = period or "1mo"

        raw = yf.download(
            tickers,
            **kwargs,

        )
    except Exception:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return result

    if len(tickers) == 1:
        ticker = tickers[0]
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker in raw.columns.get_level_values(0):
                    part = raw[ticker].copy()
                else:
                    part = raw.copy()
                    part.columns = part.columns.get_level_values(-1)
            else:
                part = raw.copy()
            result[ticker] = part.dropna(how="all")
        except Exception:
            pass
        return result

    for ticker in tickers:
        try:
            if not isinstance(raw.columns, pd.MultiIndex):
                continue
            if ticker in raw.columns.get_level_values(0):
                part = raw[ticker].copy()
            elif ticker in raw.columns.get_level_values(1):
                part = raw.xs(ticker, axis=1, level=1).copy()
            else:
                continue
            part = part.dropna(how="all")
            if not part.empty:
                result[ticker] = part
        except Exception:
            continue
    return result


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def completed_daily_history(df: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Return only daily bars completed before the as-of trading date.

    During market hours Yahoo can include a partial current-day daily candle. Using it
    as if it were complete would leak information in historical tests and distort the
    live score, so Clarity deliberately excludes the current date.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    d = _numeric_frame(df).dropna(subset=["Close"])
    if d.empty:
        return d

    if as_of is None:
        cutoff_date = datetime.now(NY).date()
    else:
        ts = pd.Timestamp(as_of)
        if ts.tzinfo is None:
            cutoff_date = ts.date()
        else:
            cutoff_date = ts.tz_convert(NY).date()

    idx_dates = pd.to_datetime(d.index).date
    return d.loc[idx_dates < cutoff_date].copy()


def daily_features(df: pd.DataFrame) -> dict | None:
    if df is None or df.empty or len(df) < 65:
        return None

    d = _numeric_frame(df).dropna(subset=["Close"])
    if len(d) < 65:
        return None

    close = d["Close"]
    volume = d["Volume"].fillna(0) if "Volume" in d else pd.Series(0, index=d.index)
    avg20 = close.rolling(20).mean()
    avg50 = close.rolling(50).mean()
    ret20 = close.pct_change(20)
    ret60 = close.pct_change(60)

    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    safe_loss = loss.replace(0, np.nan)
    rs = gain / safe_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
    rsi = rsi.where(~((gain == 0) & (loss > 0)), 0.0)
    rsi = rsi.where(~((gain == 0) & (loss == 0)), 50.0)
    volavg = volume.rolling(20).mean()

    vals = [avg20.iloc[-1], avg50.iloc[-1], ret20.iloc[-1], ret60.iloc[-1], rsi.iloc[-1]]
    if any(pd.isna(v) for v in vals):
        return None

    price = float(close.iloc[-1])
    ma20 = float(avg20.iloc[-1])
    ma50 = float(avg50.iloc[-1])
    vr = 1.0
    if pd.notna(volavg.iloc[-1]) and float(volavg.iloc[-1]) > 0:
        vr = float(volume.iloc[-1] / volavg.iloc[-1])

    return {
        "daily_ret20": float(ret20.iloc[-1]),
        "daily_ret60": float(ret60.iloc[-1]),
        "price_vs_ma20": float(price / ma20 - 1.0),
        "ma20_vs_ma50": float(ma20 / ma50 - 1.0),
        "rsi14": float(rsi.iloc[-1]),
        "daily_volume_ratio": float(vr),
        "price": price,
        "ma20": ma20,
        "ma50": ma50,
    }


def daily_opportunity(df: pd.DataFrame) -> int | None:
    """Legacy human-readable score. The evidence model does NOT train on this score."""
    feat = daily_features(df)
    if feat is None:
        return None

    score = 0.0
    score += 18 if feat["price_vs_ma20"] > 0 else 4
    score += 18 if feat["ma20_vs_ma50"] > 0 else 5
    score += 12 if feat["daily_ret20"] > 0 else 2
    score += 12 if feat["daily_ret60"] > 0 else 2

    vr = feat["daily_volume_ratio"]
    score += 15 if vr >= 1.5 else 11 if vr >= 1.1 else 7

    rv = feat["rsi14"]
    score += 18 if 45 <= rv <= 68 else 10 if 35 <= rv <= 75 else 4
    score += 7 if feat["price_vs_ma20"] > 0 else 2
    return int(round(max(0, min(100, score))))


def daily_opportunity_completed(df: pd.DataFrame, as_of=None) -> int | None:
    return daily_opportunity(completed_daily_history(df, as_of=as_of))


def _minutes_from_open(ts) -> float:
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize(NY)
        else:
            t = t.tz_convert(NY)
        return float((t.hour * 60 + t.minute) - (9 * 60 + 30))
    except Exception:
        return 0.0


def _normalize_intraday_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = _numeric_frame(df)
    idx = pd.to_datetime(d.index)
    if idx.tz is None:
        idx = idx.tz_localize(NY)
    else:
        idx = idx.tz_convert(NY)
    d.index = idx
    return d.sort_index()


def _same_slot_prior_ret15(prior: pd.DataFrame, slot: float, sessions: int = 4) -> float:
    if prior is None or prior.empty or "Close" not in prior:
        return 0.0
    vals = []
    for _, s in prior.groupby(prior.index.date, sort=True):
        s = s.sort_index()
        slots = np.asarray([_minutes_from_open(x) for x in s.index], dtype=float)
        hits = np.where(np.abs(slots - float(slot)) <= 2.5)[0]
        if len(hits) == 0:
            continue
        j = int(hits[-1])
        if j < 3:
            continue
        c = pd.to_numeric(s["Close"], errors="coerce")
        if pd.isna(c.iloc[j]) or pd.isna(c.iloc[j - 3]) or float(c.iloc[j - 3]) == 0:
            continue
        vals.append(float(c.iloc[j] / c.iloc[j - 3] - 1.0))
    return float(np.mean(vals[-sessions:])) if vals else 0.0


def _volume_ratio_for_bars(session: pd.DataFrame, prior: pd.DataFrame, positions: list[int], sessions: int = 4) -> float:
    if not positions or "Volume" not in session:
        return 1.0
    cur = pd.to_numeric(session["Volume"], errors="coerce").fillna(0)
    numer = float(cur.iloc[positions].mean()) if positions else 0.0
    baselines = []
    if prior is not None and not prior.empty and "Volume" in prior:
        pv = pd.to_numeric(prior["Volume"], errors="coerce")
        pslots = np.asarray([_minutes_from_open(x) for x in prior.index], dtype=float)
        for pos in positions:
            slot = _minutes_from_open(session.index[pos])
            hits = np.where(np.abs(pslots - slot) <= 2.5)[0]
            same = pv.iloc[hits].dropna().tail(sessions)
            if len(same) >= 2 and float(same.mean()) > 0:
                baselines.append(float(same.mean()))
    if baselines:
        denom = float(np.mean(baselines))
    elif prior is not None and not prior.empty and "Volume" in prior:
        denom = float(pd.to_numeric(prior["Volume"], errors="coerce").dropna().tail(24).mean())
    else:
        denom = 0.0
    return float(numer / denom) if denom > 0 else 1.0


def intraday_features(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> dict | None:
    """v13 market-structure-aware intraday features, using only information available at the signal time."""
    if df is None or df.empty:
        return None

    d = _normalize_intraday_index(df).dropna(subset=["High", "Low", "Close"])
    if d.empty:
        return None

    current_ts = d.index[-1]
    current_date = current_ts.date()
    session = d[d.index.date == current_date].copy().sort_index()
    if len(session) < 4:
        return None
    prior = d[d.index.date < current_date].copy()

    close = pd.to_numeric(session["Close"], errors="coerce")
    high = pd.to_numeric(session["High"], errors="coerce")
    low = pd.to_numeric(session["Low"], errors="coerce")
    volume = pd.to_numeric(session["Volume"], errors="coerce").fillna(0) if "Volume" in session else pd.Series(0.0, index=session.index)
    if close.isna().iloc[-1]:
        return None
    price = float(close.iloc[-1])
    if price <= 0:
        return None

    j = len(session) - 1
    ret15 = float(price / close.iloc[j - 3] - 1.0) if float(close.iloc[j - 3]) != 0 else 0.0
    if j >= 12 and float(close.iloc[j - 12]) != 0:
        ret60 = float(price / close.iloc[j - 12] - 1.0)
    else:
        ret60 = float(price / close.iloc[0] - 1.0) if float(close.iloc[0]) != 0 else 0.0
    if j >= 6 and float(close.iloc[j - 6]) != 0:
        prior15 = float(close.iloc[j - 3] / close.iloc[j - 6] - 1.0)
        acceleration = ret15 - prior15
    else:
        acceleration = 0.0

    ranges = (high - low).tail(12)
    bar_range = float(ranges.mean()) if len(ranges) else price * 0.001
    bar_range = max(bar_range, price * 0.001)
    range_pct = float(bar_range / price)
    vol_scale = max(range_pct, 0.0005)

    recent_positions = list(range(max(0, j - 2), j + 1))
    previous_positions = list(range(max(0, j - 5), max(0, j - 2)))
    vr = _volume_ratio_for_bars(session, prior, recent_positions)
    prior_vr = _volume_ratio_for_bars(session, prior, previous_positions) if previous_positions else 1.0
    log_vr = float(math.log(max(vr, 0.05)))
    volume_acceleration = float(log_vr - math.log(max(prior_vr, 0.05)))

    if len(session) >= 6:
        prev_highs = high.iloc[max(0, j - 20):j]
    else:
        prev_highs = pd.to_numeric(d["High"], errors="coerce").iloc[max(0, len(d) - 21):len(d) - 1]
    prior_high = float(prev_highs.max()) if len(prev_highs) and prev_highs.notna().any() else price
    breakout_distance = float((prior_high - price) / bar_range)

    typical = (high + low + close) / 3.0
    cum_vol = float(volume.sum())
    if cum_vol > 0:
        vwap = float((typical * volume).sum() / cum_vol)
    else:
        vwap = float(close.mean())
    vwap_distance = float(np.clip((price - vwap) / bar_range, -12.0, 12.0))

    if len(session) >= 6:
        opening_high = float(high.iloc[:6].max())
        opening_low = float(low.iloc[:6].min())
        opening_span = max(opening_high - opening_low, bar_range * 0.5)
        opening_range_position = float(np.clip((price - opening_low) / opening_span, -2.0, 3.0))
        opening_range_breakout_distance = float(np.clip((opening_high - price) / bar_range, -12.0, 12.0))
        opening_range_ready = 1.0
    else:
        opening_range_position = 0.5
        opening_range_breakout_distance = 0.0
        opening_range_ready = 0.0

    session_open = float(pd.to_numeric(session["Open"], errors="coerce").iloc[0]) if "Open" in session else float(close.iloc[0])
    session_return = float(price / session_open - 1.0) if session_open > 0 else 0.0

    market_ret15 = 0.0
    market_ret60 = 0.0
    market_range_pct = range_pct
    if spy_df is not None and not spy_df.empty:
        sp = _normalize_intraday_index(spy_df).dropna(subset=["High", "Low", "Close"])
        sp = sp[sp.index <= current_ts]
        sp_session = sp[sp.index.date == current_date].copy().sort_index()
        if len(sp_session) >= 4:
            sc = pd.to_numeric(sp_session["Close"], errors="coerce")
            sj = len(sc) - 1
            market_ret15 = float(sc.iloc[sj] / sc.iloc[sj - 3] - 1.0) if float(sc.iloc[sj - 3]) != 0 else 0.0
            if sj >= 12 and float(sc.iloc[sj - 12]) != 0:
                market_ret60 = float(sc.iloc[sj] / sc.iloc[sj - 12] - 1.0)
            else:
                market_ret60 = float(sc.iloc[sj] / sc.iloc[0] - 1.0) if float(sc.iloc[0]) != 0 else 0.0
            sr = (pd.to_numeric(sp_session["High"], errors="coerce") - pd.to_numeric(sp_session["Low"], errors="coerce")).tail(12).mean()
            if pd.notna(sr) and float(sc.iloc[-1]) > 0:
                market_range_pct = float(max(float(sr) / float(sc.iloc[-1]), 0.0002))

    slot = _minutes_from_open(current_ts)
    same_slot_ret15 = _same_slot_prior_ret15(prior, slot, sessions=4)
    mid = float(close.tail(20).mean())

    return {
        "price": price,
        "bar_range": bar_range,
        "ret15": ret15,
        "ret60m": ret60,
        "acceleration": acceleration,
        "ret15_volnorm": float(np.clip(ret15 / vol_scale, -12.0, 12.0)),
        "ret60_volnorm": float(np.clip(ret60 / vol_scale, -20.0, 20.0)),
        "acceleration_volnorm": float(np.clip(acceleration / vol_scale, -12.0, 12.0)),
        "intraday_volume_ratio": float(vr),
        "log_intraday_volume_ratio": log_vr,
        "volume_acceleration": float(np.clip(volume_acceleration, -4.0, 4.0)),
        "breakout_distance": float(np.clip(breakout_distance, -12.0, 12.0)),
        "above_mid": 1.0 if price > mid else 0.0,
        "vwap": vwap,
        "vwap_distance": vwap_distance,
        "above_vwap": 1.0 if price >= vwap else 0.0,
        "opening_range_position": opening_range_position,
        "opening_range_breakout_distance": opening_range_breakout_distance,
        "opening_range_ready": opening_range_ready,
        "session_return_volnorm": float(np.clip(session_return / vol_scale, -25.0, 25.0)),
        "market_ret15": market_ret15,
        "market_ret60m": market_ret60,
        "relative_ret15_volnorm": float(np.clip((ret15 - market_ret15) / vol_scale, -12.0, 12.0)),
        "relative_ret60_volnorm": float(np.clip((ret60 - market_ret60) / vol_scale, -20.0, 20.0)),
        "same_slot_ret15_mean_volnorm": float(np.clip(same_slot_ret15 / vol_scale, -12.0, 12.0)),
        "market_range_pct": float(market_range_pct),
        "range_pct": range_pct,
        "minutes_from_open": slot,
        "early_session": 1.0 if slot <= 90 else 0.0,
        "late_session": 1.0 if slot >= 270 else 0.0,
        "prior_high": prior_high,
        "mid": mid,
        "timestamp": str(pd.to_datetime(current_ts)),
    }


def intraday_quick(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> dict | None:
    """Legacy 0-100 quick score retained for explanation, not model training."""
    f = intraday_features(df, spy_df)
    if f is None:
        return None

    vr = f["intraday_volume_ratio"]
    ret15 = f["ret15"]
    ret60 = f["ret60m"]
    acceleration = f["acceleration"]
    dist = f["breakout_distance"]
    market_ret15 = f["market_ret15"]

    market = 10 if market_ret15 > 0.001 else 6 if market_ret15 >= 0 else 2
    activity = 25 if vr >= 2 else 21 if vr >= 1.5 else 16 if vr >= 1.15 else 9

    momentum = 0
    if ret15 > 0:
        momentum += 10
    if ret15 >= 0.003:
        momentum += 5
    if ret15 >= 0.007:
        momentum += 4
    if ret60 > 0:
        momentum += 4
    if acceleration > 0:
        momentum += 2
    momentum = min(momentum, 25)

    breakout = 20 if dist <= 0 else 16 if dist <= 1 else 11 if dist <= 2 else 5
    direction = 15 if f["above_mid"] and ret60 > 0 else 10 if f["above_mid"] else 4

    score = activity + momentum + breakout + direction + market
    if ret15 < 0:
        score *= 0.72
    score = int(round(max(0, min(100, score))))

    reasons = []
    cautions = []
    if vr >= 1.5:
        reasons.append("Trading activity jumped well above normal.")
    elif vr >= 1.15:
        reasons.append("Trading activity picked up.")
    if ret15 > 0.003 and acceleration > 0:
        reasons.append("Price is rising and speeding up.")
    elif ret15 > 0:
        reasons.append("The latest short-term moves point upward.")
    if dist <= 0:
        reasons.append("Price pushed through a recent short-term ceiling.")
    elif dist <= 1:
        reasons.append("Price is close to a recent short-term ceiling.")
    if market >= 6:
        reasons.append("The wider market is not fighting the move.")
    if ret15 < 0:
        cautions.append("The latest short-term move is downward.")
    if vr < 0.8:
        cautions.append("Trading activity is quieter than usual.")
    if not f["above_mid"]:
        cautions.append("Price is below its recent short-term average.")

    return {
        "score": score,
        "price": f["price"],
        "bar_range": f["bar_range"],
        "reasons": reasons[:4] or ["Several short-term signals are lining up."],
        "cautions": cautions[:3],
        "timestamp": f["timestamp"],
        "volume_ratio": vr,
        "ret15": ret15,
        "ret60": ret60,
        "acceleration": acceleration,
        "breakout_distance": dist,
        "market_ret15": market_ret15,
        "range_pct": f["range_pct"],
        "minutes_from_open": f["minutes_from_open"],
        "above_mid": f["above_mid"],
    }


def model_feature_row(daily_df: pd.DataFrame, intraday_df: pd.DataFrame, spy_df: pd.DataFrame | None = None, as_of=None) -> dict | None:
    when = as_of if as_of is not None else (intraday_df.index[-1] if intraday_df is not None and not intraday_df.empty else None)
    dfeat = daily_features(completed_daily_history(daily_df, as_of=when))
    if dfeat is None:
        return None
    ifeat = intraday_features(intraday_df, spy_df)
    if ifeat is None:
        return None

    row = {
        "daily_ret20": dfeat["daily_ret20"],
        "daily_ret60": dfeat["daily_ret60"],
        "price_vs_ma20": dfeat["price_vs_ma20"],
        "ma20_vs_ma50": dfeat["ma20_vs_ma50"],
        "rsi14": dfeat["rsi14"],
        "daily_volume_ratio": dfeat["daily_volume_ratio"],
        "ret15_volnorm": ifeat["ret15_volnorm"],
        "ret60_volnorm": ifeat["ret60_volnorm"],
        "acceleration_volnorm": ifeat["acceleration_volnorm"],
        "log_intraday_volume_ratio": ifeat["log_intraday_volume_ratio"],
        "volume_acceleration": ifeat["volume_acceleration"],
        "breakout_distance": ifeat["breakout_distance"],
        "vwap_distance": ifeat["vwap_distance"],
        "opening_range_position": ifeat["opening_range_position"],
        "opening_range_breakout_distance": ifeat["opening_range_breakout_distance"],
        "opening_range_ready": ifeat["opening_range_ready"],
        "session_return_volnorm": ifeat["session_return_volnorm"],
        "market_ret15": ifeat["market_ret15"],
        "market_ret60m": ifeat["market_ret60m"],
        "relative_ret15_volnorm": ifeat["relative_ret15_volnorm"],
        "relative_ret60_volnorm": ifeat["relative_ret60_volnorm"],
        "same_slot_ret15_mean_volnorm": ifeat["same_slot_ret15_mean_volnorm"],
        "market_range_pct": ifeat["market_range_pct"],
        "range_pct": ifeat["range_pct"],
        "early_session": ifeat["early_session"],
        "late_session": ifeat["late_session"],
    }
    if any(not np.isfinite(float(row[k])) for k in MODEL_FEATURES):
        return None
    row["price"] = ifeat["price"]
    row["bar_range"] = ifeat["bar_range"]
    row["ret15"] = ifeat["ret15"]
    row["ret60m"] = ifeat["ret60m"]
    row["timestamp"] = ifeat["timestamp"]
    return row


def predict_calibrated_probability(feature_row: dict, calibration: dict) -> float | None:
    if not feature_row or not calibration or calibration.get("status") not in {"ready", "weak"}:
        return None
    try:
        names = calibration["feature_names"]
        means = np.asarray(calibration["scaler_mean"], dtype=float)
        scales = np.asarray(calibration["scaler_scale"], dtype=float)
        coefs = np.asarray(calibration["coefficients"], dtype=float)
        intercept = float(calibration["intercept"])
        x = np.asarray([float(feature_row[n]) for n in names], dtype=float)
        scales = np.where(np.abs(scales) < 1e-12, 1.0, scales)
        z = (x - means) / scales
        logit = float(intercept + np.dot(z, coefs))
        logit = max(-35.0, min(35.0, logit))
        return float(1.0 / (1.0 + math.exp(-logit)))
    except Exception:
        return None


def evidence_percentile(probability: float, calibration: dict) -> float | None:
    """Map a model probability to its percentile among untouched historical snapshots."""
    try:
        ref = calibration.get("probability_reference", {})
        values = np.asarray(ref.get("values", []), dtype=float)
        percentiles = np.asarray(ref.get("percentiles", []), dtype=float)
        if len(values) < 2 or len(values) != len(percentiles):
            return None
        # np.interp expects nondecreasing x. Quantiles can tie, so collapse ties.
        unique_values = []
        unique_pct = []
        for v, q in zip(values, percentiles):
            if unique_values and abs(v - unique_values[-1]) < 1e-12:
                unique_pct[-1] = max(unique_pct[-1], q)
            else:
                unique_values.append(float(v))
                unique_pct.append(float(q))
        if len(unique_values) < 2:
            return 50.0
        return float(np.interp(float(probability), unique_values, unique_pct, left=0.0, right=100.0))
    except Exception:
        return None


def calibrated_signal(daily_df: pd.DataFrame, intraday_df: pd.DataFrame, spy_df: pd.DataFrame, calibration: dict) -> dict:
    row = model_feature_row(daily_df, intraday_df, spy_df)
    if row is None:
        return {"available": False, "ready": False, "reason": "Not enough usable market history."}

    if int(calibration.get("version", 0) or 0) < 3:
        return {
            "available": False,
            "ready": False,
            "reason": "Clarity v13 needs a fresh evidence calibration.",
            "features": row,
        }

    prob = predict_calibrated_probability(row, calibration)
    if prob is None:
        return {
            "available": False,
            "ready": False,
            "reason": "Historical calibration has not finished yet.",
            "features": row,
        }

    threshold = float(calibration.get("probability_threshold", 1.0))
    require_up = bool(calibration.get("guardrails", {}).get("require_positive_ret15", False))
    directional_ok = (row["ret15"] > 0) if require_up else True
    model_ready = bool(calibration.get("ready_for_alerts", False))
    ready = model_ready and directional_ok and prob >= threshold

    pctile = evidence_percentile(prob, calibration)
    threshold_pctile = calibration.get("threshold_percentile")
    if threshold_pctile is None:
        threshold_pctile = evidence_percentile(threshold, calibration)

    if not model_ready:
        grade = "Calibration not strong enough yet"
    elif not directional_ok:
        grade = "Short-term direction guardrail not met"
    elif ready and pctile is not None and pctile >= 95:
        grade = "Very strong recent historical evidence"
    elif ready and pctile is not None and pctile >= 90:
        grade = "Strong recent historical evidence"
    elif ready:
        grade = "Cleared the historical evidence line"
    elif pctile is not None and threshold_pctile is not None and pctile >= max(75.0, float(threshold_pctile) - 5.0):
        grade = "Close to the historical evidence line"
    else:
        grade = "Below the historical evidence line"

    return {
        "available": True,
        "ready": ready,
        "probability": prob,
        "threshold": threshold,
        "evidence_percentile": pctile,
        "threshold_percentile": threshold_pctile,
        "grade": grade,
        "features": row,
        "directional_ok": directional_ok,
        "calibration_status": calibration.get("status", "unknown"),
        "test_metrics": calibration.get("metrics", {}).get("test", {}),
    }


def trade_levels(price: float, bar_range: float) -> dict:
    stop_distance = max(2.0 * bar_range, price * 0.0035)
    return {
        "entry_ceiling": price + 0.35 * bar_range,
        "stop": max(0.01, price - stop_distance),
        "target": price + 2.0 * stop_distance,
        "stop_distance": stop_distance,
    }


def dollar_plan(price: float, levels: dict, max_trade_amount: float, max_planned_loss: float) -> dict | None:
    if max_trade_amount <= 0:
        return None
    stop_distance = max(0.000001, float(price) - float(levels["stop"]))
    risk_fraction = stop_distance / float(price)
    amount = float(max_trade_amount)
    if max_planned_loss > 0 and risk_fraction > 0:
        amount = min(amount, float(max_planned_loss) / risk_fraction)
    shares = amount / float(price)
    return {
        "amount": amount,
        "shares": shares,
        "planned_loss": shares * stop_distance,
        "target_profit": shares * (float(levels["target"]) - float(price)),
    }


def evaluate_future_path(future_df: pd.DataFrame, price: float, levels: dict, round_trip_cost_bps: float = 0.0) -> dict | None:
    if future_df is None or future_df.empty:
        return None
    f = _numeric_frame(future_df).dropna(subset=["High", "Low", "Close"])
    if f.empty:
        return None

    stop = float(levels["stop"])
    target = float(levels["target"])
    risk = max(1e-9, price - stop)
    target_hit = False
    stop_hit = False
    ambiguous = False
    exit_price = float(f["Close"].iloc[-1])

    for _, bar in f.iterrows():
        hi = float(bar["High"])
        lo = float(bar["Low"])
        hit_t = hi >= target
        hit_s = lo <= stop
        if hit_t and hit_s:
            ambiguous = True
            break
        if hit_t:
            target_hit = True
            exit_price = target
            break
        if hit_s:
            stop_hit = True
            exit_price = stop
            break

    if ambiguous:
        return {"ambiguous": True}

    gross_r = (exit_price - price) / risk
    cost_r = ((float(round_trip_cost_bps) / 10000.0) * price) / risk if round_trip_cost_bps > 0 else 0.0
    net_r = gross_r - cost_r

    max_up = (float(f["High"].max()) - price) / price
    max_down = (float(f["Low"].min()) - price) / price
    return {
        "ambiguous": False,
        "target_hit": bool(target_hit),
        "stop_hit": bool(stop_hit),
        "gross_r": float(gross_r),
        "net_r": float(net_r),
        "max_upside": float(max_up),
        "max_drawdown": float(max_down),
        "final_return": float(float(f["Close"].iloc[-1]) / price - 1.0),
    }


def market_mood(daily_data: dict[str, pd.DataFrame]) -> dict:
    spy = completed_daily_history(daily_data.get("SPY", pd.DataFrame()))
    vix = completed_daily_history(daily_data.get("^VIX", pd.DataFrame()))
    if spy.empty or len(spy) < 55:
        return {"label": "Unclear", "detail": "There is not enough broad-market data right now."}

    close = pd.to_numeric(spy["Close"], errors="coerce")
    avg20 = close.rolling(20).mean()
    avg50 = close.rolling(50).mean()
    price = float(close.iloc[-1])

    vix_level = None
    if not vix.empty and "Close" in vix:
        vv = pd.to_numeric(vix["Close"], errors="coerce").dropna()
        if not vv.empty:
            vix_level = float(vv.iloc[-1])

    if price > avg20.iloc[-1] > avg50.iloc[-1] and (vix_level is None or vix_level < 25):
        return {"label": "Supportive", "detail": "The wider market is generally helping upward trades."}
    if price < avg20.iloc[-1] < avg50.iloc[-1] or (vix_level is not None and vix_level >= 30):
        return {"label": "Cautious", "detail": "The wider market is under pressure, so new trades deserve extra care."}
    return {"label": "Mixed", "detail": "The wider market is giving a mix of positive and negative signals."}


def session_status(now: datetime | None = None) -> dict:
    now = now or datetime.now(NY)
    nyse = mcal.get_calendar("NYSE")
    start = now.date() - timedelta(days=2)
    end = now.date() + timedelta(days=10)
    schedule = nyse.schedule(start_date=start, end_date=end)
    if schedule.empty:
        return {"open": False, "label": "Market calendar unavailable", "next_open": None}

    now_ts = pd.Timestamp(now)
    today_rows = schedule[schedule.index.date == now.date()]
    if not today_rows.empty:
        market_open = today_rows.iloc[0]["market_open"].tz_convert(NY)
        market_close = today_rows.iloc[0]["market_close"].tz_convert(NY)
        active_start = market_open + pd.Timedelta(minutes=5)
        active_end = market_close - pd.Timedelta(minutes=10)
        if active_start <= now_ts <= active_end:
            return {
                "open": True,
                "label": f"Market open · closes {market_close.strftime('%-I:%M %p ET')}",
                "next_open": None,
                "market_open": market_open,
                "market_close": market_close,
            }
        if now_ts < market_open:
            return {
                "open": False,
                "label": f"Market opens today at {market_open.strftime('%-I:%M %p ET')}",
                "next_open": market_open,
            }

    future = schedule[schedule["market_open"] > now_ts.tz_convert("UTC")]
    if future.empty:
        return {"open": False, "label": "Market closed", "next_open": None}
    nxt = future.iloc[0]["market_open"].tz_convert(NY)
    day = "tomorrow" if nxt.date() == (now.date() + timedelta(days=1)) else nxt.strftime("%A")
    return {
        "open": False,
        "label": f"Market closed · next opens {day} at {nxt.strftime('%-I:%M %p ET')}",
        "next_open": nxt,
    }
