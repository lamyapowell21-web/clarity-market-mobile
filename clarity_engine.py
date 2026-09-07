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
    "ret15",
    "ret60m",
    "acceleration",
    "log_intraday_volume_ratio",
    "breakout_distance",
    "above_mid",
    "market_ret15",
    "range_pct",
    "minutes_from_open",
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


def get_many(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    tickers = list(dict.fromkeys([str(t).upper().strip() for t in tickers if str(t).strip()]))
    if not tickers:
        return {}

    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            repair=True,
            progress=False,
            threads=True,
            group_by="ticker",
            multi_level_index=True,
            prepost=False,
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


def intraday_features(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> dict | None:
    if df is None or df.empty or len(df) < 30:
        return None

    d = _numeric_frame(df).dropna(subset=["High", "Low", "Close"])
    if len(d) < 30:
        return None

    close = d["Close"]
    volume = d["Volume"].fillna(0) if "Volume" in d else pd.Series(0, index=d.index)
    price = float(close.iloc[-1])

    ret15 = float(price / close.iloc[-4] - 1)
    ret60 = float(price / close.iloc[-13] - 1)
    prior15 = float(close.iloc[-4] / close.iloc[-7] - 1)
    acceleration = ret15 - prior15

    recent_vol = float(volume.tail(3).mean())
    normal_vol = float(volume.iloc[-27:-3].mean())
    vr = recent_vol / normal_vol if normal_vol > 0 else 1.0

    prior_high = float(d["High"].iloc[-21:-1].max())
    mid = float(close.tail(20).mean())
    bar_range = float((d["High"] - d["Low"]).tail(12).mean())
    bar_range = max(bar_range, price * 0.001)
    dist = (prior_high - price) / bar_range

    market_ret15 = 0.0
    if spy_df is not None and not spy_df.empty and "Close" in spy_df:
        sp = pd.to_numeric(spy_df["Close"], errors="coerce").dropna()
        if len(sp) >= 4:
            market_ret15 = float(sp.iloc[-1] / sp.iloc[-4] - 1)

    return {
        "price": price,
        "bar_range": bar_range,
        "ret15": ret15,
        "ret60m": ret60,
        "acceleration": acceleration,
        "intraday_volume_ratio": float(vr),
        "log_intraday_volume_ratio": float(math.log(max(vr, 0.05))),
        "breakout_distance": float(dist),
        "above_mid": 1.0 if price > mid else 0.0,
        "market_ret15": market_ret15,
        "range_pct": float(bar_range / price),
        "minutes_from_open": _minutes_from_open(d.index[-1]),
        "prior_high": prior_high,
        "mid": mid,
        "timestamp": str(pd.to_datetime(d.index[-1])),
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
        "ret15": ifeat["ret15"],
        "ret60m": ifeat["ret60m"],
        "acceleration": ifeat["acceleration"],
        "log_intraday_volume_ratio": ifeat["log_intraday_volume_ratio"],
        "breakout_distance": ifeat["breakout_distance"],
        "above_mid": ifeat["above_mid"],
        "market_ret15": ifeat["market_ret15"],
        "range_pct": ifeat["range_pct"],
        "minutes_from_open": ifeat["minutes_from_open"],
    }
    if any(not np.isfinite(float(row[k])) for k in MODEL_FEATURES):
        return None
    row["price"] = ifeat["price"]
    row["bar_range"] = ifeat["bar_range"]
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


def calibrated_signal(daily_df: pd.DataFrame, intraday_df: pd.DataFrame, spy_df: pd.DataFrame, calibration: dict) -> dict:
    row = model_feature_row(daily_df, intraday_df, spy_df)
    if row is None:
        return {"available": False, "ready": False, "reason": "Not enough usable market history."}

    prob = predict_calibrated_probability(row, calibration)
    if prob is None:
        return {
            "available": False,
            "ready": False,
            "reason": "Historical calibration has not finished yet.",
            "features": row,
        }

    threshold = float(calibration.get("probability_threshold", 1.0))
    require_up = bool(calibration.get("guardrails", {}).get("require_positive_ret15", True))
    directional_ok = (row["ret15"] > 0) if require_up else True
    model_ready = bool(calibration.get("ready_for_alerts", False))
    ready = model_ready and directional_ok and prob >= threshold

    margin = prob - threshold
    if not model_ready:
        grade = "Calibration not strong enough yet"
    elif margin >= 0.12:
        grade = "Strong historical fit"
    elif margin >= 0.04:
        grade = "Good historical fit"
    elif margin >= 0:
        grade = "Just cleared the historical line"
    else:
        grade = "Below the historical line"

    return {
        "available": True,
        "ready": ready,
        "probability": prob,
        "threshold": threshold,
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
