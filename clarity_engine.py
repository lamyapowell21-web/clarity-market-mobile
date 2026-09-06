
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


def load_config(path: str | Path = "cloud_config.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def daily_opportunity(df: pd.DataFrame) -> int | None:
    if df is None or df.empty or len(df) < 65:
        return None

    d = df.copy()
    close = pd.to_numeric(d["Close"], errors="coerce")
    volume = pd.to_numeric(d["Volume"], errors="coerce")
    avg20 = close.rolling(20).mean()
    avg50 = close.rolling(50).mean()
    ret20 = close.pct_change(20)
    ret60 = close.pct_change(60)

    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    volavg = volume.rolling(20).mean()

    if pd.isna(avg50.iloc[-1]) or pd.isna(rsi.iloc[-1]):
        return None

    price = float(close.iloc[-1])
    score = 0.0
    score += 18 if price > avg20.iloc[-1] else 4
    score += 18 if avg20.iloc[-1] > avg50.iloc[-1] else 5
    score += 12 if ret20.iloc[-1] > 0 else 2
    score += 12 if ret60.iloc[-1] > 0 else 2

    vr = 1.0
    if pd.notna(volavg.iloc[-1]) and float(volavg.iloc[-1]) > 0:
        vr = float(volume.iloc[-1] / volavg.iloc[-1])
    score += 15 if vr >= 1.5 else 11 if vr >= 1.1 else 7

    rv = float(rsi.iloc[-1])
    score += 18 if 45 <= rv <= 68 else 10 if 35 <= rv <= 75 else 4
    score += 7 if len(avg20) >= 6 and avg20.iloc[-1] > avg20.iloc[-6] else 2

    return int(round(max(0, min(100, score))))


def intraday_quick(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> dict | None:
    if df is None or df.empty or len(df) < 30:
        return None

    d = df.copy()
    for c in ["High", "Low", "Close", "Volume"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["High", "Low", "Close"])
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

    market = 5
    if spy_df is not None and not spy_df.empty and "Close" in spy_df:
        sp = pd.to_numeric(spy_df["Close"], errors="coerce").dropna()
        if len(sp) >= 4:
            spy15 = float(sp.iloc[-1] / sp.iloc[-4] - 1)
            market = 10 if spy15 > 0.001 else 6 if spy15 >= 0 else 2

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

    breakout = 20 if price >= prior_high else 16 if dist <= 1 else 11 if dist <= 2 else 5
    direction = 15 if price > mid and ret60 > 0 else 10 if price > mid else 4

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

    if price >= prior_high:
        reasons.append("Price pushed through a recent short-term ceiling.")
    elif dist <= 1:
        reasons.append("Price is close to a recent short-term ceiling.")

    if market >= 6:
        reasons.append("The wider market is not fighting the move.")

    if ret15 < 0:
        cautions.append("The latest short-term move is downward.")
    if vr < 0.8:
        cautions.append("Trading activity is quieter than usual.")
    if price < mid:
        cautions.append("Price is below its recent short-term average.")

    return {
        "score": score,
        "price": price,
        "bar_range": bar_range,
        "reasons": reasons[:4] or ["Several short-term signals are lining up."],
        "cautions": cautions[:3],
        "timestamp": str(pd.to_datetime(d.index[-1])),
        "volume_ratio": vr,
        "ret15": ret15,
        "ret60": ret60,
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


def market_mood(daily_data: dict[str, pd.DataFrame]) -> dict:
    spy = daily_data.get("SPY", pd.DataFrame())
    vix = daily_data.get("^VIX", pd.DataFrame())

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

        # Give the first 5-minute bar time to finish and stop 10 minutes before close.
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
