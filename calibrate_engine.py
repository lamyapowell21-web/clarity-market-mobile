from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from clarity_engine import (
    MODEL_FEATURES,
    completed_daily_history,
    daily_features,
    get_many,
    load_config,
    trade_levels,
)

NY = ZoneInfo("America/New_York")
BASE = Path(__file__).parent


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_batched(tickers: list[str], period: str | None, interval: str, batch_size: int = 10, start=None, end=None) -> dict[str, pd.DataFrame]:
    out = {}
    total_batches = math.ceil(len(tickers) / max(1, batch_size))
    for batch_no, batch in enumerate(batched(tickers, batch_size), start=1):
        print(f"Downloading {interval} batch {batch_no}/{total_batches}: {', '.join(batch)}", flush=True)
        t0 = time.perf_counter()
        part = get_many(batch, period, interval, start=start, end=end)
        out.update(part)
        elapsed = time.perf_counter() - t0
        missing = [t for t in batch if t not in part or part[t] is None or part[t].empty]
        suffix = f" · {elapsed:.1f}s"
        if missing:
            suffix += f" · no usable data: {', '.join(missing)}"
        print(f"Finished {interval} batch {batch_no}/{total_batches}{suffix}", flush=True)
        time.sleep(0.25)
    return out


def normalize_intraday(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    idx = pd.to_datetime(d.index)
    if idx.tz is None:
        idx = idx.tz_localize(NY)
    else:
        idx = idx.tz_convert(NY)
    d.index = idx
    return d.sort_index()


def _minutes_from_open(ts) -> float:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(NY)
    else:
        t = t.tz_convert(NY)
    return float((t.hour * 60 + t.minute) - (9 * 60 + 30))


def _market_feature_lookup(spy_df: pd.DataFrame) -> dict[pd.Timestamp, dict]:
    spy = normalize_intraday(spy_df)
    if spy.empty or "Close" not in spy:
        return {}
    for c in ["High", "Low", "Close"]:
        if c in spy:
            spy[c] = pd.to_numeric(spy[c], errors="coerce")
    out: dict[pd.Timestamp, dict] = {}
    for _, session in spy.groupby(spy.index.date, sort=True):
        session = session.sort_index()
        close = session["Close"].to_numpy(dtype=float)
        high = session["High"].to_numpy(dtype=float)
        low = session["Low"].to_numpy(dtype=float)
        for j, ts in enumerate(session.index):
            if j >= 3 and close[j - 3] != 0:
                r15 = float(close[j] / close[j - 3] - 1.0)
            else:
                r15 = 0.0
            if j >= 12 and close[j - 12] != 0:
                r60 = float(close[j] / close[j - 12] - 1.0)
            elif close[0] != 0:
                r60 = float(close[j] / close[0] - 1.0)
            else:
                r60 = 0.0
            ranges = high[max(0, j - 11):j + 1] - low[max(0, j - 11):j + 1]
            br = float(np.nanmean(ranges)) if len(ranges) else 0.0
            rpct = float(max(br / close[j], 0.0002)) if np.isfinite(close[j]) and close[j] > 0 else 0.001
            out[ts] = {"market_ret15": r15, "market_ret60m": r60, "market_range_pct": rpct}
    return out


def _daily_feature_cache(daily_df: pd.DataFrame, session_dates: list) -> dict:
    cache = {}
    for day in session_dates:
        when = pd.Timestamp(day).tz_localize(NY)
        feat = daily_features(completed_daily_history(daily_df, as_of=when))
        if feat is not None:
            cache[day] = feat
    return cache


def _intraday_feature_frame(intra_df: pd.DataFrame, spy_df: pd.DataFrame, same_slot_sessions: int = 4) -> pd.DataFrame:
    """Build v13 features once per bar without leaking future bars or future sessions."""
    d = normalize_intraday(intra_df)
    if d.empty:
        return pd.DataFrame()
    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in d.columns]
    d = d[needed].copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[c for c in ["High", "Low", "Close"] if c in d.columns])
    if d.empty:
        return pd.DataFrame()

    market_lookup = _market_feature_lookup(spy_df)
    slot_volume_history: dict[int, list[float]] = {}
    slot_ret15_history: dict[int, list[float]] = {}
    prior_global_volumes: list[float] = []
    prior_global_highs: list[float] = []
    rows = []

    for day, session in d.groupby(d.index.date, sort=True):
        session = session.sort_index()
        close = session["Close"].to_numpy(dtype=float)
        high = session["High"].to_numpy(dtype=float)
        low = session["Low"].to_numpy(dtype=float)
        opn = session["Open"].to_numpy(dtype=float) if "Open" in session else close.copy()
        volume = session["Volume"].fillna(0).to_numpy(dtype=float) if "Volume" in session else np.zeros(len(session), dtype=float)
        times = list(session.index)
        slots = [int(round(_minutes_from_open(ts))) for ts in times]

        slot_baselines = []
        slot_counts = []
        slot_ret_means = []
        for slot in slots:
            vh = slot_volume_history.get(slot, [])
            rh = slot_ret15_history.get(slot, [])
            slot_baselines.append(float(np.mean(vh[-same_slot_sessions:])) if vh else np.nan)
            slot_counts.append(len(vh))
            slot_ret_means.append(float(np.mean(rh[-same_slot_sessions:])) if rh else 0.0)

        typical = (high + low + close) / 3.0
        cum_pv = np.cumsum(typical * volume)
        cum_v = np.cumsum(volume)

        def volume_ratio(positions: list[int]) -> float:
            if not positions:
                return 1.0
            numer = float(np.mean(volume[positions]))
            bases = [slot_baselines[k] for k in positions if slot_counts[k] >= 2 and np.isfinite(slot_baselines[k]) and slot_baselines[k] > 0]
            if bases:
                denom = float(np.mean(bases))
            elif prior_global_volumes:
                denom = float(np.mean(prior_global_volumes[-24:]))
            else:
                denom = 0.0
            return float(numer / denom) if denom > 0 else 1.0

        todays_ret15_by_slot: dict[int, float] = {}

        for j, ts in enumerate(times):
            if j < 3:
                continue
            price = float(close[j])
            if not np.isfinite(price) or price <= 0:
                continue

            ret15 = float(price / close[j - 3] - 1.0) if close[j - 3] != 0 else 0.0
            if j >= 12 and close[j - 12] != 0:
                ret60 = float(price / close[j - 12] - 1.0)
            else:
                ret60 = float(price / close[0] - 1.0) if close[0] != 0 else 0.0
            if j >= 6 and close[j - 6] != 0:
                prior15 = float(close[j - 3] / close[j - 6] - 1.0)
                acceleration = ret15 - prior15
            else:
                acceleration = 0.0

            ranges = high[max(0, j - 11):j + 1] - low[max(0, j - 11):j + 1]
            bar_range = float(np.nanmean(ranges)) if len(ranges) else price * 0.001
            bar_range = max(bar_range, price * 0.001)
            range_pct = float(bar_range / price)
            vol_scale = max(range_pct, 0.0005)

            recent_positions = list(range(max(0, j - 2), j + 1))
            previous_positions = list(range(max(0, j - 5), max(0, j - 2)))
            vr = volume_ratio(recent_positions)
            prior_vr = volume_ratio(previous_positions) if previous_positions else 1.0
            log_vr = float(math.log(max(vr, 0.05)))
            volume_acceleration = float(log_vr - math.log(max(prior_vr, 0.05)))

            if j + 1 >= 6:
                prev_highs = high[max(0, j - 20):j]
            else:
                prev_highs = np.asarray((prior_global_highs + high[:j].tolist())[-20:], dtype=float)
            prior_high = float(np.nanmax(prev_highs)) if len(prev_highs) and np.isfinite(prev_highs).any() else price
            breakout_distance = float((prior_high - price) / bar_range)

            if cum_v[j] > 0:
                vwap = float(cum_pv[j] / cum_v[j])
            else:
                vwap = float(np.mean(close[:j + 1]))
            vwap_distance = float(np.clip((price - vwap) / bar_range, -12.0, 12.0))

            if j >= 5:
                opening_high = float(np.nanmax(high[:6]))
                opening_low = float(np.nanmin(low[:6]))
                opening_span = max(opening_high - opening_low, bar_range * 0.5)
                opening_range_position = float(np.clip((price - opening_low) / opening_span, -2.0, 3.0))
                opening_range_breakout_distance = float(np.clip((opening_high - price) / bar_range, -12.0, 12.0))
                opening_range_ready = 1.0
            else:
                opening_range_position = 0.5
                opening_range_breakout_distance = 0.0
                opening_range_ready = 0.0

            session_open = float(opn[0]) if np.isfinite(opn[0]) and opn[0] > 0 else float(close[0])
            session_return = float(price / session_open - 1.0) if session_open > 0 else 0.0

            market = market_lookup.get(ts, {})
            market_ret15 = float(market.get("market_ret15", 0.0))
            market_ret60 = float(market.get("market_ret60m", 0.0))
            market_range_pct = float(market.get("market_range_pct", range_pct))
            same_slot_ret15 = float(slot_ret_means[j])
            slot = float(slots[j])

            rows.append({
                "timestamp": ts,
                "session_date": day,
                "price": price,
                "bar_range": bar_range,
                "ret15": ret15,
                "ret60m": ret60,
                "ret15_volnorm": float(np.clip(ret15 / vol_scale, -12.0, 12.0)),
                "ret60_volnorm": float(np.clip(ret60 / vol_scale, -20.0, 20.0)),
                "acceleration_volnorm": float(np.clip(acceleration / vol_scale, -12.0, 12.0)),
                "log_intraday_volume_ratio": log_vr,
                "volume_acceleration": float(np.clip(volume_acceleration, -4.0, 4.0)),
                "breakout_distance": float(np.clip(breakout_distance, -12.0, 12.0)),
                "vwap_distance": vwap_distance,
                "opening_range_position": opening_range_position,
                "opening_range_breakout_distance": opening_range_breakout_distance,
                "opening_range_ready": opening_range_ready,
                "session_return_volnorm": float(np.clip(session_return / vol_scale, -25.0, 25.0)),
                "market_ret15": market_ret15,
                "market_ret60m": market_ret60,
                "relative_ret15_volnorm": float(np.clip((ret15 - market_ret15) / vol_scale, -12.0, 12.0)),
                "relative_ret60_volnorm": float(np.clip((ret60 - market_ret60) / vol_scale, -20.0, 20.0)),
                "same_slot_ret15_mean_volnorm": float(np.clip(same_slot_ret15 / vol_scale, -12.0, 12.0)),
                "market_range_pct": market_range_pct,
                "range_pct": range_pct,
                "minutes_from_open": slot,
                "early_session": 1.0 if slot <= 90 else 0.0,
                "late_session": 1.0 if slot >= 270 else 0.0,
            })
            todays_ret15_by_slot[slots[j]] = ret15

        # Only after the session is fully processed can today enter any baseline.
        for slot, vol in zip(slots, volume):
            if np.isfinite(vol):
                hist = slot_volume_history.setdefault(slot, [])
                hist.append(float(vol))
                if len(hist) > same_slot_sessions:
                    del hist[:-same_slot_sessions]
        for slot, r15 in todays_ret15_by_slot.items():
            hist = slot_ret15_history.setdefault(slot, [])
            hist.append(float(r15))
            if len(hist) > same_slot_sessions:
                del hist[:-same_slot_sessions]
        prior_global_volumes.extend([float(x) for x in volume if np.isfinite(x)])
        if len(prior_global_volumes) > 24:
            prior_global_volumes = prior_global_volumes[-24:]
        prior_global_highs.extend([float(x) for x in high if np.isfinite(x)])
        if len(prior_global_highs) > 20:
            prior_global_highs = prior_global_highs[-20:]

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def _evaluate_future_arrays(high: np.ndarray, low: np.ndarray, close: np.ndarray, price: float, levels: dict, base_cost_bps: float, stress_cost_bps: float) -> dict | None:
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        return None
    stop = float(levels["stop"])
    target = float(levels["target"])
    risk = max(1e-9, float(price) - stop)

    hit_t = np.asarray(high, dtype=float) >= target
    hit_s = np.asarray(low, dtype=float) <= stop
    any_hit = hit_t | hit_s
    target_hit = False
    stop_hit = False
    exit_price = float(close[-1])

    if bool(np.any(any_hit)):
        k = int(np.argmax(any_hit))
        if bool(hit_t[k] and hit_s[k]):
            return {"ambiguous": True}
        if bool(hit_t[k]):
            target_hit = True
            exit_price = target
        else:
            stop_hit = True
            exit_price = stop

    gross_r = (exit_price - float(price)) / risk
    base_cost_r = ((float(base_cost_bps) / 10000.0) * float(price)) / risk if base_cost_bps > 0 else 0.0
    stress_cost_r = ((float(stress_cost_bps) / 10000.0) * float(price)) / risk if stress_cost_bps > 0 else 0.0
    return {
        "ambiguous": False,
        "target_hit": target_hit,
        "stop_hit": stop_hit,
        "net_r_base": float(gross_r - base_cost_r),
        "net_r_stress": float(gross_r - stress_cost_r),
        "final_return": float(float(close[-1]) / float(price) - 1.0),
    }


def build_samples(ticker: str, daily_df: pd.DataFrame, intra_df: pd.DataFrame, spy_df: pd.DataFrame, cfg: dict) -> list[dict]:
    intra = normalize_intraday(intra_df)
    if intra.empty or len(intra) < 45:
        return []

    same_slot_sessions = int(cfg.get("same_slot_history_sessions", 4))
    features = _intraday_feature_frame(intra, spy_df, same_slot_sessions=same_slot_sessions)
    if features.empty:
        return []

    first_allowed_ts = intra.index[30] if len(intra) > 30 else intra.index[-1]
    stride = max(1, int(cfg.get("calibration_stride_bars", 1)))
    horizons = sorted(set(int(x) for x in cfg.get("backtest_horizon_candidates_minutes", [15, 30, 60])))
    horizon_bars = {h: max(1, h // 5) for h in horizons}
    max_bars = max(horizon_bars.values())
    start_minute = int(cfg.get("calibration_start_minute_et", 15))
    end_minute = int(cfg.get("calibration_end_minute_et", 350))
    base_cost_bps = float(cfg.get("backtest_round_trip_cost_bps", 5.0))
    stress_cost_bps = float(cfg.get("backtest_stress_round_trip_cost_bps", 15.0))

    session_dates = sorted(set(intra.index.date))
    daily_cache = _daily_feature_cache(daily_df, session_dates)
    rows = []

    for day, session in intra.groupby(intra.index.date, sort=True):
        session = session.sort_index()
        dfeat = daily_cache.get(day)
        if dfeat is None:
            continue
        hi_all = session["High"].to_numpy(dtype=float)
        lo_all = session["Low"].to_numpy(dtype=float)
        cl_all = session["Close"].to_numpy(dtype=float)

        for j in range(0, len(session) - max_bars, stride):
            ts = session.index[j]
            if ts < first_allowed_ts or ts not in features.index:
                continue
            f = features.loc[ts]
            minute = float(f["minutes_from_open"])
            if minute < start_minute or minute > end_minute:
                continue

            feature = {
                "daily_ret20": float(dfeat["daily_ret20"]),
                "daily_ret60": float(dfeat["daily_ret60"]),
                "price_vs_ma20": float(dfeat["price_vs_ma20"]),
                "ma20_vs_ma50": float(dfeat["ma20_vs_ma50"]),
                "rsi14": float(dfeat["rsi14"]),
                "daily_volume_ratio": float(dfeat["daily_volume_ratio"]),
            }
            for name in MODEL_FEATURES:
                if name not in feature:
                    feature[name] = float(f[name])
            if any(not np.isfinite(float(feature[k])) for k in MODEL_FEATURES):
                continue

            price = float(f["price"])
            levels = trade_levels(price, float(f["bar_range"]))
            row = {k: float(feature[k]) for k in MODEL_FEATURES}
            row.update({
                "ticker": ticker,
                "timestamp": ts.isoformat(),
                "session_date": day.isoformat(),
                "ret15": float(f["ret15"]),
                "minutes_from_open": float(f["minutes_from_open"]),
            })

            valid_horizons = 0
            for horizon in horizons:
                hb = horizon_bars[horizon]
                if j + hb >= len(session):
                    continue
                outcome = _evaluate_future_arrays(
                    hi_all[j + 1:j + 1 + hb],
                    lo_all[j + 1:j + 1 + hb],
                    cl_all[j + 1:j + 1 + hb],
                    price,
                    levels,
                    base_cost_bps,
                    stress_cost_bps,
                )
                # A five-minute bar can touch both stop and target without revealing
                # which happened first. Exclude that horizon only rather than throwing
                # away the same timestamp for every shorter horizon too.
                if outcome is None or outcome.get("ambiguous"):
                    continue
                row[f"positive_net_{horizon}"] = 1 if float(outcome["net_r_base"]) > 0 else 0
                row[f"target_hit_{horizon}"] = 1 if outcome["target_hit"] else 0
                row[f"stop_hit_{horizon}"] = 1 if outcome["stop_hit"] else 0
                row[f"net_r_base_{horizon}"] = float(outcome["net_r_base"])
                row[f"net_r_stress_{horizon}"] = float(outcome["net_r_stress"])
                row[f"final_return_{horizon}"] = float(outcome["final_return"])
                valid_horizons += 1
            if valid_horizons:
                rows.append(row)
    return rows


def split_by_complete_trading_days(df: pd.DataFrame, cfg: dict):
    d = df.copy()
    d["timestamp_dt"] = pd.to_datetime(d["timestamp"], utc=True)
    days = sorted(d["session_date"].dropna().unique().tolist())
    if len(days) < 12:
        return d.iloc[0:0].copy(), d.iloc[0:0].copy(), d.iloc[0:0].copy(), days

    train_frac = float(cfg.get("calibration_train_fraction", 0.50))
    cal_frac = float(cfg.get("calibration_calibration_fraction", 0.25))
    train_end = max(1, int(len(days) * train_frac))
    cal_end = max(train_end + 1, int(len(days) * (train_frac + cal_frac)))
    cal_end = min(cal_end, len(days) - 1)

    train_days = set(days[:train_end])
    cal_days = set(days[train_end:cal_end])
    test_days = set(days[cal_end:])
    return (
        d[d["session_date"].isin(train_days)].copy(),
        d[d["session_date"].isin(cal_days)].copy(),
        d[d["session_date"].isin(test_days)].copy(),
        days,
    )


def raw_event_flags(prob: np.ndarray, ret15: np.ndarray, threshold: float, surge_margin: float, require_positive_ret15: bool) -> np.ndarray:
    flags = np.zeros(len(prob), dtype=bool)
    for i in range(len(prob)):
        if require_positive_ret15 and ret15[i] <= 0:
            continue
        prev = prob[max(0, i - 3):i]
        prev_high = float(np.max(prev)) if len(prev) else 0.0
        crossed = prob[i] >= threshold and prev_high < threshold
        surged = prob[i] >= threshold + surge_margin and (prob[i] - prev_high) >= surge_margin
        flags[i] = bool(crossed or surged)
    return flags


def select_alert_events(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, cfg: dict) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    d = frame.copy().reset_index(drop=True)
    d["model_probability"] = np.asarray(probabilities, dtype=float)
    d["timestamp_dt"] = pd.to_datetime(d["timestamp"], utc=True)
    cooldown = pd.Timedelta(minutes=int(cfg.get("alert_same_ticker_cooldown_minutes", 60)))
    surge_margin = float(cfg.get("alert_surge_probability_margin", 0.05))
    require_up = bool(cfg.get("require_positive_ret15", False))

    chosen = []
    for ticker, g in d.groupby("ticker", sort=False):
        g = g.sort_values("timestamp_dt").copy()
        p = g["model_probability"].to_numpy(float)
        r15 = g["ret15"].to_numpy(float)
        raw = raw_event_flags(p, r15, threshold, surge_margin, require_up)
        last_selected = None
        for pos, (_, row) in enumerate(g.iterrows()):
            if not raw[pos]:
                continue
            ts = row["timestamp_dt"]
            if last_selected is not None and ts - last_selected < cooldown:
                continue
            chosen.append(row)
            last_selected = ts
    if not chosen:
        return d.iloc[0:0].copy()
    events = pd.DataFrame(chosen)
    # Mimic the live scanner: one highest-evidence name per scan timestamp.
    idx = events.groupby("timestamp_dt")["model_probability"].idxmax()
    return events.loc[idx].sort_values("timestamp_dt").reset_index(drop=True)


def _time_bucket(minutes: float) -> str:
    if minutes <= 90:
        return "early"
    if minutes >= 270:
        return "late"
    return "midday"


def matched_baseline_metrics(frame: pd.DataFrame, events: pd.DataFrame, horizon: int) -> dict:
    if frame.empty or events.empty:
        return {"samples": 0, "positive_rate": None, "avg_r_base": None, "avg_r_stress": None}
    d = frame.copy()
    d["direction_bucket"] = np.where(d["ret15"].astype(float) >= 0, "up", "down")
    d["time_bucket"] = d["minutes_from_open"].astype(float).map(_time_bucket)
    e = events.copy()
    e["direction_bucket"] = np.where(e["ret15"].astype(float) >= 0, "up", "down")
    e["time_bucket"] = e["minutes_from_open"].astype(float).map(_time_bucket)

    pos_col = f"positive_net_{horizon}"
    r_col = f"net_r_base_{horizon}"
    stress_col = f"net_r_stress_{horizon}"
    weighted = []
    total_w = 0
    total_samples = 0
    for (direction, bucket), eg in e.groupby(["direction_bucket", "time_bucket"]):
        candidates = d[(d["direction_bucket"] == direction) & (d["time_bucket"] == bucket)]
        if candidates.empty:
            continue
        w = len(eg)
        total_w += w
        total_samples += len(candidates)
        weighted.append((
            w,
            float(candidates[pos_col].mean()),
            float(candidates[r_col].mean()),
            float(candidates[stress_col].mean()),
        ))
    if total_w == 0:
        return {"samples": 0, "positive_rate": None, "avg_r_base": None, "avg_r_stress": None}
    return {
        "samples": int(total_samples),
        "positive_rate": float(sum(w * p for w, p, _, _ in weighted) / total_w),
        "avg_r_base": float(sum(w * r for w, _, r, _ in weighted) / total_w),
        "avg_r_stress": float(sum(w * r for w, _, _, r in weighted) / total_w),
        "matching": "same 15-minute direction bucket and same intraday time bucket",
    }


def day_cluster_stats(events: pd.DataFrame, col: str) -> tuple[float | None, float | None, float | None]:
    if events.empty:
        return None, None, None
    by_day = events.groupby("session_date")[col].mean().astype(float)
    mean = float(by_day.mean())
    if len(by_day) <= 1:
        return mean, None, None
    se = float(by_day.std(ddof=1) / math.sqrt(len(by_day)))
    lower80 = mean - 0.84 * se
    return mean, se, float(lower80)


def metrics_for_events(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, cfg: dict, horizon: int) -> dict:
    events = select_alert_events(frame, probabilities, threshold, cfg)
    base = matched_baseline_metrics(frame, events, horizon)
    alerts = int(len(events))
    pos_col = f"positive_net_{horizon}"
    target_col = f"target_hit_{horizon}"
    stop_col = f"stop_hit_{horizon}"
    r_col = f"net_r_base_{horizon}"
    stress_col = f"net_r_stress_{horizon}"

    result = {
        "samples": int(len(frame)),
        "alerts": alerts,
        "alert_days": int(events["session_date"].nunique()) if alerts else 0,
        "baseline": base,
        "positive_rate": float(events[pos_col].mean()) if alerts else None,
        "target_hit_rate": float(events[target_col].mean()) if alerts else None,
        "stop_hit_rate": float(events[stop_col].mean()) if alerts else None,
        "avg_r": float(events[r_col].mean()) if alerts else None,
        "median_r": float(events[r_col].median()) if alerts else None,
        "stress_avg_r": float(events[stress_col].mean()) if alerts else None,
        "horizon_minutes": int(horizon),
    }
    if alerts and base.get("positive_rate") is not None:
        result["positive_rate_lift"] = float(result["positive_rate"] - base["positive_rate"])
        result["avg_r_lift"] = float(result["avg_r"] - base["avg_r_base"])
        result["stress_avg_r_lift"] = float(result["stress_avg_r"] - base["avg_r_stress"])
    else:
        result["positive_rate_lift"] = None
        result["avg_r_lift"] = None
        result["stress_avg_r_lift"] = None

    day_avg, day_se, day_lcb80 = day_cluster_stats(events, r_col)
    stress_day_avg, _, stress_day_lcb80 = day_cluster_stats(events, stress_col)
    result.update({
        "day_avg_r": day_avg,
        "day_se_r": day_se,
        "day_lcb80_r": day_lcb80,
        "stress_day_avg_r": stress_day_avg,
        "stress_day_lcb80_r": stress_day_lcb80,
    })
    y = frame[pos_col].astype(int).to_numpy()
    result["auc"] = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) > 1 else None
    return result


def threshold_search(frame: pd.DataFrame, probabilities: np.ndarray, cfg: dict, horizon: int):
    if frame.empty or len(probabilities) == 0:
        return None
    min_alerts = int(cfg.get("calibration_min_threshold_alerts", 15))
    min_days = int(cfg.get("calibration_min_threshold_days", 5))
    quantiles = np.linspace(0.70, 0.98, 15)
    candidates = sorted(set(float(np.quantile(probabilities, q)) for q in quantiles))

    best = None
    for thr in candidates:
        metrics = metrics_for_events(frame, probabilities, thr, cfg, horizon)
        if metrics["alerts"] < min_alerts or metrics["alert_days"] < min_days:
            continue
        stress_day = metrics.get("stress_day_avg_r")
        lcb = metrics.get("day_lcb80_r")
        lift_r = metrics.get("avg_r_lift")
        lift_p = metrics.get("positive_rate_lift")
        if None in {stress_day, lcb, lift_r, lift_p}:
            continue
        score = float(stress_day + 0.50 * lcb + 0.45 * lift_r + 0.20 * lift_p)
        candidate = {"threshold": float(thr), "selection_score": score, "metrics": metrics}
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    return best


def readiness_test(metrics: dict, cfg: dict) -> tuple[bool, list[dict]]:
    tests = [
        ("enough separated alerts", metrics.get("alerts", 0) >= int(cfg.get("calibration_min_test_alerts", 12))),
        ("enough different test days", metrics.get("alert_days", 0) >= int(cfg.get("calibration_min_test_days", 5))),
        ("average return after base drag", (metrics.get("avg_r") if metrics.get("avg_r") is not None else -999) >= float(cfg.get("calibration_min_test_avg_r", 0.12))),
        ("positive under stress drag", (metrics.get("stress_avg_r") if metrics.get("stress_avg_r") is not None else -999) > float(cfg.get("calibration_min_test_stress_avg_r", 0.0))),
        ("day-clustered lower bound above zero", (metrics.get("day_lcb80_r") if metrics.get("day_lcb80_r") is not None else -999) > float(cfg.get("calibration_min_test_day_lcb80_r", 0.0))),
        ("beats matched snapshots on average R", (metrics.get("avg_r_lift") if metrics.get("avg_r_lift") is not None else -999) >= float(cfg.get("calibration_min_test_avg_r_lift", 0.10))),
        ("beats matched snapshots on positive rate", (metrics.get("positive_rate_lift") if metrics.get("positive_rate_lift") is not None else -999) >= float(cfg.get("calibration_min_test_positive_rate_lift", 0.03))),
        ("model ranking better than chance", (metrics.get("auc") if metrics.get("auc") is not None else -999) >= float(cfg.get("calibration_min_test_auc", 0.52))),
    ]
    detail = [{"name": name, "passed": bool(ok)} for name, ok in tests]
    return all(x["passed"] for x in detail), detail


def probability_reference(probabilities: np.ndarray) -> dict:
    pct = np.arange(0, 101, 2, dtype=float)
    values = np.quantile(probabilities, pct / 100.0) if len(probabilities) else np.array([])
    return {"percentiles": [float(x) for x in pct], "values": [float(x) for x in values]}


def percentile_from_reference(value: float, ref: dict) -> float | None:
    try:
        vals = np.asarray(ref["values"], dtype=float)
        pcts = np.asarray(ref["percentiles"], dtype=float)
        return float(np.interp(float(value), vals, pcts, left=0.0, right=100.0))
    except Exception:
        return None


def main():
    cfg = load_config(BASE / "cloud_config.json")
    watchlist = [str(x).upper() for x in cfg["watchlist"]]
    calendar_days = min(59, int(cfg.get("calibration_calendar_days", 56)))
    batch_size = int(cfg.get("calibration_download_batch_size", 10))
    horizons = sorted(set(int(x) for x in cfg.get("backtest_horizon_candidates_minutes", [15, 30, 60])))

    end_date = datetime.now(NY).date()
    start_date = end_date - timedelta(days=calendar_days)
    daily = fetch_batched(watchlist, "1y", "1d", batch_size=batch_size)
    intraday_names = list(dict.fromkeys(watchlist + ["SPY"]))
    intra = fetch_batched(intraday_names, None, "5m", batch_size=batch_size, start=start_date.isoformat(), end=end_date.isoformat())
    spy = intra.get("SPY", pd.DataFrame())

    rows = []
    for ticker in watchlist:
        print(f"Building v13 historical snapshots: {ticker}", flush=True)
        t0 = time.perf_counter()
        before = len(rows)
        rows.extend(build_samples(ticker, daily.get(ticker, pd.DataFrame()), intra.get(ticker, pd.DataFrame()), spy, cfg))
        print(f"Finished {ticker}: +{len(rows)-before:,} samples · {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"Historical replay complete: {len(rows):,} total samples", flush=True)
    data = pd.DataFrame(rows)
    out_path = BASE / "clarity_calibration.json"
    min_total = int(cfg.get("calibration_min_total_samples", 2200))
    if len(data) < min_total:
        payload = {"version": 3, "status": "insufficient", "ready_for_alerts": False, "generated_at": datetime.now(NY).isoformat(), "samples": int(len(data)), "message": "Not enough historical snapshots to fit the v13 evidence model yet."}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return

    train, cal, test, days = split_by_complete_trading_days(data, cfg)
    if min(len(train), len(cal), len(test)) == 0:
        payload = {"version": 3, "status": "insufficient", "ready_for_alerts": False, "generated_at": datetime.now(NY).isoformat(), "samples": int(len(data)), "trading_days": int(len(days)), "message": "Not enough complete trading days for the v13 time-ordered train/calibration/test split."}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return

    horizon_candidates = []
    fitted = {}
    for horizon in horizons:
        target = f"positive_net_{horizon}"
        required = [target, f"net_r_base_{horizon}", f"net_r_stress_{horizon}"]
        train_h = train.dropna(subset=required).copy()
        cal_h = cal.dropna(subset=required).copy()
        test_h = test.dropna(subset=required).copy()
        if min(len(train_h), len(cal_h), len(test_h)) == 0 or train_h[target].nunique() < 2:
            horizon_candidates.append({
                "horizon_minutes": int(horizon),
                "validation_selection_score": -999.0,
                "threshold_selection": None,
                "reason": "Not enough unambiguous samples for this horizon.",
            })
            continue

        scaler_h = StandardScaler()
        Z_train_h = scaler_h.fit_transform(train_h[MODEL_FEATURES].astype(float).to_numpy())
        Z_cal_h = scaler_h.transform(cal_h[MODEL_FEATURES].astype(float).to_numpy())
        y_train = train_h[target].astype(int).to_numpy()
        model = LogisticRegression(C=float(cfg.get("calibration_logistic_c", 0.20)), max_iter=2500, random_state=42)
        model.fit(Z_train_h, y_train)
        p_cal = model.predict_proba(Z_cal_h)[:, 1]
        choice = threshold_search(cal_h, p_cal, cfg, horizon)
        validation_score = choice["selection_score"] if choice else -999.0
        horizon_candidates.append({
            "horizon_minutes": int(horizon),
            "validation_selection_score": float(validation_score),
            "threshold_selection": choice,
            "train_samples": int(len(train_h)),
            "calibration_samples": int(len(cal_h)),
            "test_samples": int(len(test_h)),
        })
        fitted[horizon] = (model, scaler_h, cal_h, test_h, p_cal, choice)
        print(f"Horizon {horizon}m validation score: {validation_score:+.4f}", flush=True)

    usable = [x for x in horizon_candidates if x["threshold_selection"] is not None]
    if not usable:
        payload = {
            "version": 3,
            "method": "v13_market_structure_multihorizon",
            "status": "weak",
            "ready_for_alerts": False,
            "generated_at": datetime.now(NY).isoformat(),
            "training_window_calendar_days": calendar_days,
            "trading_days": int(len(days)),
            "candidate_horizons": horizon_candidates,
            "message": "No horizon produced enough separated validation alerts to choose a defensible evidence line.",
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return

    selected = max(usable, key=lambda x: x["validation_selection_score"])
    horizon = int(selected["horizon_minutes"])
    model, scaler, cal_selected, test_selected, p_cal, choice = fitted[horizon]
    p_test = model.predict_proba(scaler.transform(test_selected[MODEL_FEATURES].astype(float).to_numpy()))[:, 1]
    threshold = float(choice["threshold"])
    cal_metrics = metrics_for_events(cal_selected, p_cal, threshold, cfg, horizon)
    test_metrics = metrics_for_events(test_selected, p_test, threshold, cfg, horizon)
    ready, gates = readiness_test(test_metrics, cfg)
    status = "ready" if ready else "weak"

    ref = probability_reference(p_test)
    threshold_pctile = percentile_from_reference(threshold, ref)
    payload = {
        "version": 3,
        "method": "v13_market_structure_multihorizon",
        "status": status,
        "ready_for_alerts": bool(ready),
        "generated_at": datetime.now(NY).isoformat(),
        "training_window_calendar_days": calendar_days,
        "trading_days": int(len(days)),
        "horizon_minutes": horizon,
        "candidate_horizons": horizon_candidates,
        "model_target": f"positive net R after base execution-drag assumption over the validation-selected {horizon}-minute horizon",
        "round_trip_cost_bps": float(cfg.get("backtest_round_trip_cost_bps", 5.0)),
        "stress_round_trip_cost_bps": float(cfg.get("backtest_stress_round_trip_cost_bps", 15.0)),
        "feature_names": MODEL_FEATURES,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "coefficients": [float(x) for x in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "probability_threshold": threshold,
        "threshold_percentile": float(threshold_pctile) if threshold_pctile is not None else None,
        "probability_reference": ref,
        "guardrails": {
            "require_positive_ret15": bool(cfg.get("require_positive_ret15", False)),
            "same_ticker_cooldown_minutes": int(cfg.get("alert_same_ticker_cooldown_minutes", 60)),
            "surge_probability_margin": float(cfg.get("alert_surge_probability_margin", 0.05)),
        },
        "splits": {
            "train_samples": int(next((x.get("train_samples", len(train)) for x in horizon_candidates if x["horizon_minutes"] == horizon), len(train))),
            "calibration_samples": int(len(cal_selected)),
            "test_samples": int(len(test_selected)),
            "train_days": int(train["session_date"].nunique()),
            "calibration_days": int(cal["session_date"].nunique()),
            "test_days": int(test["session_date"].nunique()),
            "train_start": str(train["session_date"].min()),
            "test_end": str(test["session_date"].max()),
        },
        "threshold_selection": choice,
        "readiness_gates": gates,
        "metrics": {"calibration": cal_metrics, "test": test_metrics},
        "notes": [
            "v13 chooses among 15-, 30-, and 60-minute horizons using only the middle validation block; the final test block remains untouched until that choice is frozen.",
            "The feature set adds volatility-normalized movement, VWAP distance, opening-range context, market-relative movement, time-of-day-aware relative-volume acceleration, and prior-session same-clock-time behavior.",
            "v13 does not force every long setup to already have positive 15-minute momentum; the regularized model can distinguish continuation-style and rebound-style conditions.",
            "Baseline comparisons are matched by current short-term direction and broad intraday time bucket rather than comparing every event with an unrelated generic snapshot.",
            "Threshold evaluation counts separated threshold-crossing/surge events with a same-ticker cooldown and one highest-evidence ticker per historical scan timestamp.",
            "Readiness still requires positive base/stress performance, day-level consistency, improvement over matched snapshots, sample size, and out-of-sample ranking above chance.",
            "Five-minute OHLC data cannot reveal target/stop order when both occur in the same bar; those samples are excluded.",
            "Execution-drag assumptions are scenarios, not measured Robinhood costs. Historical results do not guarantee future performance.",
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
