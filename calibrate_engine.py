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
    evaluate_future_path,
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
        label = f"Downloading {interval} batch {batch_no}/{total_batches}: {', '.join(batch)}"
        print(label, flush=True)
        t0 = time.perf_counter()
        part = get_many(batch, period, interval, start=start, end=end)
        out.update(part)
        elapsed = time.perf_counter() - t0
        missing = [t for t in batch if t not in part or part[t] is None or part[t].empty]
        suffix = f" · {elapsed:.1f}s"
        if missing:
            suffix += f" · no usable data: {', '.join(missing)}"
        print(f"Finished {interval} batch {batch_no}/{total_batches}{suffix}", flush=True)
        time.sleep(0.35)
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


def _spy_ret15_lookup(spy_df: pd.DataFrame) -> dict[pd.Timestamp, float]:
    spy = normalize_intraday(spy_df)
    if spy.empty or "Close" not in spy:
        return {}
    out: dict[pd.Timestamp, float] = {}
    for _, session in spy.groupby(spy.index.date, sort=True):
        close = pd.to_numeric(session["Close"], errors="coerce")
        for j, ts in enumerate(session.index):
            if j >= 3 and pd.notna(close.iloc[j]) and pd.notna(close.iloc[j - 3]) and float(close.iloc[j - 3]) != 0:
                out[ts] = float(close.iloc[j] / close.iloc[j - 3] - 1.0)
            else:
                out[ts] = 0.0
    return out


def _daily_feature_cache(daily_df: pd.DataFrame, session_dates: list) -> dict:
    cache = {}
    for day in session_dates:
        when = pd.Timestamp(day).tz_localize(NY)
        feat = daily_features(completed_daily_history(daily_df, as_of=when))
        if feat is not None:
            cache[day] = feat
    return cache


def _intraday_feature_frame(intra_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """Precompute the same feature family used live without repeatedly replaying the full history.

    The v12 implementation rebuilt historical slices for every five-minute bar. That was
    methodologically fine but computationally quadratic. This version walks each session once,
    keeps prior same-clock-time volume history, and produces one feature row per timestamp.
    """
    d = normalize_intraday(intra_df)
    if d.empty:
        return pd.DataFrame()

    needed = [c for c in ["High", "Low", "Close", "Volume"] if c in d.columns]
    d = d[needed].copy()
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[c for c in ["High", "Low", "Close"] if c in d.columns])
    if d.empty:
        return pd.DataFrame()

    spy_lookup = _spy_ret15_lookup(spy_df)
    slot_history: dict[int, list[float]] = {}
    prior_global_volumes: list[float] = []
    prior_global_highs: list[float] = []
    rows = []

    for day, session in d.groupby(d.index.date, sort=True):
        session = session.sort_index()
        close = session["Close"].to_numpy(dtype=float)
        high = session["High"].to_numpy(dtype=float)
        low = session["Low"].to_numpy(dtype=float)
        volume = session["Volume"].fillna(0).to_numpy(dtype=float) if "Volume" in session else np.zeros(len(session), dtype=float)
        times = list(session.index)

        # The same-slot baselines for the current day only use *prior* sessions.
        slot_baselines = []
        slot_counts = []
        slots = []
        for ts in times:
            slot = int(round((ts.hour * 60 + ts.minute) - (9 * 60 + 30)))
            hist = slot_history.get(slot, [])
            slots.append(slot)
            slot_counts.append(len(hist))
            slot_baselines.append(float(np.mean(hist[-10:])) if hist else np.nan)

        for j, ts in enumerate(times):
            # Live intraday_features requires four bars for a 15-minute comparison.
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

            recent_start = max(0, j - 2)
            recent_vol = float(np.mean(volume[recent_start:j + 1]))
            available_baselines = [
                slot_baselines[k]
                for k in range(recent_start, j + 1)
                if slot_counts[k] >= 3 and np.isfinite(slot_baselines[k]) and slot_baselines[k] > 0
            ]
            if available_baselines:
                normal_vol = float(np.mean(available_baselines))
            elif j + 1 >= 7:
                fallback = volume[max(0, j - 26):j - 2]
                normal_vol = float(np.mean(fallback)) if len(fallback) else 0.0
            elif prior_global_volumes:
                normal_vol = float(np.mean(prior_global_volumes[-24:]))
            else:
                normal_vol = 0.0
            vr = recent_vol / normal_vol if normal_vol > 0 else 1.0

            if j + 1 >= 6:
                prev_highs = high[max(0, j - 20):j]
            else:
                prev_highs = np.asarray((prior_global_highs + high[:j].tolist())[-20:], dtype=float)
            if len(prev_highs) == 0 or not np.isfinite(prev_highs).any():
                continue
            prior_high = float(np.nanmax(prev_highs))

            mid = float(np.mean(close[max(0, j - 19):j + 1]))
            ranges = high[max(0, j - 11):j + 1] - low[max(0, j - 11):j + 1]
            bar_range = float(np.nanmean(ranges)) if len(ranges) else price * 0.001
            bar_range = max(bar_range, price * 0.001)
            dist = float((prior_high - price) / bar_range)

            rows.append({
                "timestamp": ts,
                "session_date": day,
                "price": price,
                "bar_range": bar_range,
                "ret15": ret15,
                "ret60m": ret60,
                "acceleration": acceleration,
                "intraday_volume_ratio": float(vr),
                "log_intraday_volume_ratio": float(math.log(max(vr, 0.05))),
                "breakout_distance": dist,
                "above_mid": 1.0 if price > mid else 0.0,
                "market_ret15": float(spy_lookup.get(ts, 0.0)),
                "range_pct": float(bar_range / price),
                "minutes_from_open": float(slots[j]),
            })

        # Only after finishing the session may today's volume become a prior-session baseline.
        for slot, vol in zip(slots, volume):
            hist = slot_history.setdefault(slot, [])
            if np.isfinite(vol):
                hist.append(float(vol))
                if len(hist) > 10:
                    del hist[:-10]
        prior_global_volumes.extend([float(x) for x in volume if np.isfinite(x)])
        if len(prior_global_volumes) > 24:
            prior_global_volumes = prior_global_volumes[-24:]
        prior_global_highs.extend([float(x) for x in high if np.isfinite(x)])
        if len(prior_global_highs) > 20:
            prior_global_highs = prior_global_highs[-20:]

    if not rows:
        return pd.DataFrame()
    f = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return f



def _evaluate_future_arrays(high: np.ndarray, low: np.ndarray, close: np.ndarray, price: float, levels: dict, base_cost_bps: float, stress_cost_bps: float) -> dict | None:
    """Fast exact equivalent of evaluate_future_path for one fixed future window."""
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
        "max_upside": float((float(np.nanmax(high)) - float(price)) / float(price)),
        "max_drawdown": float((float(np.nanmin(low)) - float(price)) / float(price)),
        "final_return": float(float(close[-1]) / float(price) - 1.0),
    }

def build_samples(ticker: str, daily_df: pd.DataFrame, intra_df: pd.DataFrame, spy_df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Build historical snapshots in roughly linear time rather than repeatedly slicing history."""
    intra = normalize_intraday(intra_df)
    if intra.empty or len(intra) < 45:
        return []

    features = _intraday_feature_frame(intra, spy_df)
    if features.empty:
        return []

    # Preserve v12's initial 30-bar warm-up exactly.
    first_allowed_ts = intra.index[30] if len(intra) > 30 else intra.index[-1]

    stride = max(1, int(cfg.get("calibration_stride_bars", 1)))
    horizon_bars = max(1, int(cfg.get("backtest_horizon_minutes", 60)) // 5)
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

        for j in range(0, len(session) - horizon_bars, stride):
            ts = session.index[j]
            if ts < first_allowed_ts or ts not in features.index:
                continue
            f = features.loc[ts]
            minute = float(f["minutes_from_open"])
            if minute < start_minute or minute > end_minute:
                continue

            if j + horizon_bars >= len(session):
                continue

            feature = {
                "daily_ret20": float(dfeat["daily_ret20"]),
                "daily_ret60": float(dfeat["daily_ret60"]),
                "price_vs_ma20": float(dfeat["price_vs_ma20"]),
                "ma20_vs_ma50": float(dfeat["ma20_vs_ma50"]),
                "rsi14": float(dfeat["rsi14"]),
                "daily_volume_ratio": float(dfeat["daily_volume_ratio"]),
                "ret15": float(f["ret15"]),
                "ret60m": float(f["ret60m"]),
                "acceleration": float(f["acceleration"]),
                "log_intraday_volume_ratio": float(f["log_intraday_volume_ratio"]),
                "breakout_distance": float(f["breakout_distance"]),
                "above_mid": float(f["above_mid"]),
                "market_ret15": float(f["market_ret15"]),
                "range_pct": float(f["range_pct"]),
                "minutes_from_open": float(f["minutes_from_open"]),
                "price": float(f["price"]),
                "bar_range": float(f["bar_range"]),
            }

            price = float(feature["price"])
            levels = trade_levels(price, float(feature["bar_range"]))
            hi = session["High"].to_numpy(dtype=float)[j + 1:j + 1 + horizon_bars]
            lo = session["Low"].to_numpy(dtype=float)[j + 1:j + 1 + horizon_bars]
            cl = session["Close"].to_numpy(dtype=float)[j + 1:j + 1 + horizon_bars]
            outcome = _evaluate_future_arrays(hi, lo, cl, price, levels, base_cost_bps, stress_cost_bps)
            if outcome is None or outcome.get("ambiguous"):
                continue

            row = {k: float(feature[k]) for k in MODEL_FEATURES}
            row.update({
                "ticker": ticker,
                "timestamp": ts.isoformat(),
                "session_date": day.isoformat(),
                "positive_net": 1 if float(outcome["net_r_base"]) > 0 else 0,
                "target_hit": 1 if outcome["target_hit"] else 0,
                "stop_hit": 1 if outcome["stop_hit"] else 0,
                "net_r_base": float(outcome["net_r_base"]),
                "net_r_stress": float(outcome["net_r_stress"]),
                "max_upside": float(outcome["max_upside"]),
                "max_drawdown": float(outcome["max_drawdown"]),
                "final_return": float(outcome["final_return"]),
            })
            rows.append(row)
    return rows


def split_by_complete_trading_days(df: pd.DataFrame):
    """Split by whole market days so the same day's correlated bars never span two sets."""
    d = df.copy()
    d["timestamp_dt"] = pd.to_datetime(d["timestamp"], utc=True)
    days = sorted(d["session_date"].dropna().unique().tolist())
    if len(days) < 10:
        return d.iloc[0:0].copy(), d.iloc[0:0].copy(), d.iloc[0:0].copy(), days

    train_end = max(1, int(len(days) * 0.60))
    cal_end = max(train_end + 1, int(len(days) * 0.80))
    cal_end = min(cal_end, len(days) - 1)

    train_days = set(days[:train_end])
    cal_days = set(days[train_end:cal_end])
    test_days = set(days[cal_end:])

    train = d[d["session_date"].isin(train_days)].copy()
    cal = d[d["session_date"].isin(cal_days)].copy()
    test = d[d["session_date"].isin(test_days)].copy()
    return train, cal, test, days


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
    """Mimic live alert behavior: threshold crossing/surge + same-ticker cooldown + one top name per scan."""
    if frame.empty:
        return frame.iloc[0:0].copy()

    d = frame.copy().reset_index(drop=True)
    d["model_probability"] = np.asarray(probabilities, dtype=float)
    d["timestamp_dt"] = pd.to_datetime(d["timestamp"], utc=True)
    cooldown = pd.Timedelta(minutes=int(cfg.get("alert_same_ticker_cooldown_minutes", 60)))
    surge_margin = float(cfg.get("alert_surge_probability_margin", 0.08))
    require_up = bool(cfg.get("require_positive_ret15", True))

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
    # The real scanner only sends the highest-evidence name when several cross in the same scan.
    events = events.sort_values(["timestamp_dt", "model_probability"], ascending=[True, False])
    events = events.drop_duplicates(subset=["timestamp_dt"], keep="first")
    return events.sort_values("timestamp_dt").reset_index(drop=True)


def baseline_metrics(frame: pd.DataFrame, cfg: dict) -> dict:
    d = frame.copy()
    if bool(cfg.get("require_positive_ret15", True)):
        d = d[d["ret15"] > 0]
    if d.empty:
        return {"samples": 0, "positive_rate": None, "avg_r_base": None, "avg_r_stress": None}
    return {
        "samples": int(len(d)),
        "positive_rate": float(d["positive_net"].mean()),
        "avg_r_base": float(d["net_r_base"].mean()),
        "avg_r_stress": float(d["net_r_stress"].mean()),
    }


def day_cluster_stats(events: pd.DataFrame, col: str) -> tuple[float | None, float | None, float | None]:
    if events.empty:
        return None, None, None
    by_day = events.groupby("session_date")[col].mean().astype(float)
    mean = float(by_day.mean())
    if len(by_day) <= 1:
        return mean, None, None
    se = float(by_day.std(ddof=1) / math.sqrt(len(by_day)))
    # Approximate 80% one-sided lower bound. Deliberately not presented as a formal guarantee.
    lower80 = mean - 0.84 * se
    return mean, se, float(lower80)


def metrics_for_events(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, cfg: dict) -> dict:
    events = select_alert_events(frame, probabilities, threshold, cfg)
    base = baseline_metrics(frame, cfg)
    alerts = int(len(events))

    result = {
        "samples": int(len(frame)),
        "alerts": alerts,
        "alert_days": int(events["session_date"].nunique()) if alerts else 0,
        "baseline": base,
        "positive_rate": float(events["positive_net"].mean()) if alerts else None,
        "target_hit_rate": float(events["target_hit"].mean()) if alerts else None,
        "stop_hit_rate": float(events["stop_hit"].mean()) if alerts else None,
        "avg_r": float(events["net_r_base"].mean()) if alerts else None,
        "median_r": float(events["net_r_base"].median()) if alerts else None,
        "stress_avg_r": float(events["net_r_stress"].mean()) if alerts else None,
    }

    if alerts and base.get("positive_rate") is not None:
        result["positive_rate_lift"] = float(result["positive_rate"] - base["positive_rate"])
    else:
        result["positive_rate_lift"] = None

    day_avg, day_se, day_lcb80 = day_cluster_stats(events, "net_r_base")
    stress_day_avg, _, stress_day_lcb80 = day_cluster_stats(events, "net_r_stress")
    result.update({
        "day_avg_r": day_avg,
        "day_se_r": day_se,
        "day_lcb80_r": day_lcb80,
        "stress_day_avg_r": stress_day_avg,
        "stress_day_lcb80_r": stress_day_lcb80,
    })

    y = frame["positive_net"].astype(int).to_numpy()
    if len(np.unique(y)) > 1:
        result["auc"] = float(roc_auc_score(y, probabilities))
    else:
        result["auc"] = None
    return result


def threshold_search(frame: pd.DataFrame, probabilities: np.ndarray, cfg: dict):
    if frame.empty or len(probabilities) == 0:
        return None

    min_alerts = int(cfg.get("calibration_min_threshold_alerts", 18))
    min_days = int(cfg.get("calibration_min_threshold_days", 4))
    # A compact candidate set limits threshold-mining on a short validation window.
    quantiles = np.linspace(0.60, 0.96, 13)
    candidates = sorted(set(float(np.quantile(probabilities, q)) for q in quantiles))

    best = None
    for thr in candidates:
        metrics = metrics_for_events(frame, probabilities, thr, cfg)
        if metrics["alerts"] < min_alerts or metrics["alert_days"] < min_days:
            continue
        stress_day = metrics.get("stress_day_avg_r")
        lcb = metrics.get("day_lcb80_r")
        lift = metrics.get("positive_rate_lift")
        if stress_day is None or lcb is None or lift is None:
            continue
        # Selection is deliberately conservative: prefer thresholds that survive the
        # stress-cost scenario, have positive day-clustered return, and improve the win rate.
        score = float(stress_day + 0.50 * lcb + 0.50 * lift)
        candidate = {"threshold": float(thr), "selection_score": score, "metrics": metrics}
        if best is None or candidate["selection_score"] > best["selection_score"]:
            best = candidate
    return best


def readiness_test(metrics: dict, cfg: dict) -> tuple[bool, list[dict]]:
    tests = [
        ("enough separated alerts", metrics.get("alerts", 0) >= int(cfg.get("calibration_min_test_alerts", 12))),
        ("enough different test days", metrics.get("alert_days", 0) >= int(cfg.get("calibration_min_test_days", 4))),
        ("average return after base drag", (metrics.get("avg_r") or -999) >= float(cfg.get("calibration_min_test_avg_r", 0.15))),
        ("positive under stress drag", (metrics.get("stress_avg_r") or -999) > float(cfg.get("calibration_min_test_stress_avg_r", 0.0))),
        ("day-clustered lower bound above zero", (metrics.get("day_lcb80_r") or -999) > float(cfg.get("calibration_min_test_day_lcb80_r", 0.0))),
        ("beats comparable positive-momentum snapshots", (metrics.get("positive_rate_lift") or -999) >= float(cfg.get("calibration_min_test_positive_rate_lift", 0.03))),
        ("model ranking better than chance", (metrics.get("auc") or -999) >= float(cfg.get("calibration_min_test_auc", 0.52))),
    ]
    detail = [{"name": name, "passed": bool(ok)} for name, ok in tests]
    return all(x["passed"] for x in detail), detail


def probability_reference(probabilities: np.ndarray) -> dict:
    pct = np.arange(0, 101, 2, dtype=float)
    values = np.quantile(probabilities, pct / 100.0) if len(probabilities) else np.array([])
    return {
        "percentiles": [float(x) for x in pct],
        "values": [float(x) for x in values],
    }


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

    # yfinance's documented intraday history limit is 60 days. Use explicit dates so
    # v12 gets nearly the full allowed window without asking for an unsupported period.
    end_date = datetime.now(NY).date()
    start_date = end_date - timedelta(days=calendar_days)

    daily = fetch_batched(watchlist, "1y", "1d", batch_size=batch_size)
    intraday_names = list(dict.fromkeys(watchlist + ["SPY"]))
    intra = fetch_batched(
        intraday_names,
        None,
        "5m",
        batch_size=batch_size,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    )
    spy = intra.get("SPY", pd.DataFrame())

    rows = []
    for ticker in watchlist:
        print(f"Building historical snapshots: {ticker}", flush=True)
        t0 = time.perf_counter()
        before = len(rows)
        rows.extend(build_samples(
            ticker,
            daily.get(ticker, pd.DataFrame()),
            intra.get(ticker, pd.DataFrame()),
            spy,
            cfg,
        ))
        print(f"Finished {ticker}: +{len(rows) - before:,} samples · {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"Historical replay complete: {len(rows):,} total samples", flush=True)
    data = pd.DataFrame(rows)
    out_path = BASE / "clarity_calibration.json"
    min_total = int(cfg.get("calibration_min_total_samples", 2200))

    if len(data) < min_total or data.get("positive_net", pd.Series(dtype=int)).nunique() < 2:
        payload = {
            "version": 2,
            "status": "insufficient",
            "ready_for_alerts": False,
            "generated_at": datetime.now(NY).isoformat(),
            "samples": int(len(data)),
            "message": "Not enough historical snapshots to fit the v12 evidence model yet.",
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return

    train, cal, test, days = split_by_complete_trading_days(data)
    if min(len(train), len(cal), len(test)) == 0 or train["positive_net"].nunique() < 2:
        payload = {
            "version": 2,
            "status": "insufficient",
            "ready_for_alerts": False,
            "generated_at": datetime.now(NY).isoformat(),
            "samples": int(len(data)),
            "trading_days": int(len(days)),
            "message": "Not enough complete trading days for the time-ordered train/calibration/test split.",
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return

    X_train = train[MODEL_FEATURES].astype(float).to_numpy()
    y_train = train["positive_net"].astype(int).to_numpy()
    X_cal = cal[MODEL_FEATURES].astype(float).to_numpy()
    X_test = test[MODEL_FEATURES].astype(float).to_numpy()

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_cal = scaler.transform(X_cal)
    Z_test = scaler.transform(X_test)

    model = LogisticRegression(
        C=float(cfg.get("calibration_logistic_c", 0.35)),
        max_iter=2000,
        random_state=42,
    )
    model.fit(Z_train, y_train)
    p_cal = model.predict_proba(Z_cal)[:, 1]
    p_test = model.predict_proba(Z_test)[:, 1]

    choice = threshold_search(cal, p_cal, cfg)
    if choice is None:
        threshold = 1.0
        cal_metrics = metrics_for_events(cal, p_cal, threshold, cfg)
        test_metrics = metrics_for_events(test, p_test, threshold, cfg)
        status = "weak"
        ready = False
        gates = [{"name": "validation produced a usable threshold", "passed": False}]
    else:
        threshold = float(choice["threshold"])
        cal_metrics = metrics_for_events(cal, p_cal, threshold, cfg)
        test_metrics = metrics_for_events(test, p_test, threshold, cfg)
        ready, gates = readiness_test(test_metrics, cfg)
        status = "ready" if ready else "weak"

    ref = probability_reference(p_test)
    threshold_pctile = percentile_from_reference(threshold, ref)

    day_list = sorted(data["session_date"].unique().tolist())
    payload = {
        "version": 2,
        "method": "v12_day_split_event_backtest",
        "status": status,
        "ready_for_alerts": bool(ready),
        "generated_at": datetime.now(NY).isoformat(),
        "training_window_calendar_days": calendar_days,
        "trading_days": int(len(day_list)),
        "horizon_minutes": int(cfg.get("backtest_horizon_minutes", 60)),
        "model_target": "positive net R after base execution-drag assumption",
        "round_trip_cost_bps": float(cfg.get("backtest_round_trip_cost_bps", 5.0)),
        "stress_round_trip_cost_bps": float(cfg.get("backtest_stress_round_trip_cost_bps", 15.0)),
        "feature_names": MODEL_FEATURES,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "coefficients": [float(x) for x in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "probability_threshold": float(threshold),
        "threshold_percentile": float(threshold_pctile) if threshold_pctile is not None else None,
        "probability_reference": ref,
        "guardrails": {
            "require_positive_ret15": bool(cfg.get("require_positive_ret15", True)),
            "same_ticker_cooldown_minutes": int(cfg.get("alert_same_ticker_cooldown_minutes", 60)),
            "surge_probability_margin": float(cfg.get("alert_surge_probability_margin", 0.08)),
        },
        "splits": {
            "train_samples": int(len(train)),
            "calibration_samples": int(len(cal)),
            "test_samples": int(len(test)),
            "train_days": int(train["session_date"].nunique()),
            "calibration_days": int(cal["session_date"].nunique()),
            "test_days": int(test["session_date"].nunique()),
            "train_start": str(train["session_date"].min()),
            "test_end": str(test["session_date"].max()),
        },
        "threshold_selection": choice,
        "readiness_gates": gates,
        "metrics": {
            "calibration": cal_metrics,
            "test": test_metrics,
        },
        "notes": [
            "v12 splits history by complete trading days rather than by individual bars.",
            "The model predicts a positive net outcome after a base execution-drag assumption, rather than the much rarer full 2R target hit.",
            "Threshold evaluation counts separated threshold-crossing/surge events with a same-ticker cooldown, instead of counting every qualifying five-minute bar as a new alert.",
            "Only the highest-evidence ticker is counted when several names cross on the same historical scan timestamp.",
            "Readiness requires multiple independent-style gates on the untouched final time period, including base/stress return, day-clustered return, improvement over comparable momentum snapshots, sample size, and AUC.",
            "Five-minute OHLC data cannot reveal target/stop order when both occur in the same bar; those samples are excluded.",
            "Execution-drag assumptions are scenarios, not measured Robinhood costs. Historical results do not guarantee future performance.",
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
