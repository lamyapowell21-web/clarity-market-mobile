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
    evaluate_future_path,
    get_many,
    load_config,
    model_feature_row,
    trade_levels,
)

NY = ZoneInfo("America/New_York")
BASE = Path(__file__).parent


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_batched(tickers: list[str], period: str | None, interval: str, batch_size: int = 10, start=None, end=None) -> dict[str, pd.DataFrame]:
    out = {}
    for batch in batched(tickers, batch_size):
        print(f"Downloading {interval}: {', '.join(batch)}")
        out.update(get_many(batch, period, interval, start=start, end=end))
        time.sleep(1.0)
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


def build_samples(ticker: str, daily_df: pd.DataFrame, intra_df: pd.DataFrame, spy_df: pd.DataFrame, cfg: dict) -> list[dict]:
    """Replay five-minute snapshots using only information available at each timestamp."""
    intra = normalize_intraday(intra_df)
    spy = normalize_intraday(spy_df)
    if intra.empty or len(intra) < 45:
        return []

    stride = max(1, int(cfg.get("calibration_stride_bars", 1)))
    horizon_bars = max(1, int(cfg.get("backtest_horizon_minutes", 60)) // 5)
    start_minute = int(cfg.get("calibration_start_minute_et", 15))
    end_minute = int(cfg.get("calibration_end_minute_et", 350))
    base_cost_bps = float(cfg.get("backtest_round_trip_cost_bps", 5.0))
    stress_cost_bps = float(cfg.get("backtest_stress_round_trip_cost_bps", 15.0))

    rows = []
    for i in range(30, len(intra) - horizon_bars, stride):
        ts = intra.index[i]
        minute = (ts.hour * 60 + ts.minute) - (9 * 60 + 30)
        if minute < start_minute or minute > end_minute:
            continue

        hist = intra.iloc[:i + 1]
        spy_hist = spy.loc[:ts]
        if len(spy_hist) < 30:
            continue

        feature = model_feature_row(daily_df, hist, spy_hist, as_of=ts)
        if feature is None:
            continue

        price = float(feature["price"])
        levels = trade_levels(price, float(feature["bar_range"]))
        future = intra.iloc[i + 1:i + 1 + horizon_bars]
        future = future[future.index.date == ts.date()]
        if len(future) < horizon_bars:
            continue

        base = evaluate_future_path(future, price, levels, round_trip_cost_bps=base_cost_bps)
        stress = evaluate_future_path(future, price, levels, round_trip_cost_bps=stress_cost_bps)
        if base is None or stress is None or base.get("ambiguous") or stress.get("ambiguous"):
            continue

        row = {k: float(feature[k]) for k in MODEL_FEATURES}
        row.update({
            "ticker": ticker,
            "timestamp": ts.isoformat(),
            "session_date": ts.date().isoformat(),
            # v12 predicts whether the planned setup finishes positive AFTER the base drag assumption.
            "positive_net": 1 if float(base["net_r"]) > 0 else 0,
            "target_hit": 1 if base["target_hit"] else 0,
            "stop_hit": 1 if base["stop_hit"] else 0,
            "net_r_base": float(base["net_r"]),
            "net_r_stress": float(stress["net_r"]),
            "max_upside": float(base["max_upside"]),
            "max_drawdown": float(base["max_drawdown"]),
            "final_return": float(base["final_return"]),
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
        print(f"Building historical snapshots: {ticker}")
        rows.extend(build_samples(
            ticker,
            daily.get(ticker, pd.DataFrame()),
            intra.get(ticker, pd.DataFrame()),
            spy,
            cfg,
        ))

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
        print(json.dumps(payload, indent=2))
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
        print(json.dumps(payload, indent=2))
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
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
