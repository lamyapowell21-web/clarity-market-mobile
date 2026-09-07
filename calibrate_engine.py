from __future__ import annotations

import json
import math
import time
from datetime import datetime
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
        yield items[i:i+size]


def fetch_batched(tickers: list[str], period: str, interval: str, batch_size: int = 10) -> dict[str, pd.DataFrame]:
    out = {}
    for batch in batched(tickers, batch_size):
        print(f"Downloading {interval}: {', '.join(batch)}")
        out.update(get_many(batch, period, interval))
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
    intra = normalize_intraday(intra_df)
    spy = normalize_intraday(spy_df)
    if intra.empty or len(intra) < 45:
        return []

    stride = int(cfg.get("calibration_stride_bars", 3))
    horizon_bars = max(1, int(cfg.get("backtest_horizon_minutes", 60)) // 5)
    start_minute = int(cfg.get("calibration_start_minute_et", 15))
    end_minute = int(cfg.get("calibration_end_minute_et", 350))
    round_trip_cost_bps = float(cfg.get("backtest_round_trip_cost_bps", 0.0))

    rows = []
    for i in range(30, len(intra) - horizon_bars, stride):
        ts = intra.index[i]
        minute = (ts.hour * 60 + ts.minute) - (9 * 60 + 30)
        if minute < start_minute or minute > end_minute:
            continue

        hist = intra.iloc[:i+1]
        spy_hist = spy.loc[:ts]
        if len(spy_hist) < 30:
            continue

        feature = model_feature_row(daily_df, hist, spy_hist, as_of=ts)
        if feature is None:
            continue

        price = float(feature["price"])
        levels = trade_levels(price, float(feature["bar_range"]))
        future = intra.iloc[i+1:i+1+horizon_bars]
        future = future[future.index.date == ts.date()]
        if len(future) < horizon_bars:
            continue
        outcome = evaluate_future_path(future, price, levels, round_trip_cost_bps=round_trip_cost_bps)
        if outcome is None or outcome.get("ambiguous"):
            continue

        row = {k: float(feature[k]) for k in MODEL_FEATURES}
        row.update({
            "ticker": ticker,
            "timestamp": ts.isoformat(),
            "target_hit": 1 if outcome["target_hit"] else 0,
            "net_r": float(outcome["net_r"]),
            "max_upside": float(outcome["max_upside"]),
            "max_drawdown": float(outcome["max_drawdown"]),
            "final_return": float(outcome["final_return"]),
        })
        rows.append(row)
    return rows


def split_chronologically(df: pd.DataFrame):
    d = df.sort_values("timestamp").reset_index(drop=True)
    n = len(d)
    train_end = int(n * 0.60)
    cal_end = int(n * 0.80)
    return d.iloc[:train_end].copy(), d.iloc[train_end:cal_end].copy(), d.iloc[cal_end:].copy()


def threshold_search(prob: np.ndarray, r: np.ndarray, y: np.ndarray, min_alerts: int):
    if len(prob) == 0:
        return None
    quantiles = np.linspace(0.55, 0.95, 25)
    candidates = sorted(set(float(np.quantile(prob, q)) for q in quantiles))
    best = None
    for thr in candidates:
        mask = prob >= thr
        n = int(mask.sum())
        if n < min_alerts:
            continue
        vals = r[mask]
        avg_r = float(np.mean(vals))
        std_r = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        stderr = std_r / math.sqrt(n) if n > 0 else 999.0
        lower_bound = avg_r - 0.75 * stderr
        hit = float(np.mean(y[mask]))
        candidate = {
            "threshold": thr,
            "alerts": n,
            "avg_r": avg_r,
            "target_hit_rate": hit,
            "lower_bound_score": lower_bound,
        }
        if best is None or candidate["lower_bound_score"] > best["lower_bound_score"]:
            best = candidate
    return best


def metrics_for(prob: np.ndarray, y: np.ndarray, r: np.ndarray, threshold: float):
    mask = prob >= threshold
    alerts = int(mask.sum())
    result = {
        "samples": int(len(y)),
        "base_target_hit_rate": float(np.mean(y)) if len(y) else None,
        "alerts": alerts,
        "target_hit_rate": float(np.mean(y[mask])) if alerts else None,
        "avg_r": float(np.mean(r[mask])) if alerts else None,
        "median_r": float(np.median(r[mask])) if alerts else None,
    }
    if len(np.unique(y)) > 1:
        result["auc"] = float(roc_auc_score(y, prob))
    else:
        result["auc"] = None
    return result


def main():
    cfg = load_config(BASE / "cloud_config.json")
    watchlist = [str(x).upper() for x in cfg["watchlist"]]
    calibration_days = int(cfg.get("calibration_days", 30))
    period = str(cfg.get("calibration_period", "1mo"))
    batch_size = int(cfg.get("calibration_download_batch_size", 10))

    daily = fetch_batched(watchlist, "1y", "1d", batch_size=batch_size)
    intraday_names = list(dict.fromkeys(watchlist + ["SPY"]))
    intra = fetch_batched(intraday_names, period, "5m", batch_size=batch_size)
    spy = intra.get("SPY", pd.DataFrame())

    rows = []
    for ticker in watchlist:
        print(f"Building historical samples: {ticker}")
        rows.extend(build_samples(
            ticker,
            daily.get(ticker, pd.DataFrame()),
            intra.get(ticker, pd.DataFrame()),
            spy,
            cfg,
        ))

    data = pd.DataFrame(rows)
    min_total = int(cfg.get("calibration_min_total_samples", 1500))
    out_path = BASE / "clarity_calibration.json"

    if len(data) < min_total or data["target_hit"].nunique() < 2:
        payload = {
            "version": 1,
            "status": "insufficient",
            "ready_for_alerts": False,
            "generated_at": datetime.now(NY).isoformat(),
            "samples": int(len(data)),
            "message": "Not enough historical samples to fit the evidence model yet.",
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    train, cal, test = split_chronologically(data)
    X_train = train[MODEL_FEATURES].astype(float).to_numpy()
    y_train = train["target_hit"].astype(int).to_numpy()
    X_cal = cal[MODEL_FEATURES].astype(float).to_numpy()
    y_cal = cal["target_hit"].astype(int).to_numpy()
    r_cal = cal["net_r"].astype(float).to_numpy()
    X_test = test[MODEL_FEATURES].astype(float).to_numpy()
    y_test = test["target_hit"].astype(int).to_numpy()
    r_test = test["net_r"].astype(float).to_numpy()

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(X_train)
    Z_cal = scaler.transform(X_cal)
    Z_test = scaler.transform(X_test)

    model = LogisticRegression(
        C=float(cfg.get("calibration_logistic_c", 0.5)),
        max_iter=2000,
        random_state=42,
    )
    model.fit(Z_train, y_train)
    p_cal = model.predict_proba(Z_cal)[:, 1]
    p_test = model.predict_proba(Z_test)[:, 1]

    choice = threshold_search(
        p_cal,
        r_cal,
        y_cal,
        min_alerts=int(cfg.get("calibration_min_threshold_alerts", 30)),
    )

    if choice is None:
        threshold = 1.0
        cal_metrics = metrics_for(p_cal, y_cal, r_cal, threshold)
        test_metrics = metrics_for(p_test, y_test, r_test, threshold)
        status = "weak"
        ready = False
    else:
        threshold = float(choice["threshold"])
        cal_metrics = metrics_for(p_cal, y_cal, r_cal, threshold)
        test_metrics = metrics_for(p_test, y_test, r_test, threshold)
        min_test_alerts = int(cfg.get("calibration_min_test_alerts", 20))
        positive_test_edge = (
            test_metrics["alerts"] >= min_test_alerts
            and test_metrics["avg_r"] is not None
            and test_metrics["avg_r"] > 0
        )
        status = "ready" if positive_test_edge else "weak"
        ready = bool(positive_test_edge)

    payload = {
        "version": 1,
        "status": status,
        "ready_for_alerts": ready,
        "generated_at": datetime.now(NY).isoformat(),
        "training_window_days": calibration_days,
        "horizon_minutes": int(cfg.get("backtest_horizon_minutes", 60)),
        "round_trip_cost_bps": float(cfg.get("backtest_round_trip_cost_bps", 0.0)),
        "feature_names": MODEL_FEATURES,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "coefficients": [float(x) for x in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "probability_threshold": float(threshold),
        "guardrails": {
            "require_positive_ret15": bool(cfg.get("require_positive_ret15", True)),
        },
        "splits": {
            "train_samples": int(len(train)),
            "calibration_samples": int(len(cal)),
            "test_samples": int(len(test)),
            "train_start": str(train["timestamp"].iloc[0]) if len(train) else None,
            "test_end": str(test["timestamp"].iloc[-1]) if len(test) else None,
        },
        "threshold_selection": choice,
        "metrics": {
            "calibration": cal_metrics,
            "test": test_metrics,
        },
        "notes": [
            "Features were chosen from momentum, volume, breakout, volatility, market-context and intraday-timing research.",
            "The alert threshold was selected on a middle calibration period and evaluated on a later untouched test period.",
            "Five-minute OHLC bars cannot reveal whether target or stop happened first when both occur in the same bar, so those samples are discarded.",
            "Backtest results are historical and do not guarantee future performance.",
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
