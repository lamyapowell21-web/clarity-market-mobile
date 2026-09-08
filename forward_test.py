from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from clarity_engine import (
    MODEL_FEATURES,
    calibrated_signal,
    daily_opportunity_completed,
    evaluate_future_path,
    get_many,
    intraday_quick,
    load_calibration,
    load_config,
    session_status,
    trade_levels,
)

NY = ZoneInfo("America/New_York")
BASE = Path(__file__).parent
EVENTS_PATH = BASE / "forward_test_events.csv"
SUMMARY_PATH = BASE / "forward_test_summary.json"

EVENT_COLUMNS = [
    "event_id","observed_at_et","signal_timestamp_et","session_date","ticker","status",
    "horizon_minutes","model_probability","evidence_percentile","probability_threshold",
    "threshold_percentile","entry_price","stop_price","target_price","risk_per_share",
    "opportunity_score","quick_move_score","ret15","ret60m","vwap_distance",
    "relative_ret15_volnorm","relative_ret60_volnorm","log_intraday_volume_ratio",
    "volume_acceleration","opening_range_position","opening_range_breakout_distance",
    "market_ret15","market_ret60m","range_pct","minutes_from_open","base_cost_bps",
    "stress_cost_bps","outcome_timestamp_et","target_hit","stop_hit","ambiguous",
    "net_r_base","net_r_stress","final_return","max_upside","max_downside","feature_json"
]


def _blank_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists() or EVENTS_PATH.stat().st_size == 0:
        return _blank_events()
    try:
        d = pd.read_csv(EVENTS_PATH, dtype=str, keep_default_na=False)
    except Exception:
        return _blank_events()
    for c in EVENT_COLUMNS:
        if c not in d.columns:
            d[c] = ""
    return d[EVENT_COLUMNS].copy()


def save_events(events: pd.DataFrame) -> None:
    out = events.copy()
    for c in EVENT_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    out[EVENT_COLUMNS].to_csv(EVENTS_PATH, index=False)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
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


def _f(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def signal_for_slice(daily_df, intra_df, spy_df, calibration, bars_back: int):
    if intra_df is None or intra_df.empty:
        return None
    if bars_back <= 0:
        part = intra_df
        ts = intra_df.index[-1]
    else:
        if len(intra_df) <= 30 + bars_back:
            return None
        part = intra_df.iloc[:-bars_back]
        ts = part.index[-1]

    spy_part = spy_df.loc[:ts] if spy_df is not None and not spy_df.empty else spy_df
    return calibrated_signal(daily_df, part, spy_part, calibration)


def shadow_event_now(daily_df, intra_df, spy_df, calibration, config):
    """Fresh v13 threshold-crossing/surge event, even while live alerts are disabled."""
    threshold = float(calibration.get("probability_threshold", 1.0))
    guardrails = calibration.get("guardrails", {})
    surge_margin = float(guardrails.get(
        "surge_probability_margin",
        config.get("alert_surge_probability_margin", 0.05),
    ))
    cooldown_minutes = int(guardrails.get(
        "same_ticker_cooldown_minutes",
        config.get("alert_same_ticker_cooldown_minutes", 60),
    ))
    cooldown_bars = max(1, cooldown_minutes // 5)
    lookback = cooldown_bars * 2 + 3

    cache = {}
    for bars_back in range(0, lookback + 4):
        sig = signal_for_slice(daily_df, intra_df, spy_df, calibration, bars_back)
        if sig and sig.get("available") and sig.get("probability") is not None:
            cache[bars_back] = sig

    def raw_event(offset: int) -> bool:
        sig = cache.get(offset)
        if not sig or not sig.get("directional_ok", False):
            return False
        p = float(sig["probability"])
        previous = [
            float(cache[j]["probability"])
            for j in range(offset + 1, offset + 4)
            if j in cache and cache[j].get("probability") is not None
        ]
        previous_high = max(previous) if previous else 0.0
        crossed = p >= threshold and previous_high < threshold
        surged = p >= threshold + surge_margin and p - previous_high >= surge_margin
        return bool(crossed or surged)

    selected_offsets = []
    last_selected_ts = None
    for offset in range(lookback, -1, -1):
        if not raw_event(offset):
            continue
        sig = cache[offset]
        try:
            ts = pd.Timestamp(sig["features"]["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize(NY)
            else:
                ts = ts.tz_convert(NY)
        except Exception:
            continue
        if last_selected_ts is not None and ts - last_selected_ts < pd.Timedelta(minutes=cooldown_minutes):
            continue
        selected_offsets.append(offset)
        last_selected_ts = ts

    return cache.get(0) if 0 in selected_offsets else None


def settle_pending(events: pd.DataFrame, intraday: dict[str, pd.DataFrame], config: dict) -> tuple[pd.DataFrame, int]:
    if events.empty:
        return events, 0

    changed = 0
    base_cost = float(config.get("backtest_round_trip_cost_bps", 5.0))
    stress_cost = float(config.get("backtest_stress_round_trip_cost_bps", 15.0))

    for i in events.index:
        if str(events.at[i, "status"]).strip().lower() != "pending":
            continue

        ticker = str(events.at[i, "ticker"]).upper()
        d = _normalize(intraday.get(ticker, pd.DataFrame()))
        if d.empty:
            continue

        try:
            signal_ts = pd.Timestamp(events.at[i, "signal_timestamp_et"])
            if signal_ts.tzinfo is None:
                signal_ts = signal_ts.tz_localize(NY)
            else:
                signal_ts = signal_ts.tz_convert(NY)
            horizon = int(float(events.at[i, "horizon_minutes"]))
        except Exception:
            continue

        end_ts = signal_ts + pd.Timedelta(minutes=horizon)
        future = d[(d.index > signal_ts) & (d.index <= end_ts)].copy()
        expected_bars = max(1, horizon // 5)
        if len(future) < expected_bars:
            continue

        entry = _f(events.at[i, "entry_price"])
        stop = _f(events.at[i, "stop_price"])
        target = _f(events.at[i, "target_price"])
        if not np.isfinite(entry) or not np.isfinite(stop) or not np.isfinite(target):
            continue

        levels = {
            "stop": stop,
            "target": target,
            "entry_ceiling": entry,
            "stop_distance": max(1e-9, entry - stop),
        }
        base = evaluate_future_path(future, entry, levels, round_trip_cost_bps=base_cost)
        stress = evaluate_future_path(future, entry, levels, round_trip_cost_bps=stress_cost)
        if not base or not stress:
            continue

        events.at[i, "outcome_timestamp_et"] = end_ts.isoformat()
        events.at[i, "ambiguous"] = str(bool(base.get("ambiguous", False))).lower()

        if base.get("ambiguous", False) or stress.get("ambiguous", False):
            events.at[i, "status"] = "ambiguous"
            changed += 1
            continue

        events.at[i, "status"] = "settled"
        events.at[i, "target_hit"] = str(bool(base.get("target_hit", False))).lower()
        events.at[i, "stop_hit"] = str(bool(base.get("stop_hit", False))).lower()
        events.at[i, "net_r_base"] = f"{float(base.get('net_r', np.nan)):.10f}"
        events.at[i, "net_r_stress"] = f"{float(stress.get('net_r', np.nan)):.10f}"
        events.at[i, "max_upside"] = f"{float(base.get('max_upside', np.nan)):.10f}"
        events.at[i, "max_downside"] = f"{float(base.get('max_downside', np.nan)):.10f}"

        try:
            last_close = float(pd.to_numeric(future["Close"], errors="coerce").dropna().iloc[-1])
            events.at[i, "final_return"] = f"{(last_close / entry - 1.0):.10f}"
        except Exception:
            events.at[i, "final_return"] = ""

        changed += 1

    return events, changed


def event_row(ticker: str, sig: dict, daily_df: pd.DataFrame, intra_df: pd.DataFrame, spy_df: pd.DataFrame,
              calibration: dict, config: dict) -> dict:
    feat = sig["features"]
    ts = pd.Timestamp(feat["timestamp"])
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY)
    else:
        ts = ts.tz_convert(NY)

    quick = intraday_quick(intra_df, spy_df)
    opp = daily_opportunity_completed(daily_df, as_of=ts)
    entry = float(feat["price"])
    levels = trade_levels(entry, float(feat["bar_range"]))
    horizon = int(calibration.get("horizon_minutes", 15))
    risk = max(1e-9, entry - float(levels["stop"]))

    feature_payload = {}
    for name in MODEL_FEATURES:
        val = feat.get(name)
        try:
            feature_payload[name] = None if val is None else float(val)
        except Exception:
            feature_payload[name] = None

    return {
        "event_id": f"{ticker}|{ts.isoformat()}",
        "observed_at_et": datetime.now(NY).isoformat(),
        "signal_timestamp_et": ts.isoformat(),
        "session_date": str(ts.date()),
        "ticker": ticker,
        "status": "pending",
        "horizon_minutes": horizon,
        "model_probability": f"{float(sig['probability']):.10f}",
        "evidence_percentile": "" if sig.get("evidence_percentile") is None else f"{float(sig['evidence_percentile']):.4f}",
        "probability_threshold": f"{float(sig['threshold']):.10f}",
        "threshold_percentile": "" if sig.get("threshold_percentile") is None else f"{float(sig['threshold_percentile']):.4f}",
        "entry_price": f"{entry:.10f}",
        "stop_price": f"{float(levels['stop']):.10f}",
        "target_price": f"{float(levels['target']):.10f}",
        "risk_per_share": f"{risk:.10f}",
        "opportunity_score": "" if opp is None else int(opp),
        "quick_move_score": "" if quick is None else int(quick["score"]),
        "ret15": f"{float(feat.get('ret15', np.nan)):.10f}",
        "ret60m": f"{float(feat.get('ret60m', np.nan)):.10f}",
        "vwap_distance": f"{float(feat.get('vwap_distance', np.nan)):.10f}",
        "relative_ret15_volnorm": f"{float(feat.get('relative_ret15_volnorm', np.nan)):.10f}",
        "relative_ret60_volnorm": f"{float(feat.get('relative_ret60_volnorm', np.nan)):.10f}",
        "log_intraday_volume_ratio": f"{float(feat.get('log_intraday_volume_ratio', np.nan)):.10f}",
        "volume_acceleration": f"{float(feat.get('volume_acceleration', np.nan)):.10f}",
        "opening_range_position": f"{float(feat.get('opening_range_position', np.nan)):.10f}",
        "opening_range_breakout_distance": f"{float(feat.get('opening_range_breakout_distance', np.nan)):.10f}",
        "market_ret15": f"{float(feat.get('market_ret15', np.nan)):.10f}",
        "market_ret60m": f"{float(feat.get('market_ret60m', np.nan)):.10f}",
        "range_pct": f"{float(feat.get('range_pct', np.nan)):.10f}",
        "minutes_from_open": f"{float(feat.get('minutes_from_open', np.nan)):.2f}",
        "base_cost_bps": float(config.get("backtest_round_trip_cost_bps", 5.0)),
        "stress_cost_bps": float(config.get("backtest_stress_round_trip_cost_bps", 15.0)),
        "outcome_timestamp_et": "",
        "target_hit": "",
        "stop_hit": "",
        "ambiguous": "",
        "net_r_base": "",
        "net_r_stress": "",
        "final_return": "",
        "max_upside": "",
        "max_downside": "",
        "feature_json": json.dumps(feature_payload, separators=(",", ":"), sort_keys=True),
    }


def summarize(events: pd.DataFrame, calibration: dict) -> dict:
    settled = events[events["status"].str.lower() == "settled"].copy() if not events.empty else _blank_events()
    ambiguous = int((events["status"].str.lower() == "ambiguous").sum()) if not events.empty else 0
    pending = int((events["status"].str.lower() == "pending").sum()) if not events.empty else 0

    summary = {
        "version": 1,
        "method": "v13_live_shadow_forward_test",
        "calibration_method": calibration.get("method"),
        "calibration_generated_at": calibration.get("generated_at"),
        "horizon_minutes": int(calibration.get("horizon_minutes", 15)),
        "updated_at_et": datetime.now(NY).isoformat(),
        "events_total": int(len(events)),
        "settled_events": int(len(settled)),
        "pending_events": pending,
        "ambiguous_events": ambiguous,
        "distinct_settled_days": 0,
        "status": "collecting",
        "note": "Shadow observations only. No trade alerts are enabled by this forward-test file.",
    }

    if settled.empty:
        return summary

    base = pd.to_numeric(settled["net_r_base"], errors="coerce")
    stress = pd.to_numeric(settled["net_r_stress"], errors="coerce")
    final_ret = pd.to_numeric(settled["final_return"], errors="coerce")
    valid = base.notna()
    s = settled.loc[valid].copy()
    base = base.loc[valid]
    stress = stress.loc[valid]
    final_ret = final_ret.loc[valid]

    summary.update({
        "settled_events": int(len(s)),
        "distinct_settled_days": int(s["session_date"].nunique()),
        "positive_rate_base": float((base > 0).mean()) if len(base) else None,
        "average_net_r_base": float(base.mean()) if len(base) else None,
        "median_net_r_base": float(base.median()) if len(base) else None,
        "average_net_r_stress": float(stress.mean()) if len(stress) else None,
        "positive_final_return_rate": float((final_ret > 0).mean()) if len(final_ret) else None,
    })

    if len(s) >= 20 and s["session_date"].nunique() >= 5:
        summary["status"] = "enough_for_first_review"
    if len(s) >= 40 and s["session_date"].nunique() >= 10:
        summary["status"] = "enough_for_stronger_review"
    return summary


def write_summary(events: pd.DataFrame, calibration: dict) -> None:
    SUMMARY_PATH.write_text(json.dumps(summarize(events, calibration), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        idx = pd.date_range("2026-09-08 10:00", periods=3, freq="5min", tz=NY)
        future = pd.DataFrame(
            {"High": [100.4, 100.7, 100.8], "Low": [99.9, 100.0, 100.2], "Close": [100.3, 100.5, 100.6]},
            index=idx,
        )
        levels = {"stop": 99.0, "target": 102.0, "entry_ceiling": 100.0, "stop_distance": 1.0}
        result = evaluate_future_path(future, 100.0, levels, 5.0)
        assert result and not result["ambiguous"] and result["net_r"] > 0
        print("Forward-test self-test passed.")
        return

    config = load_config(BASE / "cloud_config.json")
    calibration = load_calibration(BASE / "clarity_calibration.json")
    if int(calibration.get("version", 0) or 0) < 3 or calibration.get("method") != "v13_market_structure_multihorizon":
        raise SystemExit("Forward test requires the completed v13 calibration.")

    watchlist = [str(x).upper() for x in config["watchlist"]]
    names = list(dict.fromkeys(watchlist + ["SPY"]))
    intraday = get_many(names, "5d", "5m")
    if not intraday:
        raise SystemExit("Could not download intraday data for the forward test.")

    events = load_events()
    events, settled_count = settle_pending(events, intraday, config)
    changed = settled_count > 0

    now = datetime.now(NY)
    market = session_status(now)

    if market.get("open"):
        daily = get_many(watchlist, "1y", "1d")
        spy = _normalize(intraday.get("SPY", pd.DataFrame()))
        candidates = []

        for ticker in watchlist:
            daily_df = daily.get(ticker, pd.DataFrame())
            intra_df = _normalize(intraday.get(ticker, pd.DataFrame()))
            if daily_df is None or daily_df.empty or intra_df.empty:
                continue
            sig = shadow_event_now(daily_df, intra_df, spy, calibration, config)
            if not sig or not sig.get("available") or sig.get("probability") is None:
                continue
            candidates.append((float(sig["probability"]), ticker, sig, daily_df, intra_df))

        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            _, ticker, sig, daily_df, intra_df = candidates[0]
            row = event_row(ticker, sig, daily_df, intra_df, spy, calibration, config)
            existing = set(events["event_id"].astype(str)) if not events.empty else set()
            if row["event_id"] not in existing:
                events = pd.concat([events, pd.DataFrame([row])], ignore_index=True)
                changed = True
                print(
                    f"Shadow event recorded: {ticker} · "
                    f"evidence {float(sig.get('evidence_percentile') or 0):.1f}th percentile · "
                    f"{row['signal_timestamp_et']}"
                )
            else:
                print("Current shadow event was already recorded.")
        else:
            print("No fresh v13 threshold-crossing/surge event at this scan.")
    else:
        print(str(market.get("label", "Market closed")) + " · settlement check only.")

    if changed:
        save_events(events)
        write_summary(events, calibration)
        print(
            f"Forward-test files updated · total={len(events)} · "
            f"settled_now={settled_count} · pending={(events['status'].str.lower() == 'pending').sum()}"
        )
    else:
        print("No forward-test file changes this run.")


if __name__ == "__main__":
    main()
