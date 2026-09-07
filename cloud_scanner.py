from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from clarity_engine import (
    calibrated_signal,
    daily_opportunity_completed,
    dollar_plan,
    get_many,
    intraday_quick,
    load_calibration,
    load_config,
    session_status,
    trade_levels,
)

NY = ZoneInfo("America/New_York")
BASE = Path(__file__).parent


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100*x:.0f}%"


def send_ntfy(topic: str, title: str, body: str, ticker: str | None = None, app_url: str | None = None, priority: int = 4) -> None:
    actions = []
    if app_url:
        actions.append({"action": "view", "label": "Open Clarity", "url": app_url, "clear": True})
    if ticker:
        actions.append({"action": "view", "label": "Open Robinhood", "url": f"https://robinhood.com/stocks/{ticker}", "clear": False})

    payload = {
        "topic": topic,
        "title": title,
        "message": body,
        "priority": int(priority),
        "tags": ["gem", "chart_with_upwards_trend"],
        "markdown": True,
    }
    if app_url:
        payload["click"] = app_url
    elif ticker:
        payload["click"] = f"https://robinhood.com/stocks/{ticker}"
    if actions:
        payload["actions"] = actions[:3]

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request("https://ntfy.sh/", data=data, method="POST", headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=20) as response:
        response.read()


def signal_for_slice(daily_df, intra_df, spy_df, calibration, bars_back: int):
    if bars_back <= 0:
        part = intra_df
        if intra_df is None or intra_df.empty:
            return None
        ts = intra_df.index[-1]
    else:
        if intra_df is None or len(intra_df) <= 30 + bars_back:
            return None
        part = intra_df.iloc[:-bars_back]
        ts = part.index[-1]

    spy_part = spy_df.loc[:ts] if spy_df is not None and not spy_df.empty else spy_df
    return calibrated_signal(daily_df, part, spy_part, calibration)


def selected_event_now(daily_df, intra_df, spy_df, calibration, config):
    """Return the current signal only if it is a fresh event after the v12 cooldown.

    This is state-free: each cloud run reconstructs recent threshold events from the
    downloaded bars, so GitHub Actions does not need a persistent alert-state file.
    """
    threshold = float(calibration.get("probability_threshold", 1.0))
    surge_margin = float(config.get("alert_surge_probability_margin", 0.08))
    cooldown_bars = max(1, int(config.get("alert_same_ticker_cooldown_minutes", 60)) // 5)
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

    # Reconstruct selected events from oldest -> newest with the same-ticker cooldown.
    selected_offsets = []
    last_selected_ts = None
    for offset in range(lookback, -1, -1):
        if not raw_event(offset):
            continue
        sig = cache[offset]
        try:
            ts = pd.Timestamp(sig["features"]["timestamp"])
        except Exception:
            continue
        if last_selected_ts is not None and ts - last_selected_ts < pd.Timedelta(minutes=cooldown_bars * 5):
            continue
        selected_offsets.append(offset)
        last_selected_ts = ts

    return cache.get(0) if 0 in selected_offsets else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(BASE / "cloud_config.json")
    calibration = load_calibration(BASE / "clarity_calibration.json")
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    app_url = os.environ.get("CLARITY_APP_URL", "").strip() or None

    if not topic:
        raise SystemExit("Missing GitHub secret NTFY_TOPIC.")

    if args.test:
        status = calibration.get("status", "missing")
        send_ntfy(
            topic,
            "Clarity evidence-engine test 💎",
            f"Phone alerts are connected. Historical evidence model status: **{status}**. No trade was placed.",
            app_url=app_url,
            priority=3,
        )
        print("Cloud notification sent.")
        return

    status = session_status(datetime.now(NY))
    if not status["open"]:
        print(status["label"])
        return

    if not calibration.get("ready_for_alerts", False):
        print("Historical calibration is not ready for live alerts. Run the evidence calibration workflow and review its out-of-sample result.")
        return

    watchlist = [str(x).upper() for x in config["watchlist"]]
    max_trade = float(os.environ.get("MAX_TRADE_AMOUNT", "0") or 0)
    max_loss = float(os.environ.get("MAX_PLANNED_LOSS", "0") or 0)

    daily = get_many(watchlist, "1y", "1d")
    intraday_names = list(dict.fromkeys(watchlist + ["SPY"]))
    intraday = get_many(intraday_names, "5d", "5m")
    spy = intraday.get("SPY", pd.DataFrame())

    candidates = []
    threshold = float(calibration.get("probability_threshold", 1.0))

    for ticker in watchlist:
        daily_df = daily.get(ticker, pd.DataFrame())
        intra_df = intraday.get(ticker, pd.DataFrame())
        sig = selected_event_now(daily_df, intra_df, spy, calibration, config)
        if not sig or not sig.get("ready"):
            continue
        prob = float(sig["probability"])
        quick = intraday_quick(intra_df, spy)
        opp = daily_opportunity_completed(daily_df, as_of=intra_df.index[-1] if not intra_df.empty else None)
        candidates.append((prob, ticker, opp, quick, sig))

    candidates.sort(reverse=True, key=lambda x: x[0])
    if not candidates:
        print("No newly crossed evidence-backed alert right now.")
        return

    prob, ticker, opp, quick, sig = candidates[0]
    if quick is None:
        return
    levels = trade_levels(quick["price"], quick["bar_range"])
    dollars = dollar_plan(quick["price"], levels, max_trade, max_loss)
    test = calibration.get("metrics", {}).get("test", {})

    body_lines = [
        f"**{ticker} passed Clarity's historical evidence line.**",
        f"Historical evidence strength **{sig.get('evidence_percentile', 0):.0f}th percentile** · alert line **{sig.get('threshold_percentile', 0):.0f}th percentile**",
        f"Latest **{money(quick['price'])}** · Don't chase above **{money(levels['entry_ceiling'])}**",
        f"Planning target **{money(levels['target'])}** · Exit-if-wrong **{money(levels['stop'])}**",
    ]

    if dollars:
        body_lines.append(f"If using about **{money(dollars['amount'])}**: target ≈ **+{money(dollars['target_profit'])}** · planned loss ≈ **-{money(dollars['planned_loss'])}**")

    if test.get("alerts"):
        body_lines.append(
            f"Untouched historical test: **{test['alerts']} separated alerts** · positive after base drag **{pct(test.get('positive_rate'))}** · average **{test.get('avg_r', 0):+.2f}R** · stress **{test.get('stress_avg_r', 0):+.2f}R**."
        )

    body_lines.append("**Why now:** " + " ".join(quick["reasons"][:2]))
    body_lines.append("_Historical fit is not a guarantee. Five-minute data can miss intrabar order and backtests can fail out of sample._")
    body = "\n\n".join(body_lines)

    print(f"Evidence alert candidate: {ticker}, percentile={sig.get('evidence_percentile')}, p={prob:.3f}, threshold={threshold:.3f}")
    if args.dry_run:
        print(body)
        return

    send_ntfy(topic, f"Clarity: {ticker} passed the evidence line", body, ticker=ticker, app_url=app_url, priority=4)
    print("Notification sent.")


if __name__ == "__main__":
    main()
