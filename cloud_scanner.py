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
        sig = signal_for_slice(daily_df, intra_df, spy, calibration, 0)
        if not sig or not sig.get("available"):
            continue

        previous = []
        for bars_back in [1, 2, 3]:
            prev = signal_for_slice(daily_df, intra_df, spy, calibration, bars_back)
            if prev and prev.get("probability") is not None:
                previous.append(float(prev["probability"]))
        previous_high = max(previous) if previous else 0.0
        prob = float(sig["probability"])

        crossed = prob >= threshold and previous_high < threshold
        surged = prob >= threshold + 0.08 and prob - previous_high >= 0.08

        if sig.get("ready") and (crossed or surged):
            quick = intraday_quick(intra_df, spy)
            opp = daily_opportunity_completed(daily_df, as_of=intra_df.index[-1] if not intra_df.empty else None)
            candidates.append((prob, ticker, opp, quick, sig, previous_high))

    candidates.sort(reverse=True, key=lambda x: x[0])
    if not candidates:
        print("No newly crossed evidence-backed alert right now.")
        return

    prob, ticker, opp, quick, sig, previous_high = candidates[0]
    if quick is None:
        return
    levels = trade_levels(quick["price"], quick["bar_range"])
    dollars = dollar_plan(quick["price"], levels, max_trade, max_loss)
    test = calibration.get("metrics", {}).get("test", {})

    body_lines = [
        f"**{ticker} passed Clarity's historical evidence line.**",
        f"Backtest-model estimate **{pct(prob)}** · alert line **{pct(threshold)}**",
        f"Latest **{money(quick['price'])}** · Don't chase above **{money(levels['entry_ceiling'])}**",
        f"Planning target **{money(levels['target'])}** · Exit-if-wrong **{money(levels['stop'])}**",
    ]

    if dollars:
        body_lines.append(f"If using about **{money(dollars['amount'])}**: target ≈ **+{money(dollars['target_profit'])}** · planned loss ≈ **-{money(dollars['planned_loss'])}**")

    if test.get("alerts"):
        body_lines.append(
            f"Untouched historical test: **{test['alerts']} alerts** · target-first rate **{pct(test.get('target_hit_rate'))}** · average **{test.get('avg_r', 0):+.2f}R**."
        )

    body_lines.append("**Why now:** " + " ".join(quick["reasons"][:2]))
    body_lines.append("_Historical fit is not a guarantee. Five-minute data can miss intrabar order and backtests can fail out of sample._")
    body = "\n\n".join(body_lines)

    print(f"Evidence alert candidate: {ticker}, p={prob:.3f}, threshold={threshold:.3f}, previous={previous_high:.3f}")
    if args.dry_run:
        print(body)
        return

    send_ntfy(topic, f"Clarity: {ticker} passed the evidence line", body, ticker=ticker, app_url=app_url, priority=4)
    print("Notification sent.")


if __name__ == "__main__":
    main()
