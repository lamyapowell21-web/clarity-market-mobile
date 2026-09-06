
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
    daily_opportunity,
    dollar_plan,
    get_many,
    intraday_quick,
    load_config,
    session_status,
    trade_levels,
)

NY = ZoneInfo("America/New_York")


def money(x: float) -> str:
    return f"${x:,.2f}"


def send_ntfy(topic: str, title: str, body: str, ticker: str | None = None, app_url: str | None = None, priority: int = 4) -> None:
    actions = []
    if app_url:
        actions.append({
            "action": "view",
            "label": "Open Clarity",
            "url": app_url,
            "clear": True,
        })
    if ticker:
        actions.append({
            "action": "view",
            "label": "Open Robinhood",
            "url": f"https://robinhood.com/stocks/{ticker}",
            "clear": False,
        })

    payload = {
        "topic": topic,
        "title": title,
        "message": body,
        "priority": int(priority),
        "tags": ["sparkles", "chart_with_upwards_trend"],
        "markdown": True,
    }
    if app_url:
        payload["click"] = app_url
    elif ticker:
        payload["click"] = f"https://robinhood.com/stocks/{ticker}"
    if actions:
        payload["actions"] = actions[:3]

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "https://ntfy.sh/",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        response.read()


def quick_score_for_slice(df: pd.DataFrame, spy: pd.DataFrame, bars_back: int) -> int | None:
    if bars_back <= 0:
        part = df
        spy_part = spy
    else:
        if len(df) <= 30 + bars_back:
            return None
        part = df.iloc[:-bars_back]
        spy_part = spy.iloc[:-bars_back] if spy is not None and len(spy) > bars_back else spy
    result = intraday_quick(part, spy_part)
    return result["score"] if result else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "cloud_config.json")
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    app_url = os.environ.get("CLARITY_APP_URL", "").strip() or None

    if not topic:
        raise SystemExit("Missing GitHub secret NTFY_TOPIC.")

    if args.test:
        send_ntfy(
            topic,
            "Clarity cloud test 💗",
            "Your S24 can receive alerts even when your personal computer is off. No trade was placed.",
            app_url=app_url,
            priority=3,
        )
        print("Cloud notification sent.")
        return

    status = session_status(datetime.now(NY))
    if not status["open"]:
        print(status["label"])
        return

    watchlist = [str(x).upper() for x in config["watchlist"]]
    opp_min = int(config["opportunity_min"])
    quick_min = int(config["quick_move_min"])

    max_trade = float(os.environ.get("MAX_TRADE_AMOUNT", "0") or 0)
    max_loss = float(os.environ.get("MAX_PLANNED_LOSS", "0") or 0)

    daily = get_many(watchlist, "1y", "1d")
    intra_names = list(dict.fromkeys(watchlist + ["SPY"]))
    intraday = get_many(intra_names, "5d", "5m")
    spy = intraday.get("SPY", pd.DataFrame())

    candidates = []

    for ticker in watchlist:
        daily_df = daily.get(ticker, pd.DataFrame())
        intra_df = intraday.get(ticker, pd.DataFrame())

        opp = daily_opportunity(daily_df)
        quick = intraday_quick(intra_df, spy)
        if opp is None or quick is None:
            continue

        previous = [
            quick_score_for_slice(intra_df, spy, 1),
            quick_score_for_slice(intra_df, spy, 2),
            quick_score_for_slice(intra_df, spy, 3),
        ]
        previous = [x for x in previous if x is not None]
        previous_high = max(previous) if previous else 0

        # Stateless deduplication:
        # alert when a setup crosses the threshold, rather than alerting every 5 minutes.
        crossed = quick["score"] >= quick_min and previous_high < quick_min

        # Also allow a sudden large jump in signal strength even if the previous bar
        # was barely over the threshold.
        surged = (
            quick["score"] >= quick_min + 8
            and quick["score"] - previous_high >= 8
        )

        if opp >= opp_min and (crossed or surged):
            combined = 0.45 * opp + 0.55 * quick["score"]
            candidates.append((combined, ticker, opp, quick, previous_high))

    candidates.sort(reverse=True, key=lambda x: x[0])

    if not candidates:
        best = []
        for ticker in watchlist:
            opp = daily_opportunity(daily.get(ticker, pd.DataFrame()))
            quick = intraday_quick(intraday.get(ticker, pd.DataFrame()), spy)
            if opp is not None and quick is not None:
                best.append((0.45 * opp + 0.55 * quick["score"], ticker, opp, quick["score"]))
        best.sort(reverse=True)
        if best:
            _, ticker, opp, q = best[0]
            print(f"No new alert crossing. Best right now: {ticker} opportunity={opp}, quick={q}")
        else:
            print("No usable market data.")
        return

    _, ticker, opp, quick, previous_high = candidates[0]
    levels = trade_levels(quick["price"], quick["bar_range"])
    dollars = dollar_plan(quick["price"], levels, max_trade, max_loss)

    body_lines = [
        f"**{ticker} is worth a 60-second review.**",
        f"Opportunity **{opp}/100** · Quick Move **{quick['score']}/100**",
        f"Latest **{money(quick['price'])}** · Don't chase above **{money(levels['entry_ceiling'])}**",
        f"Planning target **{money(levels['target'])}** · Exit-if-wrong **{money(levels['stop'])}**",
    ]

    if dollars:
        body_lines.append(
            f"If using about **{money(dollars['amount'])}**: target ≈ **+{money(dollars['target_profit'])}** · planned loss ≈ **-{money(dollars['planned_loss'])}**"
        )
    else:
        body_lines.append("Open Clarity to enter your own dollar amount and see the matching gain/loss example.")

    body_lines.append("**Why now:** " + " ".join(quick["reasons"][:3]))
    body_lines.append("_Planning aid only. A fast move can reverse and prices can jump through an exit level._")

    body = "\n\n".join(body_lines)

    print(f"Alert candidate: {ticker}, opportunity={opp}, quick={quick['score']}, previous_high={previous_high}")

    if args.dry_run:
        print(body)
        return

    send_ntfy(
        topic,
        f"Clarity: {ticker} worth a look",
        body,
        ticker=ticker,
        app_url=app_url,
        priority=4,
    )
    print("Notification sent.")


if __name__ == "__main__":
    main()
