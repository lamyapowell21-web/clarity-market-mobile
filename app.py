
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

from clarity_engine import (
    daily_opportunity,
    dollar_plan,
    get_many,
    intraday_quick,
    load_config,
    market_mood,
    session_status,
    trade_levels,
)

NY = ZoneInfo("America/New_York")
BASE = Path(__file__).parent
CONFIG = load_config(BASE / "cloud_config.json")

st.set_page_config(
    page_title="Clarity",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink:#5F4450;
        --muted:#8B6D78;
        --line:#F0D9E3;

        --pink:#E88BB3;
        --pink-deep:#C94F88;
        --pink-soft:#FFF3F8;

        --yellow:#F8E38E;
        --yellow-soft:#FFF6C9;
        --yellow-pale:#FFFBEA;

        --cream:#FFFDF7;
        --cream-2:#FFF9EF;

        --good:#EAF7ED;
        --good-ink:#34694B;
        --warn:#FFF4D2;
        --warn-ink:#705A21;
        --quiet:#FFF9E8;
        --bad:#FFE6EC;
        --bad-ink:#8A455D;
      }

      .stApp {
        background:
          radial-gradient(circle at 0% 0%, rgba(255,241,170,.68), transparent 28%),
          radial-gradient(circle at 100% 6%, rgba(255,223,236,.55), transparent 25%),
          linear-gradient(180deg,#FFFDF7 0%,#FFFEFB 46%,#FFF9F1 100%);
        color:var(--ink);
      }

      .block-container {
        max-width:760px;
        padding-top:.65rem;
        padding-left:.75rem;
        padding-right:.75rem;
        padding-bottom:4rem;
      }

      h1,h2,h3,h4 {
        color:var(--ink);
        letter-spacing:-.025em;
      }

      .hero {
        padding:1.05rem 1.1rem;
        border:1px solid var(--line);
        border-radius:23px;
        background:
          linear-gradient(
            135deg,
            rgba(255,255,255,.97),
            rgba(255,248,214,.95) 52%,
            rgba(255,240,247,.94)
          );
        box-shadow:0 10px 28px rgba(178,77,139,.08);
        margin-bottom:.65rem;
      }

      .hero h1 {
        margin:0;
        font-size:1.85rem;
        color:var(--pink-deep);
      }

      .hero p {
        margin:.25rem 0 0;
        color:var(--muted);
        font-size:.92rem;
      }

      .card,
      .green-card,
      .yellow-card,
      .quiet-card {
        padding:.9rem .95rem;
        border:1px solid var(--line);
        border-radius:19px;
        margin:.45rem 0;
        box-shadow:0 6px 18px rgba(178,77,139,.05);
      }

      .card {
        background:rgba(255,255,255,.92);
      }

      .green-card {
        background:linear-gradient(135deg,var(--good),#FFFDF8);
        color:var(--good-ink);
      }

      .yellow-card {
        background:linear-gradient(135deg,var(--warn),#FFFBEA);
        color:var(--warn-ink);
      }

      .quiet-card {
        background:
          linear-gradient(135deg,#FFF9E6,#FFF3F8);
      }

      .big {
        font-size:1.2rem;
        font-weight:800;
      }

      .muted {
        color:var(--muted);
        font-size:.87rem;
      }

      .pill {
        display:inline-block;
        padding:.27rem .58rem;
        border:1px solid var(--line);
        border-radius:999px;
        background:linear-gradient(90deg,#FFF2C7,#FFE8F2);
        color:var(--ink);
        font-weight:750;
        margin-right:.3rem;
        margin-bottom:.2rem;
      }

      div[data-testid="stMetric"] {
        border:1px solid var(--line);
        border-radius:17px;
        padding:.65rem .7rem;
        background:rgba(255,255,255,.92);
      }

      div[data-testid="stMetric"] label {
        color:var(--muted)!important;
      }

      .stButton>button,
      .stLinkButton>a {
        border-radius:14px!important;
        border:1px solid #E5BFCF!important;
        background:linear-gradient(90deg,#F6D675,#E88BB3)!important;
        color:#5F4450!important;
        font-weight:750!important;
      }

      .stButton>button:hover,
      .stLinkButton>a:hover {
        border-color:#D96A9C!important;
        color:#5F4450!important;
      }

      .stTabs [data-baseweb="tab-list"] {
        gap:.35rem;
        overflow-x:auto;
        flex-wrap:nowrap;
      }

      .stTabs [data-baseweb="tab"] {
        white-space:nowrap;
        border:1px solid var(--line);
        background:rgba(255,255,255,.80);
        border-radius:999px;
        padding-left:.8rem;
        padding-right:.8rem;
        color:var(--ink);
      }

      .stTabs [aria-selected="true"] {
        background:#FFE8F2!important;
        border-color:#E5C7D3!important;
      }

      div[data-baseweb="select"] > div,
      .stTextInput input,
      .stNumberInput input,
      textarea {
        border-color:#E7CCD8!important;
        border-radius:12px!important;
        background:#FFFDFB!important;
      }

      div[data-testid="stAlert"] {
        border-color:#EFD5DF!important;
      }

      hr {
        border-color:#F0DDE5!important;
      }

      @media (max-width:600px) {
        .block-container { padding-top:.45rem; }
        .hero { border-radius:19px; }
        .hero h1 { font-size:1.65rem; }
        h2 { font-size:1.28rem!important; }
        h3 { font-size:1.05rem!important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if "trade_amount" not in st.session_state:
    st.session_state.trade_amount = 0.0
if "max_loss" not in st.session_state:
    st.session_state.max_loss = 0.0
if "account_type" not in st.session_state:
    st.session_state.account_type = "Cash"


@st.cache_data(ttl=240, show_spinner=False)
def cached_many(tickers: tuple[str, ...], period: str, interval: str):
    return get_many(list(tickers), period, interval)


def money(x: float) -> str:
    return f"${x:,.2f}"


def stock_scan(tickers: list[str]):
    daily = cached_many(tuple(tickers), "1y", "1d")
    intra_names = list(dict.fromkeys(tickers + ["SPY"]))
    intra = cached_many(tuple(intra_names), "5d", "5m")
    spy = intra.get("SPY", pd.DataFrame())

    rows = []
    for ticker in tickers:
        opp = daily_opportunity(daily.get(ticker, pd.DataFrame()))
        quick = intraday_quick(intra.get(ticker, pd.DataFrame()), spy)
        if opp is None or quick is None:
            continue

        rows.append(
            {
                "ticker": ticker,
                "opportunity": opp,
                "quick": quick["score"],
                "price": quick["price"],
                "quick_data": quick,
                "combined": .45 * opp + .55 * quick["score"],
            }
        )

    return sorted(rows, key=lambda x: x["combined"], reverse=True), daily, intra


def render_trade_card(item: dict):
    ticker = item["ticker"]
    q = item["quick_data"]

    levels = trade_levels(q["price"], q["bar_range"])
    dollars = dollar_plan(
        q["price"],
        levels,
        st.session_state.trade_amount,
        st.session_state.max_loss,
    )

    ready = (
        item["opportunity"] >= int(CONFIG["opportunity_min"])
        and item["quick"] >= int(CONFIG["quick_move_min"])
    )

    css = "green-card" if ready else "yellow-card"
    label = "REVIEW NOW" if ready else "WATCH — NOT READY"

    st.markdown(
        f"""
        <div class="{css}">
          <div class="big">{label} · {ticker}</div>
          <div style="margin-top:.35rem;">
            <span class="pill">Opportunity {item['opportunity']}/100</span>
            <span class="pill">Quick Move {item['quick']}/100</span>
          </div>
          <div class="muted" style="margin-top:.45rem;">
            Latest {money(q['price'])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("**Why Clarity noticed it**")
    for reason in q["reasons"][:3]:
        st.write("✦ " + reason)

    a, b = st.columns(2)
    a.metric("Don't chase above", money(levels["entry_ceiling"]))
    b.metric("Planning target", money(levels["target"]))

    c, d = st.columns(2)
    c.metric("Exit-if-wrong", money(levels["stop"]))
    d.metric("Latest price", money(q["price"]))

    if dollars:
        st.markdown(
            f"""
            <div class="card">
              With about <strong>{money(dollars['amount'])}</strong>:<br>
              If target is reached ≈ <strong>+{money(dollars['target_profit'])}</strong><br>
              If exit level is reached ≈ <strong>-{money(dollars['planned_loss'])}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Add the amount you want to use in Settings to see a dollar gain/loss example."
        )

    st.link_button(
        f"Open {ticker} in Robinhood",
        f"https://robinhood.com/stocks/{ticker}",
        width="stretch",
    )

    st.caption(
        "These are planning levels based on recent movement, not guaranteed prices or profits."
    )


st.markdown(
    """
    <div class="hero">
      <h1>Clarity</h1>
      <p>Your 60-second market check. The cloud scanner can keep watching even when your personal computer is off.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

today_tab, soon_tab, check_tab, settings_tab = st.tabs(
    ["Today", "Could move soon", "Check a stock", "Settings"]
)

watchlist = [str(x).upper() for x in CONFIG["watchlist"]]


with today_tab:
    status = session_status(datetime.now(NY))

    if status["open"]:
        st.markdown(
            f'<div class="green-card"><div class="big">🌸 {status["label"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="quiet-card">
              <div class="big">🌙 {status["label"]}</div>
              <div class="muted">
                Clarity will not send trade alerts while the NYSE is closed.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.spinner("Checking the market..."):
        broad = cached_many(("SPY", "^VIX"), "1y", "1d")
        mood = market_mood(broad)

    st.markdown(
        f"""
        <div class="card">
          <div class="muted">Wider market</div>
          <div class="big">{mood['label']}</div>
          <div class="muted">{mood['detail']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if status["open"]:
        with st.spinner("Looking for anything worth interrupting work for..."):
            ranked, _, _ = stock_scan(watchlist)

        if not ranked:
            st.warning("Clarity could not read enough intraday data right now.")
        else:
            best = ranked[0]

            ready = (
                best["opportunity"] >= int(CONFIG["opportunity_min"])
                and best["quick"] >= int(CONFIG["quick_move_min"])
            )

            if ready:
                render_trade_card(best)
            else:
                st.markdown(
                    f"""
                    <div class="quiet-card">
                      <div class="big">
                        Nothing worth interrupting work for right now.
                      </div>
                      <div class="muted">
                        Best current setup: {best['ticker']} ·
                        Opportunity {best['opportunity']}/100 ·
                        Quick Move {best['quick']}/100.
                        Clarity's alert line is
                        {CONFIG['opportunity_min']} + {CONFIG['quick_move_min']}.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.caption("You can still use Check a stock while the market is closed.")


with soon_tab:
    st.subheader("Could move soon")
    st.write(
        "This ranks fast-moving setups. A high score means movement is building — not that profit is guaranteed."
    )

    with st.spinner("Comparing the watchlist..."):
        ranked, _, _ = stock_scan(watchlist)

    if ranked:
        for item in ranked[:8]:
            with st.expander(
                f"{item['ticker']} · Opportunity {item['opportunity']} · Quick Move {item['quick']}"
            ):
                st.write(f"Latest: **{money(item['price'])}**")

                for reason in item["quick_data"]["reasons"][:3]:
                    st.write("✦ " + reason)

                if item["quick_data"]["cautions"]:
                    st.write("**Keep in mind**")
                    for caution in item["quick_data"]["cautions"]:
                        st.write("• " + caution)


with check_tab:
    st.subheader("Check one stock")

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()

    if ticker:
        with st.spinner(f"Checking {ticker}..."):
            daily = cached_many((ticker,), "1y", "1d")
            intra = cached_many((ticker, "SPY"), "5d", "5m")

            opp = daily_opportunity(daily.get(ticker, pd.DataFrame()))

            quick = intraday_quick(
                intra.get(ticker, pd.DataFrame()),
                intra.get("SPY", pd.DataFrame()),
            )

        if opp is None or quick is None:
            st.error("I couldn't read enough data for that ticker right now.")
        else:
            item = {
                "ticker": ticker,
                "opportunity": opp,
                "quick": quick["score"],
                "price": quick["price"],
                "quick_data": quick,
                "combined": .45 * opp + .55 * quick["score"],
            }

            render_trade_card(item)

            chart_df = intra.get(ticker, pd.DataFrame()).tail(78).reset_index()

            if not chart_df.empty:
                date_col = chart_df.columns[0]

                chart = (
                    alt.Chart(chart_df)
                    .mark_line(strokeWidth=2.5, color="#E88BB3")
                    .encode(
                        x=alt.X(f"{date_col}:T", title=None),
                        y=alt.Y(
                            "Close:Q",
                            title="Price",
                            scale=alt.Scale(zero=False),
                        ),
                        tooltip=[
                            alt.Tooltip(f"{date_col}:T", title="Time"),
                            alt.Tooltip(
                                "Close:Q",
                                title="Price",
                                format="$.2f",
                            ),
                        ],
                    )
                    .properties(height=270)
                )

                st.altair_chart(chart, width="stretch")


with settings_tab:
    st.subheader("Your quick-trade settings")

    st.caption(
        "These stay in the current app session. They are not sent to the cloud scanner unless you separately add matching GitHub secrets."
    )

    st.session_state.trade_amount = st.number_input(
        "Amount you may want to use on one trade",
        min_value=0.0,
        value=float(st.session_state.trade_amount),
        step=10.0,
        help="Leave at $0 if you only want price levels.",
    )

    st.session_state.max_loss = st.number_input(
        "Most you want the planning example to lose",
        min_value=0.0,
        value=float(st.session_state.max_loss),
        step=1.0,
        help="Leave at $0 if you do not want Clarity to size the example around a dollar-loss limit.",
    )

    st.session_state.account_type = st.selectbox(
        "Robinhood account type",
        ["Cash", "Margin"],
        index=0 if st.session_state.account_type == "Cash" else 1,
    )

    if st.session_state.account_type == "Cash":
        st.info(
            "Cash-account sale proceeds generally need to settle before you can reuse them. Clarity does not count a sold position as fresh spendable cash."
        )
    else:
        st.info(
            "Margin accounts have different buying-power and intraday-risk rules. Clarity does not borrow money or place trades for you."
        )

    st.write("**Cloud alert thresholds**")
    st.write(f"Opportunity: **{CONFIG['opportunity_min']}/100**")
    st.write(f"Quick Move: **{CONFIG['quick_move_min']}/100**")

    st.caption(
        "The cloud scanner only alerts when a setup newly crosses these lines, which helps prevent repeated 5-minute alerts."
    )


st.divider()

st.caption(
    "Clarity is a research and paper-planning tool. It does not place trades, cannot guarantee a daily profit, "
    "and fast-moving setups can reverse quickly. Market data comes through yfinance and may be delayed or incomplete."
)
