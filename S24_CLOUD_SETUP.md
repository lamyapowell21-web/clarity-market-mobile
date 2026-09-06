
# Clarity Cloud + Galaxy S24 setup

This version does **not** need your personal computer to stay on.

After setup:

- Streamlit Community Cloud hosts the Clarity phone dashboard.
- GitHub Actions checks the market on a schedule.
- ntfy delivers alerts to your Galaxy S24.
- Robinhood is still where you personally place any trade.
- Your old local Clarity folder can stay exactly as it is.

## Before starting

You need:

1. Your existing ntfy app on the S24. Keep the same topic that already passed the notification test.
2. A free GitHub account.
3. A free Streamlit Community Cloud account.

There are no stock-data API keys in this setup.

---

# PART A — Put this project on GitHub

## 1. Create a private repository

Go to GitHub and create a new repository named:

`clarity-market-mobile`

Choose **Private**.

Do not add a README or .gitignore during creation because this folder already contains the files you need.

## 2. Upload this entire folder

Upload all of these items to the repository root:

- `app.py`
- `clarity_engine.py`
- `cloud_scanner.py`
- `cloud_config.json`
- `requirements.txt`
- `.streamlit`
- `.github`

The two dot-folders are important. Make sure `.github/workflows/clarity_scan.yml` exists after the upload.

Commit the upload to the `main` branch.

---

# PART B — Add the phone-notification secret

In your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

Create:

### Secret 1

Name:

`NTFY_TOPIC`

Value:

Use the **same ntfy topic name that already worked on your S24**.

Do not put the topic directly into the public code.

### Optional Secret 2

Name:

`MAX_TRADE_AMOUNT`

Value:

The maximum dollar amount you want a notification to use for its example.

If you do not want Clarity making dollar-size examples in notifications yet, skip this secret.

### Optional Secret 3

Name:

`MAX_PLANNED_LOSS`

Value:

The maximum dollar loss you want the notification's planning example to size around.

If you skip it, the alert will still include the price entry/target/exit levels.

---

# PART C — Test the cloud scanner

Open your GitHub repository.

Go to:

**Actions → Clarity market scan → Run workflow**

Choose:

`test`

and run it.

Your Galaxy S24 should receive:

**Clarity cloud test 💗**

This test comes from GitHub's cloud computer, not your Windows computer.

If the test arrives, you can turn your personal computer completely off and the phone-alert path still works.

---

# PART D — Put the Clarity dashboard online

Go to Streamlit Community Cloud:

`https://share.streamlit.io`

Sign in and connect your GitHub account.

Create an app from your existing repository.

Use:

- Repository: `clarity-market-mobile`
- Branch: `main`
- Main file path: `app.py`

Deploy it.

When it finishes, Streamlit gives you a URL ending in:

`.streamlit.app`

Open the app and make sure Clarity loads.

Because the GitHub repository is private, Streamlit can keep the app private. Streamlit currently allows one private Community Cloud app at a time.

---

# PART E — Make alerts open Clarity

Copy your Streamlit app URL.

Back in GitHub:

**Repository → Settings → Secrets and variables → Actions → New repository secret**

Create:

Name:

`CLARITY_APP_URL`

Value:

Paste the complete Streamlit app URL.

The next notification can now include:

- **Open Clarity**
- **Open Robinhood**

buttons.

Run the GitHub Action in `test` mode one more time to confirm the alert works after adding the app URL.

---

# PART F — Put Clarity on the Galaxy S24 home screen

On the S24:

1. Open your new `.streamlit.app` link in Chrome.
2. Sign in if your private app asks you to.
3. Tap Chrome's **⋮** menu.
4. Choose **Add to Home screen**.
5. Name it `Clarity`.
6. Add it.

Now Clarity has a home-screen icon. Your personal computer is not part of the workday setup.

---

# What happens on a normal workday

During the NYSE session, GitHub schedules the scanner about every five minutes.

The scanner:

1. Checks whether the NYSE is actually open.
2. Does nothing on weekends and market holidays.
3. Reads daily and five-minute stock data.
4. Calculates the Opportunity and Quick Move scores.
5. Alerts only when a setup **newly crosses** both thresholds.
6. Sends at most the strongest new setup from that scan.

This helps prevent the same stock from buzzing your phone every five minutes.

The default alert lines are:

- Opportunity: 80/100
- Quick Move: 84/100

You can change them later in `cloud_config.json`.

## Important timing limitation

GitHub supports five-minute scheduled workflows, but GitHub says scheduled runs can sometimes be delayed during periods of high load. Treat the scanner as an approximately-five-minute research alert, not an exchange-grade real-time trading feed.

## Important trading limitation

Clarity does not know the future and does not guarantee profit. A Quick Move score means the ingredients for faster movement are present, not that the next move must be upward. Use paper practice first and verify the alerts before relying on real money.
