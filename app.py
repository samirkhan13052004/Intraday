import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- 1. पेज सेटअप ---
st.set_page_config(
    page_title="Ultimate Pro Trading Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ Ultimate Institutional Scanner: VWAP + EMA + Traps + OI")


# --- 2. डेटा जनरेशन / लाइव डेटा फीड ---
def get_intraday_data():
    np.random.seed(42)
    dates = pd.date_range(start="2026-09-01 09:15", periods=75, freq="5min")
    price = 24100 + np.cumsum(np.random.randn(75) * 12)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price + np.random.randn(75) * 4,
            "high": price + np.random.rand(75) * 18,
            "low": price - np.random.rand(75) * 18,
            "close": price + np.random.randn(75) * 4,
            "volume": np.random.randint(15000, 95000, size=75),
        }
    )
    return df


df = get_intraday_data()

# --- 3. टेक्निकल इंडिकेटर्स कैलकुलेशन ---
# A. 20 EMA
df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

# B. Intraday VWAP
typical_price = (df["high"] + df["low"] + df["close"]) / 3
df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

# C. Swing High & Swing Low (Key Levels)
df["swing_high"] = df["high"].rolling(15).max().shift(1)
df["swing_low"] = df["low"].rolling(15).min().shift(1)

# D. Volume & Wick Calculations
df["avg_volume"] = df["volume"].rolling(20).mean()
df["rvol"] = df["volume"] / df["avg_volume"]
df["candle_range"] = (df["high"] - df["low"]).replace(0, 0.001)
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

# --- 4. साइडबार: OI और PCR इनपुट्स ---
st.sidebar.header("📊 F&O डेटा और फिल्टर्स")
pcr_value = st.sidebar.slider(
    "Live PCR (Put-Call Ratio)", min_value=0.4, max_value=1.8, value=0.72, step=0.01
)
min_rvol = st.sidebar.slider("Min RVOL Filter", 1.0, 2.5, 1.3, 0.1)

total_call_oi = st.sidebar.number_input(
    "Total Call OI (Lakhs)", value=145.2, step=1.0
)
total_put_oi = st.sidebar.number_input(
    "Total Put OI (Lakhs)", value=98.6, step=1.0
)

# --- 5. सिग्नल डिटेक्शन इंजन (Multi-Strategy) ---
df["signal"] = None
df["setup_type"] = None
df["sl"] = np.nan
df["target"] = np.nan

for i in range(20, len(df)):
    row = df.loc[i]
    prev_row = df.loc[i - 1]

    # ------------------ SETUP 1: INSTITUTIONAL BULL TRAP (SELL) ------------------
    # स्विंग हाई को छुआ, लेकिन VWAP/EMA के नीचे रिजेक्ट हुआ + PCR बेयरिश + हाई विक
    is_bull_trap = (
        row["high"] > row["swing_high"]
        and row["close"] < row["swing_high"]
        and (row["upper_wick"] / row["candle_range"] >= 0.30)
        and row["rvol"] >= min_rvol
        and pcr_value < 0.85
    )

    if is_bull_trap:
        entry = row["close"]
        sl = row["high"] + 2.0
        risk = sl - entry
        df.loc[i, "signal"] = "STRONG SELL"
        df.loc[i, "setup_type"] = "Liquidity Trap + Call OI Heavy"
        df.loc[i, "sl"] = round(sl, 2)
        df.loc[i, "target"] = round(entry - (risk * 3), 2)  # 1:3 RRR
        continue

    # ------------------ SETUP 2: INSTITUTIONAL BEAR TRAP (BUY) ------------------
    # स्विंग लो को तोड़ा, लेकिन तेजी से ऊपर क्लोज हुआ + PCR बुलिश + लोअर विक
    is_bear_trap = (
        row["low"] < row["swing_low"]
        and row["close"] > row["swing_low"]
        and (row["lower_wick"] / row["candle_range"] >= 0.30)
        and row["rvol"] >= min_rvol
        and pcr_value > 1.15
    )

    if is_bear_trap:
        entry = row["close"]
        sl = row["low"] - 2.0
        risk = entry - sl
        df.loc[i, "signal"] = "STRONG BUY"
        df.loc[i, "setup_type"] = "Liquidity Trap + Put OI Support"
        df.loc[i, "sl"] = round(sl, 2)
        df.loc[i, "target"] = round(entry + (risk * 3), 2)  # 1:3 RRR
        continue

    # ------------------ SETUP 3: VWAP + 20 EMA PULLBACK (BUY) ------------------
    is_vwap_pullback_buy = (
        row["close"] > row["vwap"]
        and row["close"] > row["ema_20"]
        and prev_row["low"] <= prev_row["ema_20"]
        and row["close"] > prev_row["high"]
        and pcr_value >= 0.95
    )

    if is_vwap_pullback_buy:
        entry = row["close"]
        sl = min(row["vwap"], row["ema_20"]) - 2.0
        risk = entry - sl
        df.loc[i, "signal"] = "BUY (PULLBACK)"
        df.loc[i, "setup_type"] = "VWAP + 20 EMA Trend Pullback"
        df.loc[i, "sl"] = round(sl, 2)
        df.loc[i, "target"] = round(entry + (risk * 2.5), 2)

# --- 6. टॉप हेडलाइन मेट्रिक्स ---
c1, c2, c3, c4 = st.columns(4)
current_ltp = df["close"].iloc[-1]
c1.metric("Current Spot LTP", f"₹{current_ltp:.2f}")
c2.metric("20 EMA", f"₹{df['ema_20'].iloc[-1]:.2f}")
c3.metric("VWAP Line", f"₹{df['vwap'].iloc[-1]:.2f}")
sentiment = (
    "Strong Bearish"
    if pcr_value < 0.75
    else "Strong Bullish"
    if pcr_value > 1.2
    else "Neutral"
)
c4.metric("Market Sentiment (PCR)", f"{pcr_value}", sentiment)

# --- 7. चार्ट विज़ुअलाइज़ेशन ---
st.subheader("📊 लाइव चार्ट: Candlestick, VWAP, 20 EMA और लेवल्स")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.75, 0.25],
)

# 1. Candlestick
fig.add_trace(
    go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price Action",
    ),
    row=1,
    col=1,
)

# 2. VWAP & 20 EMA
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["vwap"],
        line=dict(color="#FFD700", width=2),
        name="VWAP Line",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["ema_20"],
        line=dict(color="#00FFFF", width=1.5),
        name="20 EMA",
    ),
    row=1,
    col=1,
)

# 3. Swing High / Low (Liquidity Zones)
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["swing_high"],
        line=dict(color="#FF4136", dash="dot"),
        name="Resistance (Swing High)",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["swing_low"],
        line=dict(color="#2ECC40", dash="dot"),
        name="Support (Swing Low)",
    ),
    row=1,
    col=1,
)

# 4. Volume Subplot
colors = [
    "#2ECC40" if c >= o else "#FF4136"
    for c, o in zip(df["close"], df["open"])
]
fig.add_trace(
    go.Bar(
        x=df["timestamp"],
        y=df["volume"],
        marker_color=colors,
        name="Volume",
        opacity=0.7,
    ),
    row=2,
    col=1,
)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=600,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# --- 8. ट्रेड्स एवं अलर्ट्स तालिका ---
st.subheader("🎯 जनरेटेड सिग्नल्स और रिस्क मैनेजमेंट (1:3 Targets)")

signals = df[df["signal"].notnull()][
    ["timestamp", "signal", "setup_type", "close", "sl", "target", "rvol"]
].copy()
signals.columns = [
    "Time",
    "Signal Type",
    "Setup Strategy",
    "Entry Price",
    "Stop-Loss",
    "Target",
    "RVOL",
]

if not signals.empty:
    st.dataframe(
        signals.sort_values(by="Time", ascending=False),
        use_container_width=True,
    )
else:
    st.info("मार्केट में अभी कोई 5-स्टार कन्फ्लुएंस सिग्नल नहीं मिला है।")
