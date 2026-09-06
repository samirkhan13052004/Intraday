from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyotp
from SmartApi import SmartConnect
import streamlit as st

# --- पेज कॉन्फ़िगरेशन ---
st.set_page_config(
    page_title="Institutional Trading Terminal",
    layout="wide",
    page_icon="⚡",
)
st.title("⚡ Pro Terminal: Angel One Live Data + VWAP + EMA + Traps + OI")

# --- टोकन मैपिंग ---
INSTRUMENT_MAP = {
    "NIFTY 50": {"token": "99926000", "exchange": "NSE"},
    "BANK NIFTY": {"token": "99926009", "exchange": "NSE"},
    "RELIANCE": {"token": "2885", "exchange": "NSE"},
    "HDFC BANK": {"token": "1333", "exchange": "NSE"},
    "TATA MOTORS": {"token": "3456", "exchange": "NSE"},
    "ICICI BANK": {"token": "4963", "exchange": "NSE"},
    "INFOSYS": {"token": "1594", "exchange": "NSE"},
    "STATE BANK OF INDIA": {"token": "3045", "exchange": "NSE"},
}

# --- साइडबार ---
st.sidebar.header("🔐 Angel One API लॉगिन")
api_key = st.sidebar.text_input("SmartAPI Key", type="password")
client_code = st.sidebar.text_input("Client ID (उदा: A123456)")
pin = st.sidebar.text_input("MPIN (4 Digits)", type="password")
totp_secret = st.sidebar.text_input("TOTP Secret Key", type="password")

st.sidebar.markdown("---")
st.sidebar.header("🎯 इंस्ट्रूमेंट & F&O सेटिंग्स")
selected_name = st.sidebar.selectbox(
    "शेयर / इंडेक्स चुनें", list(INSTRUMENT_MAP.keys())
)
selected_inst = INSTRUMENT_MAP[selected_name]

live_pcr = st.sidebar.slider(
    "Live PCR Filter (Put-Call Ratio)", 0.4, 1.8, 0.85, 0.01
)
min_rvol = st.sidebar.slider("Min RVOL Filter", 1.0, 2.5, 1.3, 0.1)


# --- 1. Angel One डेटा फेचिंग ---
@st.cache_data(ttl=60)
def fetch_data(api_key, client_code, pin, totp_secret, token, exchange):
    try:
        totp = pyotp.TOTP(totp_secret).now()
        smartApi = SmartConnect(api_key=api_key)
        session = smartApi.generateSession(client_code, pin, totp)
        if not session["status"]:
            return None

        to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        from_date = (datetime.now() - timedelta(days=4)).strftime(
            "%Y-%m-%d 09:15"
        )

        param = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": from_date,
            "todate": to_date,
        }

        res = smartApi.getCandleData(param)
        if res["status"] and res["data"]:
            df = pd.DataFrame(
                res["data"],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        return None
    except Exception:
        return None


# डेटा लोड
df = None
if api_key and client_code and pin and totp_secret:
    with st.spinner("Angel One से लाइव 5-मिनट डेटा लोड हो रहा है..."):
        df = fetch_data(
            api_key,
            client_code,
            pin,
            totp_secret,
            selected_inst["token"],
            selected_inst["exchange"],
        )

if df is None:
    st.info("⚠️ लाइव API कनेक्ट नहीं है। नीचे डेमो डेटा प्रदर्शित हो रहा है।")
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

# --- 2. इंडिकेटर्स और इंट्रामार्केट VWAP (Daily Reset) ---
df["date"] = df["timestamp"].dt.date
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
df["pv"] = df["typical_price"] * df["volume"]

# हर दिन सुबह 9:15 पर VWAP रीसेट
df["cum_pv"] = df.groupby("date")["pv"].cumsum()
df["cum_vol"] = df.groupby("date")["volume"].cumsum()
df["vwap"] = df["cum_pv"] / df["cum_vol"]

# 20 EMA और की-लेवल्स
df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
df["swing_high"] = df["high"].rolling(15).max().shift(1)
df["swing_low"] = df["low"].rolling(15).min().shift(1)
df["avg_volume"] = df["volume"].rolling(20).mean()
df["rvol"] = df["volume"] / df["avg_volume"]
df["candle_range"] = (df["high"] - df["low"]).replace(0, 0.001)
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

# --- 3. सिग्नल डिटेक्शन (Traps + VWAP Pullback + PCR Filter) ---
df["signal"] = None
df["strategy"] = None
df["sl"] = np.nan
df["target"] = np.nan

for i in range(20, len(df)):
    row = df.loc[i]
    prev = df.loc[i - 1]

    # Setup 1: Institutional Bull Trap (Sell)
    if (
        row["high"] > row["swing_high"]
        and row["close"] < row["swing_high"]
        and (row["upper_wick"] / row["candle_range"] >= 0.30)
        and row["rvol"] >= min_rvol
        and live_pcr < 0.85
    ):
        sl = row["high"] + (row["high"] * 0.0005)
        risk = sl - row["close"]
        df.loc[i, "signal"] = "STRONG SELL"
        df.loc[i, "strategy"] = "Liquidity Sweep (Bull Trap)"
        df.loc[i, "sl"] = round(sl, 2)
        df.loc[i, "target"] = round(row["close"] - (risk * 3), 2)
        continue

    # Setup 2: Institutional Bear Trap (Buy)
    if (
        row["low"] < row["swing_low"]
        and row["close"] > row["swing_low"]
        and (row["lower_wick"] / row["candle_range"] >= 0.30)
        and row["rvol"] >= min_rvol
        and live_pcr > 1.15
    ):
        sl = row["low"] - (row["low"] * 0.0005)
        risk = row["close"] - sl
        df.loc[i, "signal"] = "STRONG BUY"
        df.loc[i, "strategy"] = "Liquidity Sweep (Bear Trap)"
        df.loc[i, "sl"] = round(sl, 2)
        df.loc[i, "target"] = round(row["close"] + (risk * 3), 2)
        continue

    # Setup 3: VWAP + 20 EMA Pullback (Buy)
    if (
        row["close"] > row["vwap"]
        and row["close"] > row["ema_20"]
        and prev["low"] <= prev["ema_20"]
        and row["close"] > prev["high"]
        and live_pcr >= 0.90
    ):
        sl = min(row["vwap"], row["ema_20"]) - (row["close"] * 0.0005)
        risk = row["close"] - sl
        df.loc[i, "signal"] = "BUY (PULLBACK)"
        df.loc[i, "strategy"] = "VWAP + 20 EMA Support"
        df.loc[i, "sl"] = round(sl, 2)
        df.loc[i, "target"] = round(row["close"] + (risk * 2.5), 2)

# --- 4. टॉप हेडलाइन मेट्रिक्स ---
m1, m2, m3, m4 = st.columns(4)
current_ltp = df["close"].iloc[-1]
m1.metric(f"LTP ({selected_name})", f"₹{current_ltp:.2f}")
m2.metric("VWAP (Intraday)", f"₹{df['vwap'].iloc[-1]:.2f}")
m3.metric("20 EMA", f"₹{df['ema_20'].iloc[-1]:.2f}")
m4.metric("Live PCR", f"{live_pcr}")

# --- 5. प्लॉटली चार्ट ---
st.subheader(f"📈 {selected_name} - 5-Min Live Chart")
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.75, 0.25],
)

fig.add_trace(
    go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["vwap"],
        line=dict(color="#FFD700", width=2),
        name="VWAP",
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
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["swing_high"],
        line=dict(color="#FF4136", dash="dot"),
        name="Resistance",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["swing_low"],
        line=dict(color="#2ECC40", dash="dot"),
        name="Support",
    ),
    row=1,
    col=1,
)

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
    height=550,
    template="plotly_dark",
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# --- 6. सिग्नल्स टेबल ---
st.subheader("🎯 एक्टिव सिग्नल्स, एंट्री, SL और 1:3 टारगेट्स")
signals = df[df["signal"].notnull()][
    ["timestamp", "signal", "strategy", "close", "sl", "target", "rvol"]
].copy()
signals.columns = [
    "Time",
    "Signal",
    "Setup",
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
    st.info("फिलहाल इस चार्ट में कोई 5-स्टार वैलिड सेटअप नहीं बना है।")
