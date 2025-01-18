import time
import pandas as pd
from binance.client import Client
import plotly.graph_objects as go
import streamlit as st

# Binance API keys
API_KEY = '6bEGb9VCH29RC1ezF3NG4YjyBiM5wtKSDiAVc7GBFzzIb7gZomscJ4SRIsVeg0z'
API_SECRET = 'Z3ecOIMXGAaoZLLkv9rGa3C6vZmwldYqj8xlK6zDSUrjNMPJFT6P6x2xwltWiLqx'
client = Client(API_KEY, API_SECRET)

# Default Parameters
default_symbol = "BTCUSDT"
factor = 1  # SuperTrend Factor
period = 7  # ATR Period
last_signal = None  # To track the last signal (`GoLong` or `GoShort`)
last_signals = {"GoLong": None, "GoShort": None, "GoLong Price": None, "GoShort Price": None}

# Fetch historical data
def fetch_data(symbol, interval, limit=500):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)
    return df

# ATR Calculation
def calculate_atr(df, period):
    df['tr'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1
    )
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    return df

# Calculate SuperTrend
def calculate_supertrend(df, factor, period):
    df = calculate_atr(df, period)
    df['hl2'] = (df['high'] + df['low']) / 2
    df['Up'] = df['hl2'] - (factor * df['atr'])
    df['Dn'] = df['hl2'] + (factor * df['atr'])

    df['TrendUp'] = 0.0
    df['TrendDown'] = 0.0
    df['Trend'] = 0

    for i in range(1, len(df)):
        if df['close'].iloc[i - 1] > df['TrendUp'].iloc[i - 1]:
            df.loc[df.index[i], 'TrendUp'] = max(df['Up'].iloc[i], df['TrendUp'].iloc[i - 1])
        else:
            df.loc[df.index[i], 'TrendUp'] = df['Up'].iloc[i]

        if df['close'].iloc[i - 1] < df['TrendDown'].iloc[i - 1]:
            df.loc[df.index[i], 'TrendDown'] = min(df['Dn'].iloc[i], df['TrendDown'].iloc[i - 1])
        else:
            df.loc[df.index[i], 'TrendDown'] = df['Dn'].iloc[i]

        if df['close'].iloc[i] > df['TrendDown'].iloc[i - 1]:
            df.loc[df.index[i], 'Trend'] = 1
        elif df['close'].iloc[i] < df['TrendUp'].iloc[i - 1]:
            df.loc[df.index[i], 'Trend'] = -1
        else:
            df.loc[df.index[i], 'Trend'] = df['Trend'].iloc[i - 1]

    df['TSL'] = df['TrendUp']
    df.loc[df['Trend'] == -1, 'TSL'] = df['TrendDown']
    return df

# Generate Signals and Update Last Signal Timestamps
def generate_signals(df_15m, last_signal, last_signals):
    log_data = []
    current_close = df_15m['close'].iloc[-1]
    current_tsl = df_15m['MTSL'].iloc[-1]

    if current_close > current_tsl and last_signal != "GoLong":
        log_data.append({"Timestamp": df_15m.index[-1], "Signal": "GoLong", "Price": current_close})
        last_signals["GoLong"] = df_15m.index[-1]
        last_signals["GoLong Price"] = current_close
        last_signal = "GoLong"

    elif current_close < current_tsl and last_signal != "GoShort":
        log_data.append({"Timestamp": df_15m.index[-1], "Signal": "GoShort", "Price": current_close})
        last_signals["GoShort"] = df_15m.index[-1]
        last_signals["GoShort Price"] = current_close
        last_signal = "GoShort"

    return pd.DataFrame(log_data), last_signal, last_signals

# Trading logic and data preparation
def trade_logic(selected_symbol, selected_timeframe, selected_main_timeframe, last_signal, last_signals):
    df_15m = fetch_data(selected_symbol, selected_timeframe)
    df_1h = fetch_data(selected_symbol, selected_main_timeframe)

    df_15m = calculate_supertrend(df_15m, factor, period)
    df_1h = calculate_supertrend(df_1h, factor, period)

    df_1h_resampled = df_1h.resample('15min').ffill().reindex(df_15m.index, method='ffill')
    df_15m['MTSL'] = df_1h_resampled['TSL']

    log_data, last_signal, last_signals = generate_signals(df_15m, last_signal, last_signals)

    return df_15m, df_1h, log_data, last_signal, last_signals

# Streamlit Dashboard
st.set_page_config(page_title="Crypto SuperTrend Dashboard", layout="wide")
st.title("Crypto Trading Dashboard")

# Sidebar
st.sidebar.header("Settings")
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
selected_symbol = st.sidebar.selectbox("Select Symbol", symbols, index=symbols.index(default_symbol))
timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]
selected_timeframe = st.sidebar.selectbox("Select Chart Timeframe", timeframes, index=3)
selected_main_timeframe = st.sidebar.selectbox("Select Main Timeframe", timeframes, index=5)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", min_value=5, max_value=60, value=15)

# Dynamic content placeholders
chart_placeholder = st.empty()
signals_log_placeholder = st.empty()
summary_placeholder = st.empty()
candle_data_placeholder = st.empty()
ohlc_data_placeholder = st.empty()
last_signals_placeholder = st.empty()

while True:
    df_15m, df_1h, log_data, last_signal, last_signals = trade_logic(
        selected_symbol, selected_timeframe, selected_main_timeframe, last_signal, last_signals
    )

    # Update chart
    with chart_placeholder:
        st.subheader(f"Live Candlestick Chart for {selected_symbol}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_15m.index, open=df_15m['open'], high=df_15m['high'], low=df_15m['low'], close=df_15m['close'],
            name='OHLC'
        ))
        fig.add_trace(go.Scatter(x=df_15m.index, y=df_15m['MTSL'], mode='lines', name=f'{selected_main_timeframe} TSL'))

        for _, row in log_data.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Timestamp']], y=[row['Price']], mode='markers',
                name=row['Signal'], marker=dict(color='green' if row['Signal'] == 'GoLong' else 'red', size=10)
            ))

        st.plotly_chart(fig, use_container_width=True, key=f"chart_{time.time()}")

    # Update Signals Log Table
    with signals_log_placeholder:
        st.subheader("Signals Log Table")
        if not log_data.empty:
            log_data = log_data.sort_values(by="Timestamp", ascending=False)
        st.dataframe(log_data)

    # Update SuperTrend Summary Table
    with summary_placeholder:
        st.subheader("SuperTrend Summary Table")
        supertrend_summary = pd.DataFrame({
            "Timeframe": ["15m", "1h"],
            "Latest TSL": [df_15m['TSL'].iloc[-1], df_1h['TSL'].iloc[-1]],
            "Trend Status": [
                "Bullish" if df_15m['Trend'].iloc[-1] == 1 else "Bearish",
                "Bullish" if df_1h['Trend'].iloc[-1] == 1 else "Bearish"
            ],
            "Latest Signal": [
                "GoLong" if df_15m['close'].iloc[-1] > df_15m['MTSL'].iloc[-1] else "GoShort",
                "GoLong" if df_15m['close'].iloc[-1] > df_15m['MTSL'].iloc[-1] else "GoShort"
            ]
        })
        st.table(supertrend_summary)

    # Update Current Candle Data and TSL Table
    with candle_data_placeholder:
        st.subheader("Current Candle Data and TSL Table")
        current_data = {
            "Open": df_15m['open'].iloc[-1],
            "High": df_15m['high'].iloc[-1],
            "Low": df_15m['low'].iloc[-1],
            "Close": df_15m['close'].iloc[-1],
            "TSL (1h)": df_15m['MTSL'].iloc[-1]
        }
        st.table(pd.DataFrame([current_data]))

    # Update Recent OHLC Data Table
    with ohlc_data_placeholder:
        st.subheader("Recent OHLC Data Table")
        st.dataframe(df_15m[['open', 'high', 'low', 'close']].tail(10))

    # Update Last GoLong and GoShort Signals Table
    with last_signals_placeholder:
        st.subheader("Last GoLong and GoShort Signals Table")
        last_signals_df = pd.DataFrame([{
            "GoLong Timestamp": last_signals["GoLong"],
            "GoLong Price": last_signals["GoLong Price"],
            "GoShort Timestamp": last_signals["GoShort"],
            "GoShort Price": last_signals["GoShort Price"]
        }])
        st.table(last_signals_df)

    # Wait before refreshing
    time.sleep(refresh_rate)