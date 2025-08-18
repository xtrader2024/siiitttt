import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import io
from datetime import datetime
from decimal import Decimal, getcontext
import statsmodels.api as sm
import base64

getcontext().prec = 50

BIST100_SYMBOLS = [
    "ASELS.IS", "AKBNK.IS", "THYAO.IS", "GARAN.IS", "ISCTR.IS",
    "KCHOL.IS", "PETKM.IS", "SISE.IS", "VAKBN.IS", "YKBNK.IS"
]

RSI_TIME_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
STOCH_FASTK_PERIOD = 14
STOCH_SLOWK_PERIOD = 3

def get_stock_data(symbol, interval, period=200):
    try:
        df = yf.download(symbol, period=f"{period}d", interval=interval)
        if df.empty or len(df) < 51:
            return pd.DataFrame()
        df = df.astype(float)
        return df
    except:
        return pd.DataFrame()

def calculate_indicators(df):
    df = df.copy()
    # SMA ve EMA
    df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_TIME_PERIOD, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_TIME_PERIOD, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()

    # ATR
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Prev_Close']].max(axis=1) - df[['Low', 'Prev_Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(14, min_periods=1).mean()

    # Stokastik
    df['Lowest_Low'] = df['Low'].rolling(STOCH_FASTK_PERIOD, min_periods=1).min()
    df['Highest_High'] = df['High'].rolling(STOCH_FASTK_PERIOD, min_periods=1).max()
    diff = df['Highest_High'] - df['Lowest_Low']
    df['%K'] = np.where(diff==0, 0, 100*(df['Close']-df['Lowest_Low'])/diff)
    df['%D'] = df['%K'].rolling(STOCH_SLOWK_PERIOD, min_periods=1).mean()
    return df

def calculate_support_resistance(df):
    df['Support'] = df['Low'].rolling(50, min_periods=1).min()
    df['Resistance'] = df['High'].rolling(50, min_periods=1).max()
    return df

def generate_signals(df, rsi_lower, rsi_upper, expected_increase_min):
    atr_threshold = df['ATR'].median()
    volume_threshold = df['Volume'].median()
    df['Buy_Signal'] = (
        (df['Close'] > df['SMA_50']) &
        (df['EMA_50'] > df['EMA_200']) &
        (df['MACD_Line'] > df['MACD_Signal']) &
        (df['MACD_Line'] > 0) &
        (df['%K'] > df['%D']) & (df['%K'] > 20) &
        (df['RSI'] > rsi_lower) & (df['RSI'] < rsi_upper) &
        (df['ATR'] > atr_threshold) &
        (df['Volume'] > volume_threshold)
    )
    df['Sell_Signal'] = (
        (df['Close'] < df['SMA_50']) &
        (df['EMA_50'] < df['EMA_200']) &
        (df['MACD_Line'] < df['MACD_Signal']) &
        (df['MACD_Line'] < 0) &
        (df['%K'] < df['%D']) &
        (df['RSI'] > 70)
    )
    df['Signal_Comment'] = np.where(df['Buy_Signal'], "BUY", np.where(df['Sell_Signal'], "SELL", "No signal"))
    return df

def forecast_next_price(df):
    df = df.copy()
    df['day'] = np.arange(len(df))
    X = sm.add_constant(df['day'])
    y = df['Close']
    model = sm.OLS(y, X).fit()
    next_day = sm.add_constant(pd.DataFrame({'day':[len(df)]}))
    forecast = model.predict(next_day)
    return forecast[0]

def calculate_expected_price(df):
    if df.empty:
        return np.nan, np.nan
    price = Decimal(df['Close'].iloc[-1])
    sma_50 = Decimal(df['SMA_50'].iloc[-1])
    if pd.isna(sma_50) or sma_50==0:
        return np.nan, np.nan
    expected_price = price*(1 + (price - sma_50)/sma_50)
    expected_increase = ((expected_price-price)/price)*100
    return float(expected_price), float(expected_increase)

def calculate_trade_levels(df, entry_pct=0.02, take_profit_pct=0.05, stop_loss_pct=0.02):
    if df.empty:
        return np.nan, np.nan, np.nan
    entry_price = Decimal(df['Close'].iloc[-1])
    take_profit_price = entry_price*(1+Decimal(take_profit_pct))
    stop_loss_price = entry_price*(1-Decimal(stop_loss_pct))
    return float(entry_price), float(take_profit_price), float(stop_loss_price)

def plot_to_png(df, symbol, entry=None, tp=None, sl=None):
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(df.index, df['Close'], label='Close', color='blue')
    ax.plot(df.index, df['SMA_50'], label='SMA50', color='green')
    ax.plot(df.index, df['EMA_50'], label='EMA50', color='red')
    ax.plot(df.index, df['ATR'], label='ATR', color='orange')
    if entry: ax.axhline(entry, color='lime', linestyle='--', label='Entry')
    if tp: ax.axhline(tp, color='gold', linestyle='--', label='TP')
    if sl: ax.axhline(sl, color='red', linestyle='--', label='SL')
    ax.set_title(symbol)
    ax.legend(); ax.grid(True)
    img = io.BytesIO(); plt.savefig(img, format='png'); img.seek(0); plt.close(fig)
    return img

def process_symbol(symbol, interval, rsi_lower, rsi_upper, min_expected_increase):
    df = get_stock_data(symbol, interval)
    if df.empty: return None
    df = calculate_indicators(df)
    df = calculate_support_resistance(df)
    df = generate_signals(df, rsi_lower, rsi_upper, min_expected_increase)
    forecast = forecast_next_price(df)
    expected_price, expected_increase = calculate_expected_price(df)
    entry, tp, sl = calculate_trade_levels(df)
    if df['Buy_Signal'].iloc[-3:].any() and expected_increase>=min_expected_increase:
        return {'symbol':symbol, 'price':df['Close'].iloc[-1], 'expected_price':expected_price,
                'expected_increase':expected_increase, 'entry':entry,'tp':tp,'sl':sl,
                'signal':df['Signal_Comment'].iloc[-1], 'plot':plot_to_png(df,symbol,entry,tp,sl)}
    return None

def main():
    st.title("BIST100 Analiz")
    interval = st.selectbox("Zaman Aralığı", ['1d','1wk'])
    rsi_lower = st.slider("RSI Alt Sınır", 30, 70, 45)
    rsi_upper = st.slider("RSI Üst Sınır", 50, 90, 75)
    min_increase = st.slider("Minimum Beklenen Artış %",0,20,5)
    if st.button("Analiz Başlat"):
        results=[]
        with st.spinner("Analiz ediliyor..."):
            for symbol in BIST100_SYMBOLS:
                res = process_symbol(symbol, interval, rsi_lower, rsi_upper, min_increase)
                if res: results.append(res)
        if results:
            for r in results:
                st.subheader(f"{r['symbol']} - {r['signal']}")
                st.write(f"Fiyat: {r['price']}, Beklenen: {r['expected_price']} ({r['expected_increase']:.2f}%)")
                st.image(r['plot'])
        else:
            st.write("Güçlü alım sinyali bulunamadı.")

if __name__=="__main__":
    main()
