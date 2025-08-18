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
    except Exception as e:
        st.error(f"Data fetching error ({symbol}): {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    # SMA ve EMA
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=RSI_TIME_PERIOD, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=RSI_TIME_PERIOD, min_periods=1).mean()
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
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()

    # Stokastik
    df['Lowest_Low'] = df['Low'].rolling(window=STOCH_FASTK_PERIOD, min_periods=1).min()
    df['Highest_High'] = df['High'].rolling(window=STOCH_FASTK_PERIOD, min_periods=1).max()
    diff = df['Highest_High'] - df['Lowest_Low']
    df['%K'] = np.where(diff == 0, 0, 100 * (df['Close'] - df['Lowest_Low']) / diff)
    df['%D'] = df['%K'].rolling(window=STOCH_SLOWK_PERIOD, min_periods=1).mean()

    return df

def calculate_support_resistance(df):
    df['Support'] = df['Low'].rolling(window=50, min_periods=1).min()
    df['Resistance'] = df['High'].rolling(window=50, min_periods=1).max()
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

    df['Signal_Comment'] = np.where(
        df['Buy_Signal'],
        "Strong BUY signal: Uptrend, MACD & Stochastic momentum aligned, volatility & volume sufficient.",
        np.where(
            df['Sell_Signal'],
            "SELL signal: Downtrend, MACD & momentum negative, RSI overbought.",
            "No signal"
        )
    )
    return df

def forecast_next_price(df):
    df = df.copy()
    df['day'] = np.arange(len(df))
    X = df[['day']]
    y = df['Close']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    next_day_index = pd.DataFrame({'day':[len(df)+1]})
    next_day_index = sm.add_constant(next_day_index, has_constant='add')
    forecast = model.predict(next_day_index)
    return forecast[0]

def calculate_expected_price(df):
    if df.empty:
        return np.nan, np.nan
    price = Decimal(df['Close'].iloc[-1])
    sma_50 = Decimal(df['SMA_50'].iloc[-1])
    if pd.isna(sma_50) or sma_50 == 0:
        return np.nan, np.nan
    expected_price = price * (1 + (price - sma_50) / sma_50)
    expected_increase_percentage = ((expected_price - price) / price) * 100
    return float(expected_price), float(expected_increase_percentage)

def calculate_trade_levels(df, entry_pct=0.02, take_profit_pct=0.05, stop_loss_pct=0.02):
    if df.empty:
        return np.nan, np.nan, np.nan
    entry_price = Decimal(df['Close'].iloc[-1])
    take_profit_price = entry_price * (1 + Decimal(take_profit_pct))
    stop_loss_price = entry_price * (1 - Decimal(stop_loss_pct))
    return float(entry_price), float(take_profit_price), float(stop_loss_price)
