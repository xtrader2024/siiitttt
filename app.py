import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
import statsmodels.api as sm
import base64

getcontext().prec = 50

# Örnek BIST100 sembolleri
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
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Bollinger Bandları kaldırdık

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_TIME_PERIOD, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_TIME_PERIOD, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD_Line'] = df['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean() - df['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()

    # ATR
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Prev_Close']].max(axis=1) - df[['Low', 'Prev_Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()

    # Stokastik
    df['Lowest_Low'] = df['Low'].rolling(window=STOCH_FASTK_PERIOD, min_periods=1).min()
    df['Highest_High'] = df['High'].rolling(window=STOCH_FASTK_PERIOD, min_periods=1).max()
    df['%K'] = 100 * (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']).replace(0, np.nan)
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
        "Strong BUY signal",
        np.where(
            df['Sell_Signal'],
            "SELL signal",
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
    next_day_index = pd.DataFrame({'day': [len(df)]})
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

def plot_to_png(df, symbol, entry=None, tp=None, sl=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax.plot(df.index, df['SMA_50'], label='SMA 50', color='green')
    ax.plot(df.index, df['EMA_50'], label='EMA 50', color='red')
    if 'Support' in df.columns:
        ax.plot(df.index, df['Support'], label='Support', color='cyan', linestyle='--')
    if 'Resistance' in df.columns:
        ax.plot(df.index, df['Resistance'], label='Resistance', color='magenta', linestyle='--')
    if entry:
        ax.axhline(entry, color='lime', linestyle='-.', label='Entry Price')
    if tp:
        ax.axhline(tp, color='gold', linestyle='-.', label='Take Profit')
    if sl:
        ax.axhline(sl, color='red', linestyle='-.', label='Stop Loss')
    ax.set_title(f'{symbol} Analysis')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return img

def process_symbol(symbol, interval, rsi_lower, rsi_upper, min_expected_increase):
    df = get_stock_data(symbol, interval)
    if df.empty:
        return None
    df = calculate_indicators(df)
    df = calculate_support_resistance(df)
    df = generate_signals(df, rsi_lower, rsi_upper, min_expected_increase)
    forecast = forecast_next_price(df)
    expected_price, expected_increase_percentage = calculate_expected_price(df)
    entry_price, take_profit_price, stop_loss_price = calculate_trade_levels(df)
    debug_info = {
        'Buy_Signal_Last_5': df['Buy_Signal'].iloc[-5:].tolist(),
        'Expected_Increase': expected_increase_percentage,
        'RSI_Last': df['RSI'].iloc[-1],
        'MACD_Line_Last': df['MACD_Line'].iloc[-1]
    }
    if df['Buy_Signal'].iloc[-3:].any() and expected_increase_percentage >= min_expected_increase:
        return {
            'coin_name': symbol,
            'price': df['Close'].iloc[-1],
            'expected_price': expected_price,
            'expected_increase_percentage': expected_increase_percentage,
            'sma_50': df['SMA_50'].iloc[-1],
            'rsi_14': df['RSI'].iloc[-1],
            'macd_line': df['MACD_Line'].iloc[-1],
            'macd_signal': df['MACD_Signal'].iloc[-1],
            'atr': df['ATR'].iloc[-1],
            'stoch_k': df['%K'].iloc[-1],
            'stoch_d': df['%D'].iloc[-1],
            'forecast_next_day_price': forecast,
            'entry_price': entry_price,
            'take_profit_price': take_profit_price,
            'stop_loss_price': stop_loss_price,
            'signal_comment': df['Signal_Comment'].iloc[-1],
            'plot': plot_to_png(df, symbol, entry=entry_price, tp=take_profit_price, sl=stop_loss_price),
            'debug': debug_info
        }
    else:
        return None

def main():
    st.title("BIST100 Hisse Analizi")
    interval = st.selectbox('Zaman Aralığı', ['1d', '1wk'], index=0)
    rsi_lower = st.slider('RSI Alt Sınır', 30, 70, 45)
    rsi_upper = st.slider('RSI Üst Sınır', 50, 90, 75)
    min_expected_increase = st.slider('Minimum Beklenen Artış %', 0, 20, 5)
    start_button = st.button('Analiz Başlat')
    if not start_button:
        return
    results = []
    with st.spinner('Analiz ediliyor...'):
        for symbol in BIST100_SYMBOLS:
            res = process_symbol(symbol, interval, rsi_lower, rsi_upper, min_expected_increase)
            if res:
                results.append(res)
    if results:
        st.write(f"Analiz edilen toplam hisse sayısı: {len(results)}")
        for res in results:
            with st.expander(f"{res['coin_name']} - {res['signal_comment']}"):
                st.write(f"Mevcut Fiyat: {res['price']}")
                st.write(f"Beklenen Fiyat: {res['expected_price']}")
                st.write(f"Beklenen Artış Yüzdesi: {res['expected_increase_percentage']:.2f}%")
                st.write(f"RSI: {res['rsi_14']:.2f}, MACD Line: {res['macd_line']:.4f}, MACD Signal: {res['macd_signal']:.4f}")
                st.image(res['plot'])
                with st.expander("Debug Bilgisi"):
                    st.json(res['debug'])
        df_res = pd.DataFrame(results)
        csv = df_res.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">CSV Sonuçlarını İndir</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("Güçlü alım sinyali bulunamadı.")

if __name__ == "__main__":
    main()
