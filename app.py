import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from decimal import Decimal, getcontext
import base64
import yfinance as yf

getcontext().prec = 50

# Örnek BIST100 sembolleri
BIST100_SYMBOLS = [
    'GARAN.IS', 'AKBNK.IS', 'ISCTR.IS', 'ASELS.IS', 'THYAO.IS'
]

BOLLINGER_WINDOW = 20
RSI_TIME_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
STOCH_FASTK_PERIOD = 14
STOCH_SLOWK_PERIOD = 3

def get_stock_data(symbol, interval='1d', period='3mo'):
    try:
        df = yf.download(symbol, interval=interval, period=period)
        if df.empty or len(df) < 51:
            st.warning(f"Yetersiz veri: {symbol}")
            return pd.DataFrame()
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"Veri çekme hatası ({symbol}): {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df['BB_Middle'] = df['Close'].rolling(window=BOLLINGER_WINDOW).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=BOLLINGER_WINDOW).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=BOLLINGER_WINDOW).std()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_TIME_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_TIME_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD_Line'] = df['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean() - df['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()

    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Prev_Close']].max(axis=1) - df[['Low', 'Prev_Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['Lowest_Low'] = df['Low'].rolling(window=STOCH_FASTK_PERIOD).min()
    df['Highest_High'] = df['High'].rolling(window=STOCH_FASTK_PERIOD).max()
    df['%K'] = 100 * (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])
    df['%D'] = df['%K'].rolling(window=STOCH_SLOWK_PERIOD).mean()

    return df

def calculate_support_resistance(df):
    df['Support'] = df['Low'].rolling(window=50).min()
    df['Resistance'] = df['High'].rolling(window=50).max()
    return df

def generate_signals(df, rsi_lower, rsi_upper):
    df['Buy_Signal'] = (
        (df['Close'] > df['SMA_50']) &
        (df['EMA_50'] > df['EMA_200']) &
        (df['MACD_Line'] > df['MACD_Signal']) &
        (df['%K'] > df['%D']) &
        (df['RSI'] > rsi_lower) & (df['RSI'] < rsi_upper)
    )

    df['Sell_Signal'] = (
        (df['Close'] < df['SMA_50']) &
        (df['EMA_50'] < df['EMA_200']) &
        (df['MACD_Line'] < df['MACD_Signal']) &
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
    next_day_index = np.array([[len(df) + 1]])
    next_day_df = pd.DataFrame(next_day_index, columns=['day'])
    next_day_df = sm.add_constant(next_day_df, has_constant='add')
    forecast = model.predict(next_day_df)
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
    ax.plot(df.index, df['BB_Upper'], label='BB Upper', color='purple', linestyle='--')
    ax.plot(df.index, df['BB_Lower'], label='BB Lower', color='purple', linestyle='--')
    ax.plot(df.index, df['ATR'], label='ATR', color='orange')
    ax.plot(df.index, df['Support'], label='Support', color='cyan', linestyle='--')
    ax.plot(df.index, df['Resistance'], label='Resistance', color='magenta', linestyle='--')
    if entry: ax.axhline(entry, color='lime', linestyle='-.', label='Entry Price')
    if tp: ax.axhline(tp, color='gold', linestyle='-.', label='Take Profit')
    if sl: ax.axhline(sl, color='red', linestyle='-.', label='Stop Loss')
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

# Process single stock
def process_stock(symbol, rsi_lower, rsi_upper):
    df = get_stock_data(symbol)
    if df.empty: return None
    df = calculate_indicators(df)
    df = calculate_support_resistance(df)
    df = generate_signals(df, rsi_lower, rsi_upper)
    forecast = forecast_next_price(df)
    expected_price, expected_increase_percentage = calculate_expected_price(df)
    entry_price, take_profit_price, stop_loss_price = calculate_trade_levels(df)
    if df['Buy_Signal'].iloc[-3:].any():
        return {
            'stock': symbol,
            'price': df['Close'].iloc[-1],
            'expected_price': expected_price,
            'expected_increase_percentage': expected_increase_percentage,
            'signal_comment': df['Signal_Comment'].iloc[-1],
            'entry_price': entry_price,
            'take_profit_price': take_profit_price,
            'stop_loss_price': stop_loss_price,
            'plot': plot_to_png(df, symbol, entry=entry_price, tp=take_profit_price, sl=stop_loss_price)
        }
    else:
        return None

# Streamlit interface
def main():
    st.title("BIST100 Hisse Analizi")
    rsi_lower = st.slider('RSI Alt Sınır', 30, 70, 45)
    rsi_upper = st.slider('RSI Üst Sınır', 50, 90, 75)
    start_button = st.button("Analiz Başlat")
    if not start_button: return

    results = []
    with st.spinner('Analiz ediliyor...'):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_stock, symbol, rsi_lower, rsi_upper) for symbol in BIST100_SYMBOLS]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

    if results:
        for res in results:
            with st.expander(f"{res['stock']} - {res['signal_comment']}"):
                st.write(f"Mevcut Fiyat: {res['price']}")
                st.write(f"Beklenen Fiyat: {res['expected_price']}")
                st.write(f"Beklenen Artış %: {res['expected_increase_percentage']:.2f}%")
                st.image(res['plot'])
        df_res = pd.DataFrame(results)
        csv = df_res.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">CSV Sonuçlarını İndir</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("Güçlü alım sinyali bulunamadı.")

if __name__ == "__main__":
    main()
