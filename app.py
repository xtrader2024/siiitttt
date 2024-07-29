import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
import streamlit as st
from datetime import datetime, timedelta
import statsmodels.api as sm

# Binance API keys
api_key = 'TAAgJilKcF9LHg977hGa3fVXdd9TUv6EmaZu7YgkCa4f8aAcxT5lvRI1gkh8mvw2'  # Kendi API anahtarınızı buraya ekleyin
api_secret = 'Yw48JHkJu3dz0YpJrPJz9ektNHUvYZtNePTeQLzDAe0CRk33wyKbebyRV0q4xwJk'  # Kendi API gizli anahtarınızı buraya ekleyin
client = Client(api_key, api_secret)

# Sabit periyot değerleri
RSI_TIME_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BOLLINGER_WINDOW = 20

# Get historical data
def get_binance_data(symbol, interval, start_str, end_str):
    try:
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        if not klines or len(klines) < 51:
            st.write(f"Uyarı: {symbol} kripto para verisi alınamadı veya yetersiz.")
            return pd.DataFrame()  # Boş bir DataFrame döndür
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        return df
    except Exception as e:
        st.write(f"Veri alma hatası {symbol} için: {str(e)}")
        return pd.DataFrame()

# Calculate indicators
def calculate_indicators(df):
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    df['BB_Middle'] = df['close'].rolling(window=BOLLINGER_WINDOW).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(window=BOLLINGER_WINDOW).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(window=BOLLINGER_WINDOW).std()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_TIME_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_TIME_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['MACD_Line'] = df['close'].ewm(span=MACD_FAST_PERIOD, adjust=False).mean() - df['close'].ewm(span=MACD_SLOW_PERIOD, adjust=False).mean()
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    
    df['Prev_Close'] = df['close'].shift(1)
    df['TR'] = df[['high', 'Prev_Close']].max(axis=1) - df[['low', 'Prev_Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    return df

# Generate buy signals based on indicators
def generate_signals(df):
    df['Buy_Signal'] = (df['close'] > df['SMA_50']) & (df['RSI'] < 30)
    df['Sell_Signal'] = (df['close'] < df['SMA_50']) & (df['RSI'] > 70)
    return df

# Forecast next day price
def forecast_next_price(df):
    df = df.copy()
    df['day'] = np.arange(len(df))
    X = df[['day']]
    y = df['close']
    
    # Fit the model
    model = sm.OLS(y, sm.add_constant(X)).fit()
    
    # Prepare next_day for prediction
    next_day_index = np.array([[len(df) + 1]])  # [[next_day]] biçiminde
    next_day_df = pd.DataFrame(next_day_index, columns=['day'])
    next_day_df = sm.add_constant(next_day_df, has_constant='add')  # Ensure constant column is added
    
    # Predict the next day's price
    forecast = model.predict(next_day_df)
    
    return forecast[0]

# Calculate expected price and percentage increase
def calculate_expected_price(df):
    if df.empty:
        return np.nan, np.nan
    
    price = df['close'].iloc[-1]
    sma_50 = df['SMA_50'].iloc[-1]
    
    if pd.isna(sma_50) or sma_50 == 0:
        return np.nan, np.nan
    
    # Calculate expected price
    expected_price = price * (1 + (price - sma_50) / sma_50)
    
    # Calculate expected increase percentage
    expected_increase_percentage = ((expected_price - price) / price) * 100 - 1  # Burada %1 eksik yazdırmak için -1 eklenmiştir
    
    return expected_price, expected_increase_percentage

# Get all USDT pairs
def get_all_usdt_pairs():
    exchange_info = client.get_exchange_info()
    symbols = [s['symbol'] for s in exchange_info['symbols']]
    usdt_pairs = [s for s in symbols if s.endswith('USDT')]
    return usdt_pairs

# Streamlit app
st.title('Kripto Para Analiz ve Tahmin')

# Input fields
interval = st.selectbox('Zaman Aralığı', ['1h', '4h', '1d'])

# Time range
end_date = datetime.now()
start_date = end_date - timedelta(days=51)
start_str = start_date.strftime('%d %b %Y')
end_str = end_date.strftime('%d %b %Y')

# Fetch all USDT pairs
usdt_pairs = get_all_usdt_pairs()
st.write(f"Analiz Edilen USDT Pariteleri: {len(usdt_pairs)}")

# Notify user that the process is starting
st.write("Kripto paralar çekiliyor ve analiz ediliyor...")

# Analyze each pair
results = []

for symbol in usdt_pairs:
    df = get_binance_data(symbol, interval, start_str, end_str)
    if df.empty:
        # Veri alınamayan veya yetersiz veri bulunan pariteleri atla
        continue
    
    try:
        df = calculate_indicators(df)
        df = generate_signals(df)
        forecast = forecast_next_price(df)
        expected_price, expected_increase_percentage = calculate_expected_price(df)
        
        if df['Buy_Signal'].iloc[-1]:  # Check the most recent signal
            results.append({
                'Symbol': symbol,
                'Forecast_Next_Day_Price': forecast,
                'Last_Close': df['close'].iloc[-1],
                'Buy_Signal': df['Buy_Signal'].iloc[-1],
                'SMA_50': df['SMA_50'].iloc[-1],
                'EMA_50': df['EMA_50'].iloc[-1],
                'RSI': df['RSI'].iloc[-1],
                'MACD_Line': df['MACD_Line'].iloc[-1],
                'MACD_Signal': df['MACD_Signal'].iloc[-1],
                'BB_Upper': df['BB_Upper'].iloc[-1],
                'BB_Lower': df['BB_Lower'].iloc[-1],
                'ATR': df['ATR'].iloc[-1],
                'Expected_Price_24h': expected_price,
                'Expected_Increase_Percentage': expected_increase_percentage
            })
    except Exception as e:
        st.write(f"Analiz hatası {symbol} için: {str(e)}")
        continue

# Display results
if len(results) > 0:
    results_df = pd.DataFrame(results)
    st.write("Analiz Sonuçları:")
    st.write(results_df)
else:
    st.write("Alım sinyali veren coin bulunamadı.")

# Optionally, plot data for the first coin in the results
if len(results) > 0:
    symbol = results[0]['Symbol']
    df = get_binance_data(symbol, interval, start_str, end_str)
    if df.empty:
        st.write(f"{symbol} için yeterli veri bulunamadı.")
    else:
        df = calculate_indicators(df)
        df = generate_signals(df)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df.index, df['close'], label='Kapanış Fiyatı', color='blue')
        ax.plot(df.index, df['SMA_50'], label='50 Günlük SMA', color='green')
        ax.plot(df.index, df['EMA_50'], label='50 Günlük EMA', color='red')
        ax.plot(df.index, df['BB_Upper'], label='BB Üst Bandı', color='purple', linestyle='--')
        ax.plot(df.index, df['BB_Lower'], label='BB Alt Bandı', color='purple', linestyle='--')
        ax.plot(df.index, df['ATR'], label='ATR', color='orange')
        ax.set_title(f'{symbol} Analizi')
        ax.set_xlabel('Tarih')
        ax.set_ylabel('Fiyat')
        ax.legend()
        st.pyplot(fig)
