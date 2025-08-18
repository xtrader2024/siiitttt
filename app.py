import yfinance as yf
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

getcontext().prec = 50

# Teknik gösterge parametreleri
BOLLINGER_WINDOW = 20
RSI_TIME_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
STOCH_FASTK_PERIOD = 14
STOCH_SLOWK_PERIOD = 3

# Örnek BIST100 hisseleri
BIST100_SYMBOLS = ["AKBNK.IS", "GARAN.IS", "THYAO.IS", "ASELS.IS", "KRDMD.IS"]  # istediğin hisse sembollerini ekle

# Dil seçenekleri
TEXTS = {
    'en': {
        'title': 'BIST100 Stock Analysis',
        'time_interval': 'Time Interval',
        'start_analysis': 'Start Analysis',
        'insufficient_data': 'Insufficient data',
        'current_price': 'Current Price',
        'expected_price': 'Expected Price',
        'expected_increase_percentage': 'Expected Increase Percentage',
        'sma_50': 'SMA 50',
        'rsi_14': 'RSI 14',
        'macd_line': 'MACD Line',
        'macd_signal': 'MACD Signal',
        'bb_upper_band': 'BB Upper Band',
        'bb_middle_band': 'BB Middle Band',
        'bb_lower_band': 'BB Lower Band',
        'atr': 'ATR',
        'stochastic_k': 'Stochastic %K',
        'stochastic_d': 'Stochastic %D',
        'entry_price': 'Entry Price',
        'take_profit_price': 'Take Profit Price',
        'stop_loss_price': 'Stop Loss Price',
        'signal_comment': 'Signal Comment',
        'download_csv': 'Download CSV Results',
        'debug_info': 'Debug Info'
    },
    'tr': {
        'title': 'BIST100 Hisse Analizi',
        'time_interval': 'Zaman Aralığı',
        'start_analysis': 'Analiz Başlat',
        'insufficient_data': 'Yetersiz veri',
        'current_price': 'Mevcut Fiyat',
        'expected_price': 'Beklenen Fiyat',
        'expected_increase_percentage': 'Beklenen Artış Yüzdesi',
        'sma_50': 'SMA 50',
        'rsi_14': 'RSI 14',
        'macd_line': 'MACD Çizgisi',
        'macd_signal': 'MACD Sinyali',
        'bb_upper_band': 'BB Üst Bandı',
        'bb_middle_band': 'BB Orta Bandı',
        'bb_lower_band': 'BB Alt Bandı',
        'atr': 'ATR',
        'stochastic_k': 'Stokastik %K',
        'stochastic_d': 'Stokastik %D',
        'entry_price': 'Giriş Fiyatı',
        'take_profit_price': 'Kar Alma Fiyatı',
        'stop_loss_price': 'Zarar Durdur Fiyatı',
        'signal_comment': 'Sinyal Yorumu',
        'download_csv': 'CSV Sonuçlarını İndir',
        'debug_info': 'Debug Bilgisi'
    }
}

# Teknik göstergeler
def calculate_indicators(df):
    # SMA ve EMA
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False, min_periods=1).mean()

    # Bollinger Band
    df['BB_Middle'] = df['Close'].rolling(window=BOLLINGER_WINDOW, min_periods=1).mean()
    df['BB_Std'] = df['Close'].rolling(window=BOLLINGER_WINDOW, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_TIME_PERIOD, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_TIME_PERIOD, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)

    # MACD
    ema_fast = df['Close'].ewm(span=MACD_FAST_PERIOD, adjust=False, min_periods=1).mean()
    ema_slow = df['Close'].ewm(span=MACD_SLOW_PERIOD, adjust=False, min_periods=1).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=MACD_SIGNAL_PERIOD, adjust=False, min_periods=1).mean()

    # ATR
    df['Prev_Close'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Prev_Close']].max(axis=1) - df[['Low', 'Prev_Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=14, min_periods=1).mean()

    # Stochastic
    df['Lowest_Low'] = df['Low'].rolling(window=STOCH_FASTK_PERIOD, min_periods=1).min()
    df['Highest_High'] = df['High'].rolling(window=STOCH_FASTK_PERIOD, min_periods=1).max()
    df['%K'] = 100 * (df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low']).replace(0, np.nan)
    df['%K'] = df['%K'].fillna(0)
    df['%D'] = df['%K'].rolling(window=STOCH_SLOWK_PERIOD, min_periods=1).mean()

    return df

# Destek ve direnç
def calculate_support_resistance(df):
    df['Support'] = df['Low'].rolling(window=50, min_periods=1).min()
    df['Resistance'] = df['High'].rolling(window=50, min_periods=1).max()
    return df

# Sinyal üretimi
def generate_signals(df, rsi_lower, rsi_upper, min_expected_increase):
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

# Fiyat tahmini (basit lineer regresyon)
def forecast_next_price(df):
    df = df.copy()
    df['day'] = np.arange(len(df))
    X = sm.add_constant(df[['day']])
    y = df['Close']
    model = sm.OLS(y, X).fit()
    next_day_df = sm.add_constant(pd.DataFrame({'day':[len(df)]}), has_constant='add')
    forecast = model.predict(next_day_df)
    return forecast[0]

# Beklenen fiyat ve artış
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

# Alım-satım seviyeleri
def calculate_trade_levels(df, entry_pct=0.02, take_profit_pct=0.05, stop_loss_pct=0.02):
    if df.empty:
        return np.nan, np.nan, np.nan
    entry_price = Decimal(df['Close'].iloc[-1])
    take_profit_price = entry_price * (1 + Decimal(take_profit_pct))
    stop_loss_price = entry_price * (1 - Decimal(stop_loss_pct))
    return float(entry_price), float(take_profit_price), float(stop_loss_price)

# Grafik oluşturma
def plot_to_png(df, symbol, entry=None, tp=None, sl=None):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax.plot(df.index, df['SMA_50'], label='SMA 50', color='green')
    ax.plot(df.index, df['EMA_50'], label='EMA 50', color='red')
    ax.plot(df.index, df['BB_Upper'], label='BB Upper Band', color='purple', linestyle='--')
    ax.plot(df.index, df['BB_Lower'], label='BB Lower Band', color='purple', linestyle='--')
    if 'Support' in df.columns:
        ax.plot(df.index, df['Support'], label='Support', color='cyan', linestyle='--')
    if 'Resistance' in df.columns:
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

# Hisse işleme
def process_stock(symbol, interval, rsi_lower, rsi_upper, min_expected_increase):
    try:
        df = yf.download(symbol, period="6mo", interval=interval)
        if df.empty or len(df) < 50:
            return None
        df = calculate_indicators(df)
        df = calculate_support_resistance(df)
        df = generate_signals(df, rsi_lower, rsi_upper, min_expected_increase)
        forecast = forecast_next_price(df)
        expected_price, expected_increase_percentage = calculate_expected_price(df)
        entry_price, take_profit_price, stop_loss_price = calculate_trade_levels(df)
        if df['Buy_Signal'].iloc[-3:].any() and expected_increase_percentage >= min_expected_increase:
            debug_info = {
                'RSI_Last': df['RSI'].iloc[-1],
                'MACD_Line_Last': df['MACD_Line'].iloc[-1]
            }
            return {
                'coin_name': symbol,
                'price': df['Close'].iloc[-1],
                'expected_price': expected_price,
                'expected_increase_percentage': expected_increase_percentage,
                'sma_50': df['SMA_50'].iloc[-1],
                'rsi_14': df['RSI'].iloc[-1],
                'macd_line': df['MACD_Line'].iloc[-1],
                'macd_signal': df['MACD_Signal'].iloc[-1],
                'bb_upper': df['BB_Upper'].iloc[-1],
                'bb_middle': df['BB_Middle'].iloc[-1],
                'bb_lower': df['BB_Lower'].iloc[-1],
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
    except Exception as e:
        st.error(f"Processing error ({symbol}): {e}")
        return None

# Streamlit arayüzü
def main():
    global language
    language = st.selectbox('Select Language / Dil Seçin', ['en', 'tr'])
    st.title(TEXTS[language]['title'])

    interval = st.selectbox(TEXTS[language]['time_interval'], ['1d', '1wk'], index=0)
    rsi_lower = st.slider('RSI Lower Bound / RSI Alt Sınır', 30, 70, 45)
    rsi_upper = st.slider('RSI Upper Bound / RSI Üst Sınır', 50, 90, 75)
    min_expected_increase = st.slider('Min Expected Increase % / Minimum Beklenen Artış %', 0, 20, 5)
    start_button = st.button(TEXTS[language]['start_analysis'])
    if not start_button: return

    results = []
    with st.spinner('Analyzing...'):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_stock, s, interval, rsi_lower, rsi_upper, min_expected_increase) for s in BIST100_SYMBOLS]
            for future in as_completed(futures):
                res = future.result()
                if res: results.append(res)

    if results:
        signals = [r for r in results if r['signal_comment'] != 'No signal']
        no_signals = [r for r in results if r['signal_comment'] == 'No signal']
        sorted_results = signals + no_signals

        st.write(f"Total stocks analyzed: {len(results)}")
        for res in sorted_results:
            with st.expander(f"{res['coin_name']} - {res['signal_comment']}"):
                st.write(f"{TEXTS[language]['current_price']}: {res['price']}")
                st.write(f"{TEXTS[language]['expected_price']}: {res['expected_price']}")
                st.write(f"{TEXTS[language]['expected_increase_percentage']}: {res['expected_increase_percentage']:.2f}%")
                st.write(f"RSI: {res['rsi_14']:.2f}, MACD Line: {res['macd_line']:.4f}, MACD Signal: {res['macd_signal']:.4f}")
                st.image(res['plot'])
                with st.expander(TEXTS[language]['debug_info']):
                    st.json(res['debug'])

        df_res = pd.DataFrame(results)
        csv = df_res.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">{TEXTS[language]["download_csv"]}</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("No strong buy signals found.")

if __name__ == "__main__":
    main()
