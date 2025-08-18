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

BOLLINGER_WINDOW = 20
RSI_TIME_PERIOD = 14

TEXTS = {
    'en': {
        'title': 'BIST100 Stock Analysis',
        'start_analysis': 'Start Analysis',
        'insufficient_data': 'Insufficient data',
        'total_coins_analyzed': 'Total stocks analyzed',
        'current_price': 'Current Price',
        'expected_price': 'Expected Price',
        'expected_increase_percentage': 'Expected Increase Percentage',
        'sma_50': 'SMA 50',
        'rsi_14': 'RSI 14',
        'bb_upper_band': 'BB Upper Band',
        'bb_middle_band': 'BB Middle Band',
        'bb_lower_band': 'BB Lower Band',
        'entry_price': 'Entry Price',
        'take_profit_price': 'Take Profit Price',
        'stop_loss_price': 'Stop Loss Price',
        'signal_comment': 'Signal Comment',
        'download_csv': 'Download CSV Results',
        'debug_info': 'Debug Info'
    },
    'tr': {
        'title': 'BIST100 Hisse Analizi',
        'start_analysis': 'Analiz Başlat',
        'insufficient_data': 'Yetersiz veri',
        'total_coins_analyzed': 'Analiz edilen toplam hisse sayısı',
        'current_price': 'Mevcut Fiyat',
        'expected_price': 'Beklenen Fiyat',
        'expected_increase_percentage': 'Beklenen Artış Yüzdesi',
        'sma_50': 'SMA 50',
        'rsi_14': 'RSI 14',
        'bb_upper_band': 'BB Üst Bandı',
        'bb_middle_band': 'BB Orta Bandı',
        'bb_lower_band': 'BB Alt Bandı',
        'entry_price': 'Giriş Fiyatı',
        'take_profit_price': 'Kar Alma Fiyatı',
        'stop_loss_price': 'Zarar Durdur Fiyatı',
        'signal_comment': 'Sinyal Yorumu',
        'download_csv': 'CSV Sonuçlarını İndir',
        'debug_info': 'Debug Bilgisi'
    }
}

# BIST100 sembollerini Yahoo Finance üzerinden çekme
def get_bist100_symbols():
    try:
        url = "https://www investing.com/indices/ise-100-components"  # Örnek site, CSV veya API yoksa manuel liste gerekebilir
        # Alternatif olarak güncel sembolleri hazır bir CSV dosyasından çekebilirsin.
        # Şimdilik örnek hardcoded liste:
        return [
            "ASELS.IS", "AKBNK.IS", "THYAO.IS", "GARAN.IS", "ISCTR.IS",
            "KCHOL.IS", "PETKM.IS", "SISE.IS", "VAKBN.IS", "YKBNK.IS"
        ]
    except Exception as e:
        st.error(f"Could not fetch BIST100 symbols: {e}")
        return []

def fetch_data(symbol, period='6mo', interval='1d'):
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df.empty or len(df) < 50:
            return pd.DataFrame()
        df = df[['Open','High','Low','Close','Volume']].copy()
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"{symbol} veri çekme hatası: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    if len(df) < max(BOLLINGER_WINDOW, 50, RSI_TIME_PERIOD):
        return pd.DataFrame()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['BB_Middle'] = df['Close'].rolling(BOLLINGER_WINDOW).mean()
    rolling_std = df['Close'].rolling(BOLLINGER_WINDOW).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * rolling_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * rolling_std
    df[['BB_Upper','BB_Middle','BB_Lower']] = df[['BB_Upper','BB_Middle','BB_Lower']].fillna(method='bfill')
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0).rolling(RSI_TIME_PERIOD).mean()
    loss = -delta.where(delta<0,0).rolling(RSI_TIME_PERIOD).mean()
    rs = gain/loss
    df['RSI'] = 100 - (100/(1+rs))
    return df

def forecast_next_price(df):
    df = df.copy()
    df['day'] = np.arange(len(df))
    X = df[['day']]
    y = df['Close']
    model = sm.OLS(y, sm.add_constant(X)).fit()
    next_day_df = sm.add_constant(pd.DataFrame({'day':[len(df)+1]}))
    forecast = model.predict(next_day_df)
    return forecast[0]

def calculate_expected_price(df):
    if df.empty:
        return np.nan, np.nan
    price = Decimal(df['Close'].iloc[-1])
    sma_50 = Decimal(df['SMA_50'].iloc[-1])
    if pd.isna(sma_50) or sma_50==0:
        return np.nan, np.nan
    expected_price = price * (1 + (price - sma_50)/sma_50)
    expected_increase_percentage = ((expected_price - price)/price)*100
    return float(expected_price), float(expected_increase_percentage)

def calculate_trade_levels(df, entry_pct=0.02, tp_pct=0.05, sl_pct=0.02):
    if df.empty:
        return np.nan, np.nan, np.nan
    entry_price = Decimal(df['Close'].iloc[-1])
    take_profit = entry_price*(1+Decimal(tp_pct))
    stop_loss = entry_price*(1-Decimal(sl_pct))
    return float(entry_price), float(take_profit), float(stop_loss)

def plot_to_png(df, symbol, entry=None, tp=None, sl=None):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df['Close'], label='Close', color='blue')
    ax.plot(df.index, df['SMA_50'], label='SMA 50', color='green')
    ax.plot(df.index, df['BB_Upper'], label='BB Upper', color='purple', linestyle='--')
    ax.plot(df.index, df['BB_Lower'], label='BB Lower', color='purple', linestyle='--')
    if entry: ax.axhline(entry,color='lime',linestyle='-.',label='Entry')
    if tp: ax.axhline(tp,color='gold',linestyle='-.',label='Take Profit')
    if sl: ax.axhline(sl,color='red',linestyle='-.',label='Stop Loss')
    ax.set_title(f'{symbol} Analysis')
    ax.set_xlabel('Date'); ax.set_ylabel('Price')
    ax.legend(); ax.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return img

def process_symbol(symbol):
    df = fetch_data(symbol)
    if df.empty:
        return None
    df = calculate_indicators(df)
    expected_price, expected_inc = calculate_expected_price(df)
    entry, tp, sl = calculate_trade_levels(df)
    last_rsi = df['RSI'].iloc[-1]
    signal = "BUY" if last_rsi<70 and df['Close'].iloc[-1]>df['SMA_50'].iloc[-1] else "No signal"
    return {
        'symbol': symbol,
        'price': df['Close'].iloc[-1],
        'expected_price': expected_price,
        'expected_increase_percentage': expected_inc,
        'sma_50': df['SMA_50'].iloc[-1],
        'rsi_14': last_rsi,
        'entry_price': entry,
        'take_profit_price': tp,
        'stop_loss_price': sl,
        'signal_comment': signal,
        'plot': plot_to_png(df, symbol, entry, tp, sl)
    }

def main():
    language = st.selectbox('Select Language / Dil Seçin', ['en','tr'])
    st.title(TEXTS[language]['title'])
    start_button = st.button(TEXTS[language]['start_analysis'])
    if not start_button:
        return

    symbols = get_bist100_symbols()
    results = []
    with st.spinner('Analyzing...'):
        for sym in symbols:
            res = process_symbol(sym)
            if res: results.append(res)

    if results:
        for res in results:
            with st.expander(f"{res['symbol']} - {res['signal_comment']}"):
                st.write(f"{TEXTS[language]['current_price']}: {res['price']}")
                st.write(f"{TEXTS[language]['expected_price']}: {res['expected_price']}")
                st.write(f"{TEXTS[language]['expected_increase_percentage']}: {res['expected_increase_percentage']:.2f}%")
                st.write(f"SMA50: {res['sma_50']:.2f}, RSI14: {res['rsi_14']:.2f}")
                st.image(res['plot'])
        df_res = pd.DataFrame(results)
        csv = df_res.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="bist_analysis.csv">{TEXTS[language]["download_csv"]}</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("No strong signals found.")

if __name__=="__main__":
    main()
