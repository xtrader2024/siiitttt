import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta  # pip install ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Detaylı)", layout="centered")

def get_data(symbol, period="3mo", interval="1d"):
    try:
        df = yf.download(f"{symbol}.IS", period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("Veri boş geldi.")
        return df
    except Exception as e:
        st.error(f"Veri alınırken hata oluştu: {e}")
        return None

def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    inds = {}

    # Trend göstergeleri
    inds['SMA20'] = close.rolling(window=20).mean()
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean()

    # Momentum
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()

    # Hacim
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()

    # Trend gücü
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    # Osilatörler
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    inds['STOCH_K'] = stoch.stoch()
    inds['STOCH_D'] = stoch.stoch_signal()

    macd = ta.trend.MACD(close)
    inds['MACD'] = macd.macd()
    inds['MACD_SIGNAL'] = macd.macd_signal()

    willr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    inds['WILLR'] = willr.williams_r()

    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    inds['OBV'] = obv.on_balance_volume()

    # Son değerleri al
    last_inds = {k: v.iloc[-1] for k,v in inds.items()}
    return last_inds

def generate_signals(inds, close_price):
    signals = []

    # RSI
    if inds['RSI'] > 70:
        signals.append("RSI aşırı alım → Satış sinyali")
    elif inds['RSI'] < 30:
        signals.append("RSI aşırı satım → Alış sinyali")
    else:
        signals.append("RSI nötr")

    # ADX
    if inds['ADX'] > 25:
        signals.append("Trend güçlü")
    else:
        signals.append("Trend zayıf veya yatay")

    # MACD
    if inds['MACD'] > inds['MACD_SIGNAL']:
        signals.append("MACD pozitif → Alış sinyali")
    else:
        signals.append("MACD negatif → Satış sinyali")

    # SMA/EMA trend yönü
    if close_price > inds['EMA20'] and close_price > inds['SMA20']:
        signals.append("Fiyat EMA20 ve SMA20 üzerinde → Yukarı trend")
    elif close_price < inds['EMA20'] and close_price < inds['SMA20']:
        signals.append("Fiyat EMA20 ve SMA20 altında → Aşağı trend")
    else:
        signals.append("Fiyat SMA ve EMA arasında → Belirsiz trend")

    # CCI
    if inds['CCI'] > 100:
        signals.append("CCI aşırı alım → Satış sinyali")
    elif inds['CCI'] < -100:
        signals.append("CCI aşırı satım → Alış sinyali")
    else:
        signals.append("CCI nötr")

    # MFI
    if inds['MFI'] > 80:
        signals.append("MFI aşırı alım → Satış sinyali")
    elif inds['MFI'] < 20:
        signals.append("MFI aşırı satım → Alış sinyali")
    else:
        signals.append("MFI nötr")

    # Williams %R
    if inds['WILLR'] < -80:
        signals.append("Williams %R aşırı satım → Alış sinyali")
    elif inds['WILLR'] > -20:
        signals.append("Williams %R aşırı alım → Satış sinyali")
    else:
        signals.append("Williams %R nötr")

    return signals

def analyze_trend_momentum(df_daily, df_4h, symbol):
    close_price = df_daily['Close'].iloc[-1]
    inds = calculate_indicators(df_daily)

    signals = generate_signals(inds, close_price)

    # Basit hedef fiyat tahmini
    try:
        avg_change = df_4h['Close'].pct_change().mean()
        target_price = close_price * (1 + avg_change * 5)
    except Exception as e:
        target_price = None
        st.warning(f"Hedef fiyat hesaplanamadı: {e}")

    return inds, signals, close_price, target_price

st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

symbol = st.text_input("🔎 Hisse kodunu girin (örn: AEFES)", value="AEFES").upper()

if symbol:
    st.write(f"📈 {symbol} için analiz başlatılıyor...")

    df_daily = get_data(symbol, period="3mo", interval="1d")
    df_4h = get_data(symbol, period="7d", interval="4h")

    if df_daily is None or df_4h is None:
        st.error("Veriler tam olarak alınamadı, lütfen tekrar deneyin.")
    else:
        try:
            inds, signals, close_price, target_price = analyze_trend_momentum(df_daily, df_4h, symbol)

            st.markdown(f"### {symbol} Analiz Sonucu")
            st.write(f"- **Son Kapanış Fiyatı:** {close_price:.2f} ₺")

            st.markdown("#### Teknik İndikatörler ve Osilatörler (Son Değerler)")
            for k, v in inds.items():
                st.write(f"- {k}: {v:.2f}")

            st.markdown("#### 📊 Al/Sat/Nötr Sinyaller")
            for s in signals:
                st.write(f"- {s}")

            st.markdown("#### 📈 Tahmini Hedef Fiyat ve Trend Yorumu")
            if target_price:
                st.write(f"- Tahmini hedef fiyat (5 gün sonrası): {target_price:.2f} ₺")
            else:
                st.write("- Tahmini hedef fiyat hesaplanamadı.")

        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {e}")
