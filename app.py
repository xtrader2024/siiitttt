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
        if len(df) < 30:
            st.warning(f"{symbol}: Yeterli veri yok (minimum 30 satır gerekli).")
            return None
        return df
    except Exception as e:
        st.error(f"Veri indirme hatası: {e}")
        return None

def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    inds = {}
    try:
        inds['SMA20'] = close.rolling(window=20).mean().iloc[-1]
        inds['EMA20'] = close.ewm(span=20, adjust=False).mean().iloc[-1]

        inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1]

        inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]
        inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        inds['STOCH_K'] = stoch.stoch().iloc[-1]
        inds['STOCH_D'] = stoch.stoch_signal().iloc[-1]

        macd = ta.trend.MACD(close)
        inds['MACD'] = macd.macd().iloc[-1]
        inds['MACD_SIGNAL'] = macd.macd_signal().iloc[-1]

        willr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
        inds['WILLR'] = willr.williams_r().iloc[-1]

        obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
        inds['OBV'] = obv.on_balance_volume().iloc[-1]
    except Exception as e:
        st.error(f"İndikatör hesaplama hatası: {e}")
        return None

    return inds

def generate_signals(inds, close_price):
    signals = []

    # RSI sinyali
    if inds['RSI'] > 70:
        signals.append("RSI aşırı alım → Satış sinyali")
    elif inds['RSI'] < 30:
        signals.append("RSI aşırı satım → Alım sinyali")
    else:
        signals.append("RSI nötr")

    # ADX sinyali
    if inds['ADX'] > 25:
        signals.append("ADX > 25 → Güçlü trend")
    else:
        signals.append("ADX ≤ 25 → Zayıf trend")

    # MACD sinyali
    if inds['MACD'] > inds['MACD_SIGNAL']:
        signals.append("MACD pozitif → Alım sinyali")
    else:
        signals.append("MACD negatif → Satış sinyali")

    # Stochastic sinyali
    if inds['STOCH_K'] > 80 and inds['STOCH_D'] > 80:
        signals.append("Stochastic aşırı alım → Satış sinyali")
    elif inds['STOCH_K'] < 20 and inds['STOCH_D'] < 20:
        signals.append("Stochastic aşırı satım → Alım sinyali")
    else:
        signals.append("Stochastic nötr")

    # CCI sinyali
    if inds['CCI'] > 100:
        signals.append("CCI yüksek → Alım sinyali")
    elif inds['CCI'] < -100:
        signals.append("CCI düşük → Satış sinyali")
    else:
        signals.append("CCI nötr")

    # Basit trend yönü
    if close_price > inds['EMA20'] and close_price > inds['SMA20']:
        signals.append("Fiyat EMA20 ve SMA20 üzerinde → Yukarı trend")
    else:
        signals.append("Fiyat EMA20 ve SMA20 altında → Aşağı trend")

    return signals

def analyze_trend_momentum(inds, close_price, df_4h):
    trend = "Yukarı" if close_price > inds['EMA20'] and close_price > inds['SMA20'] else "Aşağı"
    trend_strength = "Güçlü" if inds['ADX'] > 25 else "Zayıf"
    momentum = "Pozitif" if inds['RSI'] > 50 and inds['MACD'] > inds['MACD_SIGNAL'] else "Negatif"

    target_price = None
    try:
        if df_4h is not None and not df_4h.empty:
            avg_change = df_4h['Close'].pct_change().mean()
            target_price = close_price * (1 + avg_change * 5)
    except Exception as e:
        st.warning(f"Hedef fiyat tahmini sırasında hata: {e}")

    return trend, trend_strength, momentum, target_price

# Ana akış
st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

symbol = st.text_input("🔎 Hisse kodunu girin (örn: AEFES)", value="AEFES").upper()

if symbol:
    st.write(f"📈 {symbol} için analiz başlatılıyor...")

    df_daily = get_data(symbol, period="3mo", interval="1d")
    df_4h = get_data(symbol, period="7d", interval="4h")  # 4 saatlik veri burada indiriliyor

    if df_daily is None or df_4h is None:
        st.error("Veri alınamadı veya eksik. Lütfen hisse kodunu kontrol edip tekrar deneyin.")
    else:
        inds = calculate_indicators(df_daily)
        if inds is None:
            st.error("İndikatör hesaplanamadı.")
        else:
            close_price = df_daily['Close'].iloc[-1]

            st.markdown(f"### {symbol} Analiz Sonucu")
            st.write(f"- **Son Kapanış Fiyatı:** {close_price:.2f} ₺")

            # İndikatör değerleri detaylı liste
            st.markdown("#### Teknik İndikatörler ve Osilatörler")
            for k, v in inds.items():
                if isinstance(v, float):
                    st.write(f"- {k}: {v:.2f}")
                else:
                    st.write(f"- {k}: {v}")

            # Genel teknik yorum ve hedef fiyat
            trend, trend_strength, momentum, target_price = analyze_trend_momentum(inds, close_price, df_4h)

            st.markdown("#### 📊 Genel Teknik Yorum:")
            st.write(f"- **Trend Yönü:** {trend}")
            st.write(f"- **Trend Gücü (ADX):** {trend_strength}")
            st.write(f"- **Momentum (RSI + MACD):** {momentum}")
            if target_price:
                st.write(f"- **Tahmini Hedef Fiyat (5 gün sonrası):** {target_price:.2f} ₺")
            else:
                st.write("- **Tahmini Hedef Fiyat:** Hesaplanamadı")

            # AL/SAT/NÖTR sinyalleri
            st.markdown("#### 🔔 İndikatörlere Göre Sinyaller:")
            signals = generate_signals(inds, close_price)
            for s in signals:
                st.write(f"- {s}")
