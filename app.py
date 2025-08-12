import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Detaylı)", layout="centered")

def get_data(symbol, period="3mo", interval="1d"):
    try:
        df = yf.download(f"{symbol}.IS", period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"{symbol} için veri bulunamadı.")
            return None
        # Eğer çoklu seviye kolon varsa düzelt
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
    try:
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        inds = {
            'SMA20': close.rolling(window=20).mean().iloc[-1],
            'EMA20': close.ewm(span=20, adjust=False).mean().iloc[-1],
            'RSI': ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1],
            'CCI': ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1],
            'MFI': ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1],
            'ADX': ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1],
            'STOCH_K': ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch().iloc[-1],
            'STOCH_D': ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch_signal().iloc[-1],
            'MACD': ta.trend.MACD(close).macd().iloc[-1],
            'MACD_SIGNAL': ta.trend.MACD(close).macd_signal().iloc[-1],
            'WILLR': ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1],
            'OBV': ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]
        }
        return inds
    except Exception as e:
        st.error(f"İndikatör hesaplama hatası: {e}")
        return None

def analyze_trend_momentum(inds, close_price, df_4h):
    try:
        trend = "Yukarı" if close_price > inds['EMA20'] and close_price > inds['SMA20'] else "Aşağı"
        trend_strength = "Güçlü" if inds['ADX'] > 25 else "Zayıf"
        momentum = "Pozitif" if inds['RSI'] > 50 and inds['MACD'] > inds['MACD_SIGNAL'] else "Negatif"

        target_price = None
        if df_4h is not None and not df_4h.empty:
            avg_change = df_4h['Close'].pct_change().mean()
            target_price = close_price * (1 + avg_change * 5)
        return trend, trend_strength, momentum, target_price
    except Exception as e:
        st.warning(f"Trend/momentum analizi hatası: {e}")
        return "Bilinmiyor", "Bilinmiyor", "Bilinmiyor", None

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

# Streamlit UI
st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

symbol = st.text_input("🔎 Hisse kodunu girin (örn: AEFES)", value="ASELS").upper().strip()

if symbol:
    st.write(f"📈 {symbol} için analiz başlatılıyor...")

    df_daily = get_data(symbol, period="3mo", interval="1d")
    df_4h = get_data(symbol, period="7d", interval="4h")

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

            st.markdown("#### Teknik İndikatörler ve Osilatörler")
            for k, v in inds.items():
                st.write(f"- {k}: {v:.2f}")

            trend, trend_strength, momentum, target_price = analyze_trend_momentum(inds, close_price, df_4h)

            st.markdown("#### 📊 Genel Teknik Yorum:")
            st.write(f"- **Trend Yönü:** {trend}")
            st.write(f"- **Trend Gücü (ADX):** {trend_strength}")
            st.write(f"- **Momentum (RSI + MACD):** {momentum}")
            if target_price is not None:
                st.write(f"- **Tahmini Hedef Fiyat (5 gün sonrası):** {target_price:.2f} ₺")
            else:
                st.write("- **Tahmini Hedef Fiyat:** Hesaplanamadı")

            st.markdown("#### 🔔 İndikatörlere Göre Sinyaller:")
            signals = generate_signals(inds, close_price)
            for s in signals:
                st.write(f"- {s}")
