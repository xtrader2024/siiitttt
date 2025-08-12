import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta  # pip install ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Detaylı)", layout="centered")

MIN_PERIOD = 30  # Minimum veri satırı sayısı, indikatörlerin doğru hesaplanması için

def get_data(symbol, period="3mo", interval="1d"):
    try:
        df = yf.download(f"{symbol}.IS", period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        if df.empty or len(df) < MIN_PERIOD:
            st.warning(f"{symbol}: Yeterli veri yok (minimum {MIN_PERIOD} satır gerekli).")
            return None
        return df
    except Exception as e:
        st.error(f"Veri alınırken hata: {e}")
        return None

def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    inds = {}

    # Trend: SMA, EMA
    inds['SMA20'] = close.rolling(window=20).mean().iloc[-1]
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean().iloc[-1]

    # Momentum: RSI, CCI
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1]

    # Volume: MFI
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]

    # Trend Strength: ADX
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

    # Oscillators: Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    inds['STOCH_K'] = stoch.stoch().iloc[-1]
    inds['STOCH_D'] = stoch.stoch_signal().iloc[-1]

    # MACD
    macd = ta.trend.MACD(close)
    inds['MACD'] = macd.macd().iloc[-1]
    inds['MACD_SIGNAL'] = macd.macd_signal().iloc[-1]

    # Williams %R
    willr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    inds['WILLR'] = willr.williams_r().iloc[-1]

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    inds['OBV'] = obv.on_balance_volume().iloc[-1]

    return inds

def estimate_target_price(close_price, df_4h):
    try:
        df_4h = df_4h.copy()
        df_4h['log_return'] = np.log(df_4h['Close'] / df_4h['Close'].shift(1))
        avg_log_return = df_4h['log_return'].mean()
        vol_log_return = df_4h['log_return'].std()

        periods = 5  # 5 gün sonrası hedef

        # Beklenen log getiri ve standart sapma için çarpan (örneğin 1.96 ~ %95 güven aralığı)
        z = 1.96

        expected_price = close_price * np.exp(avg_log_return * periods)
        upper_bound = expected_price * np.exp(z * vol_log_return * np.sqrt(periods))
        lower_bound = expected_price * np.exp(-z * vol_log_return * np.sqrt(periods))

        return expected_price, lower_bound, upper_bound
    except Exception as e:
        st.warning(f"Hedef fiyat tahmini yapılamadı: {e}")
        return None, None, None

def analyze_trend_momentum(inds, close_price, df_4h):
    trend = "Yukarı" if close_price > inds['EMA20'] and close_price > inds['SMA20'] else "Aşağı"
    trend_strength = "Güçlü" if inds['ADX'] > 25 else "Zayıf"
    momentum = "Pozitif" if inds['RSI'] > 50 and inds['MACD'] > inds['MACD_SIGNAL'] else "Negatif"

    target_price, lower_bound, upper_bound = estimate_target_price(close_price, df_4h)

    return trend, trend_strength, momentum, target_price, lower_bound, upper_bound

st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

symbol = st.text_input("🔎 Hisse kodunu girin (örn: AEFES)", value="AEFES").upper()

if symbol:
    st.write(f"📈 {symbol} için analiz başlatılıyor...")

    df = get_data(symbol, period="3mo", interval="1d")
    df_4h = get_data(symbol, period="7d", interval="4h")

    if df is None or df_4h is None:
        st.error("Yeterli veri alınamadı, lütfen tekrar deneyin.")
    else:
        try:
            inds = calculate_indicators(df)
            close_price = df['Close'].iloc[-1]

            st.markdown(f"### {symbol} Analiz Sonucu")
            st.write(f"- **Son Kapanış Fiyatı:** {close_price:.2f} ₺")

            st.markdown("#### Teknik İndikatörler ve Osilatörler")
            for k, v in inds.items():
                st.write(f"- {k}: {v:.2f}")

            trend, trend_strength, momentum, target_price, lower_bound, upper_bound = analyze_trend_momentum(inds, close_price, df_4h)

            st.markdown("#### 📊 Genel Teknik Yorum:")
            st.write(f"- **Trend Yönü:** {trend}")
            st.write(f"- **Trend Gücü (ADX):** {trend_strength}")
            st.write(f"- **Momentum (RSI + MACD):** {momentum}")

            if target_price:
                st.write(f"- **Tahmini Hedef Fiyat (5 gün sonrası):** {target_price:.2f} ₺")
                st.write(f"  - Güven aralığı: {lower_bound:.2f} ₺ - {upper_bound:.2f} ₺")
            else:
                st.write("- **Tahmini Hedef Fiyat:** Hesaplanamadı")

        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {e}")
