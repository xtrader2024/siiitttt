import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="📊 Hisse Teknik Analiz", layout="centered")
st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

symbol = st.text_input("🔎 Hisse kodunu girin (örn: AEFES):").upper()

def analyze(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        if df.empty:
            return None, f"{symbol} için veri alınamadı."

        df.dropna(inplace=True)
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # İndikatör hesaplamaları
        indicators = {}
        indicators['RSI'] = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
        indicators['MACD'] = ta.trend.MACD(close).macd_diff().iloc[-1]
        indicators['SMA20'] = ta.trend.SMAIndicator(close, 20).sma_indicator().iloc[-1]
        indicators['EMA20'] = ta.trend.EMAIndicator(close, 20).ema_indicator().iloc[-1]
        indicators['MFI'] = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index().iloc[-1]
        indicators['ADX'] = ta.trend.ADXIndicator(high, low, close).adx().iloc[-1]
        indicators['CCI'] = ta.trend.CCIIndicator(high, low, close).cci().iloc[-1]
        indicators['STOCH'] = ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1]
        indicators['WILLR'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r().iloc[-1]
        indicators['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]

        latest_close = close.iloc[-1]
        sma20 = indicators['SMA20']
        ema20 = indicators['EMA20']

        # Puanlama ve açıklamalar
        score = 0
        details = []

        # RSI
        if indicators['RSI'] > 50:
            score += 1
            details.append(f"✅ RSI ({indicators['RSI']:.2f}) → Pozitif momentum")
        else:
            details.append(f"❌ RSI ({indicators['RSI']:.2f}) → Momentum zayıf")

        # MACD
        if indicators['MACD'] > 0:
            score += 1
            details.append(f"✅ MACD ({indicators['MACD']:.4f}) → Al sinyali")
        else:
            details.append(f"❌ MACD ({indicators['MACD']:.4f}) → Sat sinyali")

        # SMA20
        if latest_close > sma20:
            score += 1
            details.append(f"✅ Fiyat > SMA20 ({sma20:.2f}) → Kısa vadede yükseliş")
        else:
            details.append(f"❌ Fiyat < SMA20 ({sma20:.2f}) → Kısa vadede düşüş")

        # EMA20
        if latest_close > ema20:
            score += 1
            details.append(f"✅ Fiyat > EMA20 ({ema20:.2f}) → Kısa vadeli trend pozitif")
        else:
            details.append(f"❌ Fiyat < EMA20 ({ema20:.2f}) → Kısa vadeli trend negatif")

        # MFI
        if indicators['MFI'] > 50:
            score += 1
            details.append(f"✅ MFI ({indicators['MFI']:.2f}) → Para girişi var")
        else:
            details.append(f"❌ MFI ({indicators['MFI']:.2f}) → Zayıf para akışı")

        # ADX
        if indicators['ADX'] > 20:
            score += 1
            details.append(f"✅ ADX ({indicators['ADX']:.2f}) → Güçlü trend")
        else:
            details.append(f"❌ ADX ({indicators['ADX']:.2f}) → Zayıf trend")

        # CCI
        if indicators['CCI'] > 0:
            score += 1
            details.append(f"✅ CCI ({indicators['CCI']:.2f}) → Alım baskısı")
        else:
            details.append(f"❌ CCI ({indicators['CCI']:.2f}) → Satım baskısı")

        # STOCH
        if indicators['STOCH'] > 50:
            score += 1
            details.append(f"✅ Stokastik (%K: {indicators['STOCH']:.2f}) → Al sinyali")
        else:
            details.append(f"❌ Stokastik (%K: {indicators['STOCH']:.2f}) → Sat sinyali")

        # Williams %R
        if indicators['WILLR'] > -80:
            score += 1
            details.append(f"✅ Williams %R ({indicators['WILLR']:.2f}) → Güçlü momentum")
        else:
            details.append(f"❌ Williams %R ({indicators['WILLR']:.2f}) → Aşırı satımda")

        # OBV yorumu
        if df['OBV'].iloc[-1] > df['OBV'].iloc[-10]:
            score += 1
            details.append(f"✅ OBV yükseliyor → Hacim destekli yükseliş")
        else:
            details.append(f"❌ OBV düşüyor → Hacim desteği zayıf")

        # Genel sinyal
        if score >= 8:
            signal = "🔼 GÜÇLÜ AL"
        elif score >= 6:
            signal = "⚠️ AL Sinyali"
        else:
            signal = "🔽 NÖTR / ZAYIF"

        return {
            "Hisse": symbol,
            "Fiyat": round(latest_close, 2),
            "Puan": score,
            "Sinyal": signal,
            "Detaylar": details
        }, None

    except Exception as e:
        return None, f"{symbol} için analiz hatası: {e}"

if symbol:
    st.info(f"📈 {symbol} için analiz başlatılıyor...")
    result, error = analyze(symbol)

    if error:
        st.error(error)
    elif result:
        st.success(f"✅ {result['Hisse']} Analiz Sonucu")
        st.markdown(f"- **Fiyat:** {result['Fiyat']} ₺")
        st.markdown(f"- **Puan:** {result['Puan']} / 10")
        st.markdown(f"- **Sinyal:** {result['Sinyal']}")
        st.markdown("### 📋 Detaylı Göstergeler")
        for detail in result['Detaylar']:
            st.markdown(f"- {detail}")
