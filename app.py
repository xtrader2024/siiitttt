import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Elle Giriş)", layout="centered")
st.title("📊 BIST100 Teknik Analiz (Elle Giriş)")

# Hisse kodu giriş alanı
symbol_input = st.text_input("Lütfen analiz etmek istediğiniz hisse kodunu girin (örn: AEFES):").upper()

def analyze(symbol):
    try:
        df = yf.download(
            f"{symbol}.IS",
            period="7d",
            interval="1h",
            progress=False,
            multi_level_index=False  # Düzgün çalışması için önemli
        )

        if df.empty:
            return None, f"{symbol}: veri alınamadı veya hisse kodu hatalı."

        df.dropna(inplace=True)

        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']

        df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        df['MACD'] = ta.trend.MACD(close=close).macd_diff()
        df['SMA20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        df['EMA20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()

        latest = df.iloc[-1]
        score = sum([
            latest['RSI'] > 50,
            latest['MACD'] > 0,
            latest['Close'] > latest['SMA20'],
            latest['Close'] > latest['EMA20']
        ])

        signal = "🔼 AL" if score >= 3 else ("⚠️ İzlenebilir" if score == 2 else "🔽 NÖTR")

        return {
            "Hisse": symbol,
            "Fiyat": round(latest['Close'], 2),
            "Puan": score,
            "Sinyal": signal
        }, None

    except Exception as e:
        return None, f"{symbol}: analiz yapılamadı ({e})"

if symbol_input:
    st.write(f"🔍 **{symbol_input}** için analiz yapılıyor...")
    result, err = analyze(symbol_input)

    if err:
        st.error(err)
    elif result:
        st.success(f"📈 {result['Hisse']} Analiz Sonucu")
        st.markdown(f"- **Fiyat:** {result['Fiyat']}")
        st.markdown(f"- **Puan:** {result['Puan']} / 4")
        st.markdown(f"- **Sinyal:** {result['Sinyal']}")
