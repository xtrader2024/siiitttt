import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Adım Adım)", layout="wide")
st.title("📊 BIST100 Teknik Analiz (Adım Adım)")

symbols = ["AEFES", "AGHOL", "AGROT", "AKBNK", "AKFYE"]  # Kısa örnek listesi

if "index" not in st.session_state:
    st.session_state.index = 0

current_symbol = symbols[st.session_state.index]
st.subheader(f"Analiz ediliyor: {current_symbol}")

def analyze(symbol):
    try:
        df = yf.download(f"{symbol}.IS",
                         period="7d", interval="1h",
                         progress=False, multi_index=False)
        if df.empty:
            return None, f"{symbol}: veri alınamadı."
        df.dropna(inplace=True)

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        latest = df.iloc[-1]

        df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        df['MACD'] = ta.trend.MACD(close=close).macd_diff()
        df['SMA20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        df['EMA20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        score = sum([
            df['RSI'].iloc[-1] > 50,
            df['MACD'].iloc[-1] > 0,
            latest['Close'] > df['SMA20'].iloc[-1],
            latest['Close'] > df['EMA20'].iloc[-1]
        ])

        signal = "🔼 AL" if score >= 3 else "⚽ Nötr"

        return {
            "Hisse": symbol,
            "Fiyat": round(latest['Close'], 2),
            "Puan": score,
            "Sinyal": signal
        }, None

    except Exception as e:
        return None, f"{symbol} analizi başarısız: {e}"

result, err = analyze(current_symbol)

if err:
    st.error(err)
elif result:
    st.markdown(f"### {result['Hisse']} Analiz Sonucu")
    st.write(f"Fiyat: {result['Fiyat']}\n\nPuan: {result['Puan']} / 4\n\nSinyal: {result['Sinyal']}")
else:
    st.warning("Beklenmeyen bir hata oluştu.")

if st.session_state.index < len(symbols) - 1:
    if st.button("➡️ Devam"):
        st.session_state.index += 1
        st.experimental_rerun()
else:
    st.success("✅ Tüm hisseler analiz edildi.")
