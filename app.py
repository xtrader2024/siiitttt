import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(page_title="BIST100 Hisse Tekn. Analiz (Adım Adım)", layout="wide")
st.title("📊 BIST100 Hisse Tekn. Analiz (Adım Adım)")

# Hisse listesi
symbols = [
    "AEFES", "AGHOL", "AGROT", "AKBNK", "AKFYE", "AKFGY", "AKSA", "AKSEN", "ALARK", "ALFAS",
    "ALTNY", "ANHYT", "ANSGR", "ARCLK", "ARDYZ", "ASELS", "ASTOR", "AVPGY", "BERA", "BFREN",
    "BIENY", "BIMAS", "BSOKE", "BTCIM", "CANTE", "CCOLA", "CIMSA", "CLEBI", "CVKMD", "DOAS",
    "DOHOL", "ECILC", "ECZYT", "EGEEN", "EKGYO", "ENERY", "ENJSA", "ENKAI", "EREGL", "FROTO",
    "GARAN", "GSRAY", "KCAER", "KCHOL", "KONTR", "KOZAA", "KOZAL", "KRDMD", "LIDER", "MAGEN",
    "MAVI", "MGROS", "OYAKC", "ODAS", "OTKAR", "PGSUS", "PETKM", "QUAGR", "REEDR", "SASA",
    "SAYAS", "SDTTR", "SMRTG", "SISE", "SKBNK", "SOKM", "SELEC", "TAVHL", "TCELL", "THYAO",
    "TMSN", "TKFEN", "TOASO", "TSPOR", "TTKOM", "TTRAK", "TUKAS", "TUPRS", "TURSG", "ULKER",
    "VAKBN", "VESTL", "YEOTK", "YKBNK"
]

# Oturum durumu tanımla
if "index" not in st.session_state:
    st.session_state.index = 0

current_symbol = symbols[st.session_state.index]
st.subheader(f"Analiz ediliyor: {current_symbol}")

# Teknik analiz fonksiyonu
def analyze(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        df.dropna(inplace=True)
        if df.empty:
            return None

        # Göstergeler (manuel hesaplama)
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df['MACD'] = compute_macd(df['Close'])
        df['ADX'] = compute_adx(df)

        latest = df.iloc[-1]
        score = 0
        score += int(latest['RSI'] > 50)
        score += int(latest['MACD'] > 0)
        score += int(latest['Close'] > latest['SMA20'])
        score += int(latest['Close'] > latest['EMA20'])
        score += int(latest['ADX'] > 20)

        signal = "🔼 AL" if score >= 4 else ("⚠️ İzlenebilir" if score == 3 else "🔽 NÖTR")

        return {
            "Hisse": symbol,
            "Fiyat": round(latest['Close'], 2),
            "Puan": score,
            "Sinyal": signal
        }
    except Exception as e:
        st.warning(f"{symbol} için analiz yapılamadı: {e}")
        return None

# RSI hesaplama
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# MACD hesaplama
def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

# ADX hesaplama
def compute_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

# Analizi çalıştır
result = analyze(current_symbol)

if result:
    st.markdown(f"### {result['Hisse']} Analiz Sonucu")
    st.write(f"Fiyat: {result['Fiyat']}")
    st.write(f"Puan: {result['Puan']} / 10")
    st.write(f"Sinyal: {result['Sinyal']}")
else:
    st.error(f"{current_symbol} için analiz yapılamadı.")

# Devam butonu
if st.session_state.index < len(symbols) - 1:
    if st.button("➡️ Devam"):
        st.session_state.index += 1
        st.experimental_rerun()
else:
    st.success("🎉 Tüm hisseler analiz edildi.")
