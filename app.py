import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz", layout="wide")
st.title("📊 BIST100 Hisse Senetleri Teknik Analiz (Adım Adım)")

symbols = [
    "AEFES", "AGHOL", "AGROT", "AKBNK", "AKFYE", "AKFGY", "AKSA", "AKSEN", "ALARK", "ALFAS",
    "ALTNY", "ANHYT", "ANSGR", "ARCLK", "ARDYZ", "ASELS", "ASTOR", "AVPGY", "BERA", "BFREN",
    "BIENY", "BIMAS", "BSOKE", "BTCIM", "CANTE", "CCOLA", "CIMSA", "CLEBI", "CVKMD", "DOAS",
    "DOHOL", "ECILC", "ECZYT", "EGEEN", "EKGYO", "ENERY", "ENJSA", "ENKAI", "EREGL", "FROTO",
    "GARAN", "GSRAY", "KCAER", "KCHOL", "KONTR", "KOZAA", "KOZAL", "KRDMD", "LIDER"
]

# Başlangıçta index tanımla
if 'index' not in st.session_state:
    st.session_state.index = 0

# Teknik analiz fonksiyonu
def analyze_stock(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        if df.empty:
            return None, f"{symbol} için veri alınamadı."

        df.dropna(inplace=True)

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close).macd_diff()
        sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        mfi = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index()
        adx = ta.trend.ADXIndicator(high, low, close).adx()
        cci = ta.trend.CCIIndicator(high, low, close).cci()
        stoch = ta.momentum.StochasticOscillator(high, low, close).stoch()
        willr = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        latest = {
            'RSI': rsi.iloc[-1],
            'MACD': macd.iloc[-1],
            'Close': close.iloc[-1],
            'SMA20': sma20.iloc[-1],
            'EMA20': ema20.iloc[-1],
            'MFI': mfi.iloc[-1],
            'ADX': adx.iloc[-1],
            'CCI': cci.iloc[-1],
            'STOCH': stoch.iloc[-1],
            'WILLR': willr.iloc[-1],
            'OBV': obv.iloc[-1],
            'OBV_10ago': obv.iloc[-11] if len(obv) > 10 else obv.iloc[0]
        }

        score = 0
        score += latest['RSI'] > 50
        score += latest['MACD'] > 0
        score += latest['Close'] > latest['SMA20']
        score += latest['Close'] > latest['EMA20']
        score += latest['MFI'] > 50
        score += latest['ADX'] > 20
        score += latest['CCI'] > 0
        score += latest['STOCH'] > 50
        score += latest['WILLR'] > -80
        score += latest['OBV'] > latest['OBV_10ago']

        signal = "🔼 AL" if score >= 7 else ("⚠️ İzlenebilir" if score == 6 else "🔽 NÖTR")

        result = f"""
        ### {symbol} Analiz Sonucu
        - Fiyat: {round(latest['Close'], 2)}
        - Puan: {score} / 10
        - Sinyal: {signal}
        """
        return result, None
    except Exception as e:
        return None, f"{symbol} için analiz yapılamadı: {e}"

# Mevcut hisse
current_symbol = symbols[st.session_state.index]

# Analiz sonucu
result, error = analyze_stock(current_symbol)

# Göster
if error:
    st.error(error)
else:
    st.markdown(result)

# Devam butonu (sadece varsa göster)
if st.session_state.index < len(symbols) - 1:
    if st.button("➡️ Devam"):
        st.session_state.index += 1
        st.experimental_rerun()
else:
    st.success("✅ Tüm hisseler analiz edildi.")
