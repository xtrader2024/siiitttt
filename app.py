import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz", layout="wide")
st.title("📊 BIST100 Hisse Senetleri Teknik Analiz (Adım Adım)")

symbols = [
    "AEFES", "AGHOL", "AGROT", "AKBNK", "AKFYE", "AKFGY", "AKSA", "AKSEN", "ALARK", "ALFAS",
    "ALTNY", "ANHYT", "ANSGR", "ARCLK", "ARDYZ", "ASELS", "ASTOR", "AVPGY", "BERA", "BFREN",
    # ... (istenilen diğer hisseler)
]

if 'index' not in st.session_state:
    st.session_state.index = 0

def analyze_stock(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        if df.empty:
            return None, f"{symbol} için veri bulunamadı."
        df.dropna(inplace=True)

        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

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

        latest_idx = -1
        latest = {
            'RSI': rsi.iloc[latest_idx],
            'MACD': macd.iloc[latest_idx],
            'Close': close.iloc[latest_idx],
            'SMA20': sma20.iloc[latest_idx],
            'EMA20': ema20.iloc[latest_idx],
            'MFI': mfi.iloc[latest_idx],
            'ADX': adx.iloc[latest_idx],
            'CCI': cci.iloc[latest_idx],
            'STOCH': stoch.iloc[latest_idx],
            'WILLR': willr.iloc[latest_idx],
            'OBV': obv.iloc[latest_idx]
        }

        obv_10_ago = obv.iloc[latest_idx - 10] if len(obv) > 10 else obv.iloc[0]

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
        score += latest['OBV'] > obv_10_ago

        signal = "🔼 AL" if score >= 7 else ("⚠️ İzlenebilir" if score == 6 else "🔽 NÖTR")

        result_text = f"""
        ### {symbol} Analiz Sonucu
        - Fiyat: {round(latest['Close'], 2)}
        - Puan: {score} / 10
        - Sinyal: {signal}
        """

        return result_text, None

    except Exception as e:
        return None, f"{symbol} için analiz yapılamadı: {e}"

current_symbol = symbols[st.session_state.index]

result, error = analyze_stock(current_symbol)

if error:
    st.error(error)
else:
    st.markdown(result)

if st.session_state.index < len(symbols) - 1:
    if st.button("Devam"):
        st.session_state.index += 1
        st.experimental_rerun()
else:
    st.success("Tüm hisseler analiz edildi.")
