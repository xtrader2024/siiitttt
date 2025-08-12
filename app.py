import streamlit as st
import yfinance as yf
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz", layout="wide")
st.title("📊 BIST100 Hisse Tekn. Analiz (Adım Adım)")

symbols = ["AEFES", "AGHOL", "AGROT", "AKBNK", "AKFYE"]  # Kısa liste ile test et

# Oturumda hangi hisseyi analiz ettiğimizi tutuyoruz
if 'index' not in st.session_state:
    st.session_state.index = 0

def analyze_stock(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        if df.empty:
            return None, f"{symbol}: veri bulunamadı."

        df.dropna(inplace=True)
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
        macd = ta.trend.MACD(close).macd_diff().iloc[-1]
        sma20 = ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1]
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
        mfi = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index().iloc[-1]
        adx = ta.trend.ADXIndicator(high, low, close).adx().iloc[-1]
        cci = ta.trend.CCIIndicator(high, low, close).cci().iloc[-1]
        stoch = ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1]
        willr = ta.momentum.WilliamsRIndicator(high, low, close).williams_r().iloc[-1]
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        obv_last = obv.iloc[-1]
        obv_10ago = obv.iloc[-11] if len(obv) > 10 else obv.iloc[0]

        score = sum([
            rsi > 50,
            macd > 0,
            close.iloc[-1] > sma20,
            close.iloc[-1] > ema20,
            mfi > 50,
            adx > 20,
            cci > 0,
            stoch > 50,
            willr > -80,
            obv_last > obv_10ago
        ])

        signal = "🔼 AL" if score >= 7 else "⚠️ İzlenebilir" if score == 6 else "🔽 NÖTR"
        result = f"""
        ### {symbol} Analiz Sonucu
        - Fiyat: {round(close.iloc[-1], 2)}
        - Puan: {score} / 10
        - Sinyal: {signal}
        """
        return result, None

    except Exception as e:
        return None, f"{symbol}: analiz yapılamadı ({e})"

current_symbol = symbols[st.session_state.index]
res, err = analyze_stock(current_symbol)

if err:
    st.error(err)
else:
    st.markdown(res)

# Sadece "Devam" butonu aktifken st.experimental_rerun() çağrısını yapıyoruz
if st.session_state.index < len(symbols) - 1:
    if st.button("➡️ Devam"):
        st.session_state.index += 1
        st.experimental_rerun()
else:
    st.success("✅ Tüm hisseler analiz edildi.")
