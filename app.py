import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz", layout="wide")
st.title("📊 BIST100 Hisse Senetleri Teknik Analiz (Adım Adım)")

symbols = [
    "AEFES", "AGHOL", "AGROT", "AKBNK", "AKFYE", "AKFGY", "AKSA", "AKSEN", "ALARK", "ALFAS",
    # ... diğer semboller
]

@st.cache_data(show_spinner=True)
def analyze_stock(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        if df.empty:
            st.write(f"{symbol} için veri bulunamadı.")
            return None
        df.dropna(inplace=True)

        close = df['Close']  # pd.Series, 1 boyutlu kesin!
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Teknik göstergeler - tek boyutlu pd.Series gönderiyoruz
        rsi = ta.momentum.RSIIndicator(close=close).rsi()
        macd_diff = ta.trend.MACD(close=close).macd_diff()
        sma20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        mfi = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume).money_flow_index()
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close).adx()
        obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        cci = ta.trend.CCIIndicator(high=high, low=low, close=close).cci()
        stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch()
        willr = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close).williams_r()

        latest = df.iloc[-1]

        score = 0
        score += rsi.iloc[-1] > 50
        score += macd_diff.iloc[-1] > 0
        score += latest['Close'] > sma20.iloc[-1]
        score += latest['Close'] > ema20.iloc[-1]
        score += mfi.iloc[-1] > 50
        score += adx.iloc[-1] > 20
        score += cci.iloc[-1] > 0
        score += stoch.iloc[-1] > 50
        score += willr.iloc[-1] > -80
        score += obv.iloc[-1] > obv.iloc[-10]

        try:
            df_4h = yf.download(f"{symbol}.IS", period="5d", interval="4h", progress=False)
            df_4h.dropna(inplace=True)
            avg_change = df_4h['Close'].pct_change().mean()
            target_price = latest['Close'] * (1 + avg_change * 4)
        except:
            target_price = latest['Close']

        signal = "🔼 AL" if score >= 7 else ("⚠️ İzlenebilir" if score == 6 else "🔽 NÖTR")

        return {
            "Hisse": symbol,
            "Fiyat": round(latest['Close'], 2),
            "Puan": score,
            "Sinyal": signal,
            "Hedef Fiyat (4h)": round(target_price, 2)
        }

    except Exception as e:
        st.write(f"{symbol} için analiz yapılamadı: {e}")
        return None

results = []
for sym in symbols:
    st.write(f"Analiz ediliyor: {sym}")
    result = analyze_stock(sym)
    if result:
        results.append(result)

if results:
    df_results = pd.DataFrame(results)
    df_filtered = df_results[df_results['Puan'] >= 7]

    st.subheader("🔍 Güçlü Al Sinyali Veren Hisseler (Puan ≥ 7)")
    st.dataframe(df_filtered.sort_values(by='Puan', ascending=False), use_container_width=True)

    st.subheader("📋 Tüm Sonuçlar")
    st.dataframe(df_results.sort_values(by='Puan', ascending=False), use_container_width=True)
else:
    st.warning("❌ Hiçbir hisse için analiz sonucu alınamadı. Veri kaynaklarını veya internet bağlantınızı kontrol edin.")
