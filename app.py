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
    "GARAN", "GSRAY", "KCAER", "KCHOL", "KONTR", "KOZAA", "KOZAL", "KRDMD", "LIDER", "MAGEN",
    "MAVI", "MGROS", "OYAKC", "ODAS", "OTKAR", "PGSUS", "PETKM", "QUAGR", "REEDR", "SASA",
    "SAYAS", "SDTTR", "SMRTG", "SISE", "SKBNK", "SOKM", "SELEC", "TAVHL", "TCELL", "THYAO",
    "TMSN", "TKFEN", "TOASO", "TSPOR", "TTKOM", "TTRAK", "TUKAS", "TUPRS", "TURSG", "ULKER",
    "VAKBN", "VESTL", "YEOTK", "YKBNK"
]

@st.cache_data(show_spinner=True)
def analyze_stock(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        if df.empty:
            st.write(f"{symbol} için veri bulunamadı.")
            return None
        df.dropna(inplace=True)

        # Teknik göstergeler
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd_diff()
        df['SMA20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['EMA20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['MFI'] = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).money_flow_index()
        df['ADX'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        df['CCI'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
        df['STOCH'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
        df['WILLR'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()

        latest = df.iloc[-1]

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
        score += df['OBV'].iloc[-1] > df['OBV'].iloc[-10]

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
