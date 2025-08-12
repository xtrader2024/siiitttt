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

def analyze_stock(symbol):
    st.write(f"Analiz ediliyor: {symbol}")
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        if df.empty:
            st.write(f"{symbol} için veri bulunamadı.")
            return None

        df.dropna(inplace=True)

        # Sütunları 1-boyutlu hale getir (squeeze)
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        # Teknik Göstergeler
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

        # OBV için 10 bar öncesi değer
        obv_10_ago = obv.iloc[latest_idx - 10] if len(obv) > 10 else obv.iloc[0]

        # Skor hesapla (0–10)
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

        # 4 saatlik grafikten hedef fiyat tahmini
        try:
            df_4h = yf.download(f"{symbol}.IS", period="5d", interval="4h", progress=False)
            df_4h.dropna(inplace=True)
            close_4h = df_4h['Close'].squeeze()
            avg_change = close_4h.pct_change().mean()
            target_price = latest['Close'] * (1 + avg_change * 4)
        except Exception:
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
    res = analyze_stock(sym)
    if res:
        results.append(res)

df_results = pd.DataFrame(results)

if not df_results.empty:
    df_filtered = df_results[df_results['Puan'] >= 7]

    st.subheader("🔍 Güçlü Al Sinyali Veren Hisseler (Puan ≥ 7)")
    st.dataframe(df_filtered.sort_values(by='Puan', ascending=False), use_container_width=True)

    st.subheader("📋 Tüm Sonuçlar")
    st.dataframe(df_results.sort_values(by='Puan', ascending=False), use_container_width=True)
else:
    st.warning("❌ Hiçbir hisse için analiz sonucu alınamadı. Veri kaynaklarını veya internet bağlantınızı kontrol edin.")
