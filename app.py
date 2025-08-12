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

if "stock_index" not in st.session_state:
    st.session_state.stock_index = 0

def analyze_stock(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="7d", interval="1h", progress=False)
        df.dropna(inplace=True)

        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        # İndikatörler
        df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
        df['MACD'] = ta.trend.MACD(close).macd_diff()
        df['SMA20'] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
        df['EMA20'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        df['MFI'] = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index()
        df['ADX'] = ta.trend.ADXIndicator(high, low, close).adx()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df['CCI'] = ta.trend.CCIIndicator(high, low, close).cci()
        df['STOCH'] = ta.momentum.StochasticOscillator(high, low, close).stoch()
        df['WILLR'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

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
        
        # OBV karşılaştırması için yeterli veri olduğundan emin ol
        if len(df['OBV']) >= 10:
            score += df['OBV'].iloc[-1] > df['OBV'].iloc[-10]
        else:
            score += 0

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
            "Hedef Fiyat": round(target_price, 2)
        }

    except Exception as e:
        st.error(f"{symbol} için veri alınamadı: {e}")
        return None

current_symbol = symbols[st.session_state.stock_index]
result = analyze_stock(current_symbol)

if result:
    st.subheader(f"📈 {result['Hisse']} Analizi")
    st.write(f"**Fiyat:** {result['Fiyat']} ₺")
    st.write(f"**Puan:** {result['Puan']} / 10")
    st.write(f"**Sinyal:** {result['Sinyal']}")
    st.write(f"**4 Saatlik Hedef Fiyat:** {result['Hedef Fiyat']} ₺")
else:
    st.warning(f"{current_symbol} için analiz yapılamadı.")

if st.button("➡️ Sonraki Hisse"):
    if st.session_state.stock_index < len(symbols) - 1:
        st.session_state.stock_index += 1
        st.experimental_rerun()
    else:
        st.success("✅ Tüm hisseler analiz edildi.")
