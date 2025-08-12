import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta  

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
        
        # pandas_ta ile teknik göstergeleri hesapla
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD_diff'] = macd['MACDh_12_26_9']
        df['SMA20'] = ta.sma(df['Close'], length=20)
        df['EMA20'] = ta.ema(df['Close'], length=20)
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        df['STOCH'] = stoch['STOCHk_14_3_3']
        df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)

        latest = df.iloc[-1]

        score = 0
        score += int(latest['RSI'] > 50)
        score += int(latest['MACD_diff'] > 0)
        score += int(latest['Close'] > latest['SMA20'])
        score += int(latest['Close'] > latest['EMA20'])
        score += int(latest['MFI'] > 50)
        score += int(latest['ADX'] > 20)
        score += int(latest['CCI'] > 0)
        score += int(latest['STOCH'] > 50)
        score += int(latest['WILLR'] > -80)
        
        if len(df['OBV']) >= 10:
            obv_last = df['OBV'].iloc[-1]
            obv_10_before = df['OBV'].iloc[-10]
            score += int(obv_last > obv_10_before)
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
        st.error(f"{symbol} için veri alınamadı veya analiz yapılamadı: {e}")
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
