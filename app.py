import streamlit as st
import yfinance as yf
import pandas as pd
import ta  # pip install ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Detaylı)", layout="centered")

bist100_symbols = [
    "AKBNK", "ASELS", "BIMAS", "FROTO", "GARAN", "KCHOL", "KRDMD", "PETKM", "SISE",
    "THYAO", "TUPRS", "VAKBN", "YKBNK", "ALARK", "ARCLK", "BANVT", "BRISA", "CCOLA",
    "COCA", "DOHOL", "DOAS", "EGEEN", "EREGL", "FENER", "GUBRF", "HEKTS",
    "ISCTR", "ISGYO", "KARSN", "KOZAA", "KOZAL", "KONYA", "KORDS", "MGROS", "MPARK",
    "NTHOL", "OZKGY", "PGSUS", "SAHOL", "SELEC", "SKBNK", "SOKM", "SRVGY",
    "SODA", "TAVHL", "TCELL", "TMSN", "TOASO", "TTRAK", "ULKER", "VESTL",
    "YATAS", "YKGYO", "ZOREN", "AYGAZ", "BRSAN", "EPCIS", "MAALT", "NETAS",
    "TIRE", "TRKCM", "TRGYO", "TSKB", "TKFEN", "GUBRF", "AKSEN", "AFYON",
    "ARDYZ", "IZMDC", "OTKAR", "AGHOL", "ANACM", "AKSA"
]

def get_data(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        if len(df) < 30:
            st.warning(f"{symbol}: Yeterli veri yok (minimum 30 satır gerekli). Toplam veri: {len(df)}")
            return None
        return df
    except Exception as e:
        st.error(f"Veri indirme hatası: {e}")
        return None

def calculate_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    inds = {}

    inds['SMA20'] = close.rolling(window=20).mean().iloc[-1]
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean().iloc[-1]

    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1]

    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]

    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    inds['STOCH_K'] = stoch.stoch().iloc[-1]
    inds['STOCH_D'] = stoch.stoch_signal().iloc[-1]

    macd = ta.trend.MACD(close)
    inds['MACD'] = macd.macd().iloc[-1]
    inds['MACD_SIGNAL'] = macd.macd_signal().iloc[-1]

    willr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    inds['WILLR'] = willr.williams_r().iloc[-1]

    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    inds['OBV'] = obv.on_balance_volume().iloc[-1]

    return inds

def analyze_trend_momentum(inds, close_price, symbol):
    trend = "Yukarı" if close_price > inds['EMA20'] and close_price > inds['SMA20'] else "Aşağı"
    trend_strength = "Güçlü" if inds['ADX'] > 25 else "Zayıf"
    momentum = "Pozitif" if inds['RSI'] > 50 and inds['MACD'] > inds['MACD_SIGNAL'] else "Negatif"

    try:
        df_4h = yf.download(f"{symbol}.IS", period="7d", interval="4h", progress=False)
        if isinstance(df_4h.columns, pd.MultiIndex):
            df_4h.columns = df_4h.columns.droplevel(1)
        df_4h.dropna(inplace=True)
        avg_change = df_4h['Close'].pct_change().mean()
        target_price = close_price * (1 + avg_change * 5)
    except Exception:
        target_price = None

    return trend, trend_strength, momentum, target_price

st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

symbol = st.selectbox("🔎 Hisse seçin (örn: ASELS)", bist100_symbols)

if symbol:
    st.write(f"📈 {symbol} için analiz başlatılıyor...")
    df = get_data(symbol)

    if df is None or df.empty:
        st.error(f"{symbol} için veri alınamadı veya veri eksik.")
    else:
        try:
            inds = calculate_indicators(df)
            close_price = df['Close'].iloc[-1]

            st.markdown(f"### {symbol} Analiz Sonucu")
            st.write(f"- **Son Kapanış Fiyatı:** {close_price:.2f} ₺")

            st.markdown("#### Teknik İndikatörler ve Osilatörler")
            for k, v in inds.items():
                if isinstance(v, float):
                    st.write(f"- {k}: {v:.2f}")
                else:
                    st.write(f"- {k}: {v}")

            trend, trend_strength, momentum, target_price = analyze_trend_momentum(inds, close_price, symbol)

            st.markdown("#### 📊 Genel Teknik Yorum:")
            st.write(f"- **Trend Yönü:** {trend}")
            st.write(f"- **Trend Gücü (ADX):** {trend_strength}")
            st.write(f"- **Momentum (RSI + MACD):** {momentum}")
            if target_price:
                st.write(f"- **Tahmini Hedef Fiyat (5 gün sonrası):** {target_price:.2f} ₺")
            else:
                st.write("- **Tahmini Hedef Fiyat:** Hesaplanamadı")

        except Exception as e:
            st.error(f"Analiz sırasında hata oluştu: {e}")
