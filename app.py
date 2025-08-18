import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz (İndikatör Yorumlu)", layout="centered")

# -------------------------
# Veri Çekme Fonksiyonu
# -------------------------
@st.cache_data(ttl=3600)
def get_data(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(f"{symbol}.IS", period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# -------------------------
# İndikatör Hesaplama
# -------------------------
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
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    inds['StochRSI_K'] = stoch_rsi.stochrsi_k().iloc[-1]
    inds['StochRSI_D'] = stoch_rsi.stochrsi_d().iloc[-1]
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
    boll = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    inds['BB_high'] = boll.bollinger_hband().iloc[-1]
    inds['BB_low'] = boll.bollinger_lband().iloc[-1]
    inds['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
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

# -------------------------
# İndikatör Yorumları
# -------------------------
def interpret_indicators(inds, close_price, df):
    comments = {}

    comments['SMA20'] = "Fiyat üstünde → Yükseliş eğilimi" if close_price > inds['SMA20'] else "Fiyat altında → Düşüş eğilimi"
    comments['EMA20'] = "Fiyat üstünde → Yükseliş eğilimi" if close_price > inds['EMA20'] else "Fiyat altında → Düşüş eğilimi"

    if inds['RSI'] > 70:
        comments['RSI'] = "Aşırı alım (düşüş riski)"
    elif inds['RSI'] < 30:
        comments['RSI'] = "Aşırı satım (yükseliş potansiyeli)"
    else:
        comments['RSI'] = "Nötr"

    if inds['CCI'] > 100:
        comments['CCI'] = "Aşırı alım bölgesi"
    elif inds['CCI'] < -100:
        comments['CCI'] = "Aşırı satım bölgesi"
    else:
        comments['CCI'] = "Nötr"

    comments['MFI'] = "Para girişi güçlü" if inds['MFI'] > 50 else "Para çıkışı baskın"
    comments['ADX'] = "Trend güçlü" if inds['ADX'] > 25 else "Trend zayıf"

    comments['StochRSI_K'] = "Yüksek momentum" if inds['StochRSI_K'] > 80 else ("Düşük momentum" if inds['StochRSI_K'] < 20 else "Nötr")
    comments['STOCH_K'] = "Aşırı alım" if inds['STOCH_K'] > 80 else ("Aşırı satım" if inds['STOCH_K'] < 20 else "Nötr")

    comments['MACD'] = "Al sinyali" if inds['MACD'] > inds['MACD_SIGNAL'] else "Sat sinyali"

    comments['WILLR'] = "Aşırı satım" if inds['WILLR'] < -80 else ("Aşırı alım" if inds['WILLR'] > -20 else "Nötr")

    comments['ATR'] = "Volatilite yüksek" if inds['ATR'] > df['Close'].pct_change().std()*close_price else "Volatilite normal"
    comments['OBV'] = "Hacim destekliyor" if inds['OBV'] > 0 else "Hacim zayıf"

    comments['BB_high'] = "Üst banda yakın → Aşırı alım riski" if close_price >= inds['BB_high'] else ""
    comments['BB_low'] = "Alt banda yakın → Aşırı satım fırsatı" if close_price <= inds['BB_low'] else ""

    return comments

# -------------------------
# Streamlit Arayüz
# -------------------------
st.sidebar.header("⚙️ Ayarlar")
symbol = st.sidebar.text_input("🔎 Hisse kodu", value="AEFES").upper()
period = st.sidebar.selectbox("Dönem", ["1mo","3mo","6mo","1y"], index=2)
interval = st.sidebar.selectbox("Zaman Aralığı", ["1d","1h","30m"], index=0)

st.title("📊 BIST100 Teknik Analiz (İndikatör Yorumlu)")

if symbol:
    df = get_data(symbol, period=period, interval=interval)
    if df is None or df.empty or len(df) < 30:
        st.error(f"{symbol} için yeterli veri yok.")
    else:
        inds = calculate_indicators(df)
        close_price = df['Close'].iloc[-1]
        comments = interpret_indicators(inds, close_price, df)

        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        st.markdown("### 🔎 İndikatörler ve Yorumlar")
        result_df = pd.DataFrame(
            [(k, f"{v:.2f}" if isinstance(v,float) else v, comments.get(k,"")) for k,v in inds.items()],
            columns=["İndikatör", "Değer", "Yorum"]
        )
        st.dataframe(result_df, use_container_width=True)
