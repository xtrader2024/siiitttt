import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go

st.set_page_config(page_title="BIST100 Teknik Analiz (Detaylı)", layout="wide")

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

    # Trend: SMA, EMA
    inds['SMA20'] = close.rolling(window=20).mean().iloc[-1]
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean().iloc[-1]

    # Momentum: RSI, CCI
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1]

    # StochRSI
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    inds['StochRSI_K'] = stoch_rsi.stochrsi_k().iloc[-1]
    inds['StochRSI_D'] = stoch_rsi.stochrsi_d().iloc[-1]

    # Volume: MFI
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]

    # Trend Strength: ADX
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

    # Bollinger Bands
    boll = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    inds['BB_high'] = boll.bollinger_hband().iloc[-1]
    inds['BB_low'] = boll.bollinger_lband().iloc[-1]

    # ATR (volatilite ölçümü)
    inds['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    inds['STOCH_K'] = stoch.stoch().iloc[-1]
    inds['STOCH_D'] = stoch.stoch_signal().iloc[-1]

    # MACD
    macd = ta.trend.MACD(close)
    inds['MACD'] = macd.macd().iloc[-1]
    inds['MACD_SIGNAL'] = macd.macd_signal().iloc[-1]

    # Williams %R
    willr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    inds['WILLR'] = willr.williams_r().iloc[-1]

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    inds['OBV'] = obv.on_balance_volume().iloc[-1]

    return inds

# -------------------------
# Trend & Momentum Analizi
# -------------------------
def analyze_trend_momentum(inds, close_price, symbol):
    trend = "Yukarı" if close_price > inds['EMA20'] and close_price > inds['SMA20'] else "Aşağı"
    trend_strength = "Güçlü" if inds['ADX'] > 25 else "Zayıf"
    momentum = "Pozitif" if inds['RSI'] > 50 and inds['MACD'] > inds['MACD_SIGNAL'] else "Negatif"

    return trend, trend_strength, momentum

# -------------------------
# Sinyal ve Skorlama
# -------------------------
def generate_signals_and_score(inds):
    signals = []
    score = 0

    # RSI
    if inds['RSI'] > 70:
        signals.append("RSI aşırı alım → Satış sinyali")
    elif inds['RSI'] < 30:
        signals.append("RSI aşırı satım → Al sinyali")
        score += 10
    else:
        signals.append("RSI nötr")

    # MACD
    if inds['MACD'] > inds['MACD_SIGNAL']:
        signals.append("MACD yukarı kesmiş → Al sinyali")
        score += 15
    else:
        signals.append("MACD aşağı kesmiş → Sat sinyali")

    # ADX
    if inds['ADX'] > 25:
        signals.append("ADX > 25 → Trend güçlü")
        score += 10
    else:
        signals.append("ADX ≤ 25 → Trend zayıf")

    # Bollinger
    if inds['BB_low'] and inds['BB_high']:
        if inds['BB_low'] > 0 and inds['BB_high'] > 0:
            if inds['RSI'] < 30 and inds['StochRSI_K'] < 20:
                signals.append("Bollinger alt bandına yakın → Al fırsatı")
                score += 10

    # MFI
    if inds['MFI'] < 20:
        signals.append("MFI düşük → Al sinyali")
        score += 10
    elif inds['MFI'] > 80:
        signals.append("MFI yüksek → Sat sinyali")

    return signals, min(score, 100)  # max 100

# -------------------------
# Streamlit Arayüz
# -------------------------
st.sidebar.header("⚙️ Ayarlar")
symbol = st.sidebar.text_input("🔎 Hisse kodu", value="AEFES").upper()
period = st.sidebar.selectbox("Dönem", ["1mo","3mo","6mo","1y"], index=2)
interval = st.sidebar.selectbox("Zaman Aralığı", ["1d","1h","30m"], index=0)

st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

if symbol:
    df = get_data(symbol, period=period, interval=interval)
    if df is None or df.empty or len(df) < 30:
        st.error(f"{symbol} için yeterli veri yok.")
    else:
        inds = calculate_indicators(df)
        close_price = df['Close'].iloc[-1]

        # Mum Grafiği
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Mum Grafiği"
        )])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(20).mean(), mode="lines", name="SMA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'].ewm(span=20).mean(), mode="lines", name="EMA20"))
        st.plotly_chart(fig, use_container_width=True)

        # İndikatörler
        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        st.markdown("### 🔎 Teknik İndikatörler")
        inds_df = pd.DataFrame(list(inds.items()), columns=["İndikatör", "Değer"])
        st.dataframe(inds_df)

        # Genel yorum
        trend, trend_strength, momentum = analyze_trend_momentum(inds, close_price, symbol)
        st.markdown("### 📊 Genel Teknik Yorum")
        st.write(f"- **Trend Yönü:** {trend}")
        st.write(f"- **Trend Gücü (ADX):** {trend_strength}")
        st.write(f"- **Momentum:** {momentum}")

        # Sinyaller
        signals, score = generate_signals_and_score(inds)
        st.markdown("### 📢 Sinyaller")
        for s in signals:
            st.write(f"- {s}")

        st.markdown(f"### 🟢 Genel Skor: **{score}/100**")
