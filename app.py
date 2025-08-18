import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Yorumlu)", layout="centered")

# -------------------------
# Veri Çekme
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
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    inds = {}

    # Trend
    inds['SMA20'] = close.rolling(window=20).mean().iloc[-1]
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean().iloc[-1]

    # Momentum
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1]
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    inds['StochRSI_K'] = stoch_rsi.stochrsi_k().iloc[-1]
    inds['StochRSI_D'] = stoch_rsi.stochrsi_d().iloc[-1]

    # Volume
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]
    inds['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]

    # Trend Strength
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

    # Bollinger
    boll = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    inds['BB_high'] = boll.bollinger_hband().iloc[-1]
    inds['BB_low'] = boll.bollinger_lband().iloc[-1]

    # ATR (Volatilite)
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
    inds['WILLR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]

    return inds

# -------------------------
# İndikatör Yorumları
# -------------------------
def interpret_indicators(inds, close_price, df):
    comments = {}
    comments['SMA20'] = "Yükseliş eğilimi" if close_price > inds['SMA20'] else "Düşüş eğilimi"
    comments['EMA20'] = "Yükseliş eğilimi" if close_price > inds['EMA20'] else "Düşüş eğilimi"

    comments['RSI'] = "Aşırı alım" if inds['RSI'] > 70 else ("Aşırı satım" if inds['RSI'] < 30 else "Nötr")
    comments['CCI'] = "Aşırı alım" if inds['CCI'] > 100 else ("Aşırı satım" if inds['CCI'] < -100 else "Nötr")
    comments['MFI'] = "Para girişi güçlü" if inds['MFI'] > 50 else "Para çıkışı baskın"
    comments['ADX'] = "Trend güçlü" if inds['ADX'] > 25 else "Trend zayıf"
    comments['StochRSI_K'] = "Yüksek momentum" if inds['StochRSI_K'] > 80 else ("Düşük momentum" if inds['StochRSI_K'] < 20 else "Nötr")
    comments['STOCH_K'] = "Aşırı alım" if inds['STOCH_K'] > 80 else ("Aşırı satım" if inds['STOCH_K'] < 20 else "Nötr")
    comments['MACD'] = "Al sinyali" if inds['MACD'] > inds['MACD_SIGNAL'] else "Sat sinyali"
    comments['WILLR'] = "Aşırı satım" if inds['WILLR'] < -80 else ("Aşırı alım" if inds['WILLR'] > -20 else "Nötr")
    comments['ATR'] = "Volatilite yüksek" if inds['ATR'] > df['Close'].pct_change().std()*close_price else "Volatilite normal"
    comments['OBV'] = "Hacim destekliyor" if inds['OBV'] > 0 else "Hacim zayıf"
    comments['BB_high'] = "Üst banda yakın (aşırı alım riski)" if close_price >= inds['BB_high'] else ""
    comments['BB_low'] = "Alt banda yakın (alım fırsatı)" if close_price <= inds['BB_low'] else ""
    return comments

# -------------------------
# Trend ve Tahmin
# -------------------------
def analyze_trend_momentum(inds, close_price, symbol):
    trend = "Yukarı" if close_price > inds['EMA20'] and close_price > inds['SMA20'] else "Aşağı"
    trend_strength = "Güçlü" if inds['ADX'] > 25 else "Zayıf"
    momentum = "Pozitif" if inds['RSI'] > 50 and inds['MACD'] > inds['MACD_SIGNAL'] else "Negatif"

    # Basit 5 günlük fiyat tahmini
    try:
        df_4h = get_data(symbol, period="7d", interval="4h")
        if df_4h is not None and not df_4h.empty:
            avg_change = df_4h['Close'].pct_change().mean()
            target_price = close_price * (1 + avg_change * 5)
        else:
            target_price = None
    except:
        target_price = None

    return trend, trend_strength, momentum, target_price

# -------------------------
# Sinyal ve Skor
# -------------------------
def generate_signals_and_score(inds):
    signals, score = [], 0
    if inds['RSI'] < 30:
        signals.append("RSI aşırı satım → AL")
        score += 4
    elif inds['RSI'] > 70:
        signals.append("RSI aşırı alım → SAT")
    else:
        signals.append("RSI nötr")

    if inds['MACD'] > inds['MACD_SIGNAL']:
        signals.append("MACD yukarı kesmiş → AL")
        score += 3
    else:
        signals.append("MACD aşağı kesmiş → SAT")

    if inds['ADX'] > 25:
        signals.append("Trend güçlü (ADX>25)")
        score += 3
    else:
        signals.append("Trend zayıf (ADX≤25)")

    return signals, score

# -------------------------
# Streamlit Arayüz
# -------------------------
st.sidebar.header("⚙️ Ayarlar")
symbol = st.sidebar.text_input("🔎 Hisse kodu", value="AEFES").upper()
period = st.sidebar.selectbox("Dönem", ["1mo","3mo","6mo","1y"], index=2)
interval = st.sidebar.selectbox("Zaman Aralığı", ["1d","1h","30m"], index=0)

st.title("📊 BIST100 Teknik Analiz (Yorum + Tahmin + Tavsiye)")

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

        # İndikatör Tablosu
        st.markdown("### 🔎 İndikatörler ve Yorumlar")
        result_df = pd.DataFrame(
            [(k, f"{v:.2f}" if isinstance(v,float) else v, comments.get(k,"")) for k,v in inds.items()],
            columns=["İndikatör", "Değer", "Yorum"]
        )
        st.dataframe(result_df, use_container_width=True)

        # Genel Yorum
        trend, trend_strength, momentum, target_price = analyze_trend_momentum(inds, close_price, symbol)
        st.markdown("### 📊 Genel Teknik Yorum")
        st.write(f"- **Trend Yönü:** {trend}")
        st.write(f"- **Trend Gücü (ADX):** {trend_strength}")
        st.write(f"- **Momentum (RSI+MACD):** {momentum}")
        if target_price:
            st.write(f"- **5 Günlük Tahmini Fiyat:** {target_price:.2f} ₺")
        else:
            st.write("- **5 Günlük Tahmini Fiyat:** Hesaplanamadı")

        # Al/Sat Tavsiyeleri
        signals, score = generate_signals_and_score(inds)
        st.markdown("### 📢 Al/Sat Sinyalleri")
        for s in signals:
            st.write(f"- {s}")
        st.markdown(f"### 🟢 Toplam Al Sinyali Puanı: **{score}/10**")
