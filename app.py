import streamlit as st
import yfinance as yf
import pandas as pd
import ta  # pip install ta

st.set_page_config(page_title="BIST100 Teknik Analiz (Detaylı)", layout="centered")

@st.cache_data(ttl=3600)  # 1 saat önbellek, istersen artırabilirsin
def get_data(symbol):
    try:
        df = yf.download(f"{symbol}.IS", period="3mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        if len(df) < 30:
            return None, "Yeterli veri yok (minimum 30 satır gerekli)."
        return df, None
    except Exception as e:
        return None, f"Veri indirme hatası: {e}"

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

    # Volume: MFI
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1]

    # Trend Strength: ADX
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

    # Oscillators: Stochastic Oscillator
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

    # OBV (On Balance Volume)
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    inds['OBV'] = obv.on_balance_volume().iloc[-1]

    return inds

def generate_signals(inds):
    signals = []

    # RSI Sinyalleri
    if inds['RSI'] > 70:
        signals.append("RSI aşırı alım → Satış sinyali")
    elif inds['RSI'] < 30:
        signals.append("RSI aşırı satım → Al sinyali")
    else:
        signals.append("RSI nötr")

    # MACD Sinyalleri
    if inds['MACD'] > inds['MACD_SIGNAL']:
        signals.append("MACD çizgisi sinyal çizgisinin üstünde → Al sinyali")
    else:
        signals.append("MACD çizgisi sinyal çizgisinin altında → Sat sinyali")

    # ADX Trend Gücü
    if inds['ADX'] > 25:
        signals.append("ADX > 25 → Trend güçlü")
    else:
        signals.append("ADX ≤ 25 → Trend zayıf")

    return signals

def analyze_trend_momentum(inds, close_price):
    trend = "Yukarı" if close_price > inds['EMA20'] and close_price > inds['SMA20'] else "Aşağı"
    trend_strength = "Güçlü" if inds['ADX'] > 25 else "Zayıf"
    momentum = "Pozitif" if inds['RSI'] > 50 and inds['MACD'] > inds['MACD_SIGNAL'] else "Negatif"

    # Basit hedef fiyat tahmini: son 5 gün % değişim ortalaması ile 5 gün sonrası tahmini
    pct_changes = inds.get('pct_changes', None)
    if pct_changes is not None:
        avg_change = pct_changes
        target_price = close_price * (1 + avg_change * 5)
    else:
        target_price = None

    return trend, trend_strength, momentum, target_price

st.title("📊 BIST100 Teknik Analiz (Detaylı İndikatörlerle)")

# Örnek BIST100 sembol listesi (özellikle kısa tutuldu)
bist100_symbols = [
    "AEFES", "AKBNK", "AKSA", "ALARK", "ARCLK", "ASELS", "BIMAS", "EKGYO", "ENKAI",
    "FROTO", "GARAN", "GUBRF", "HALKB", "ISCTR", "KCHOL", "KOZAL", "KRDMD", "PETKM",
    "PGSUS", "SAHOL", "SISE", "TAVHL", "TCELL", "THYAO", "TOASO", "TTKOM", "TUPRS", "VAKBN", "YKBNK"
]

symbol = st.selectbox("🔎 Hisse kodunu seçin", options=bist100_symbols, index=0)

if symbol:
    st.write(f"📈 {symbol} için analiz başlatılıyor...")
    df, error = get_data(symbol)

    if error:
        st.error(f"{symbol}: {error}")
    else:
        try:
            inds = calculate_indicators(df)
            close_price = df['Close'].iloc[-1]

            # Basit hedef fiyat için günlük kapanışların ortalama % değişimini hesapla
            pct_change = df['Close'].pct_change().tail(5).mean()
            inds['pct_changes'] = pct_change

            st.markdown(f"### {symbol} Analiz Sonucu")
            st.write(f"- **Son Kapanış Fiyatı:** {close_price:.2f} ₺")

            # İndikatör değerleri detaylı liste
            st.markdown("#### Teknik İndikatörler ve Osilatörler")
            for k, v in inds.items():
                if isinstance(v, float):
                    st.write(f"- {k}: {v:.2f}")
                else:
                    st.write(f"- {k}: {v}")

            # Sinyalleri göster
            signals = generate_signals(inds)
            st.markdown("#### 📢 Al/Sat/Nötr Sinyalleri:")
            for s in signals:
                st.write(f"- {s}")

            # Trend, momentum ve hedef fiyat
            trend, trend_strength, momentum, target_price = analyze_trend_momentum(inds, close_price)

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
