import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BIST100 Teknik Analiz + RF Tahmin", layout="centered")

# -------------------------
# Veri Çekme
# -------------------------
@st.cache_data(ttl=3600)
def get_data(symbol, period="2y", interval="1d"):
    try:
        df = yf.download(f"{symbol}.IS", period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return None

# -------------------------
# Teknik İndikatör ve Özellik Mühendisliği
# -------------------------
def calculate_features(df):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    df_feat = pd.DataFrame(index=df.index)
    df_feat['Close'] = close
    df_feat['Return'] = close.pct_change()
    df_feat['High_Low'] = (high - low)/close
    df_feat['Close_Open'] = (close - df['Open'])/close
    # Trend
    df_feat['SMA20'] = close.rolling(20).mean()
    df_feat['EMA20'] = close.ewm(span=20, adjust=False).mean()
    # Momentum
    df_feat['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df_feat['MACD'] = macd.macd()
    df_feat['MACD_SIGNAL'] = macd.macd_signal()
    df_feat['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    # Hacim
    df_feat['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    df_feat['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    # Volatilite
    df_feat['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df_feat = df_feat.dropna()
    return df_feat

# -------------------------
# İndikatör Yorumlama
# -------------------------
def interpret_indicators(latest_ind):
    comments = {}
    comments['SMA20'] = "Yükseliş" if latest_ind['Close'] > latest_ind['SMA20'] else "Düşüş"
    comments['EMA20'] = "Yükseliş" if latest_ind['Close'] > latest_ind['EMA20'] else "Düşüş"
    comments['RSI'] = "Aşırı alım" if latest_ind['RSI'] > 70 else ("Aşırı satım" if latest_ind['RSI'] < 30 else "Nötr")
    comments['MACD'] = "Al" if latest_ind['MACD'] > latest_ind['MACD_SIGNAL'] else "Sat"
    comments['ADX'] = "Trend güçlü" if latest_ind['ADX'] > 25 else "Trend zayıf"
    comments['MFI'] = "Para girişi güçlü" if latest_ind['MFI'] > 50 else "Para çıkışı baskın"
    comments['OBV'] = "Hacim destekliyor" if latest_ind['OBV'] > 0 else "Hacim zayıf"
    comments['ATR'] = "Volatilite yüksek" if latest_ind['ATR'] > 0.02*latest_ind['Close'] else "Volatilite normal"
    return comments

# -------------------------
# Random Forest Tahmini
# -------------------------
def rf_predict(df_feat):
    features = df_feat.columns.tolist()
    X = df_feat[features].values[:-1]
    y = df_feat['Close'].values[1:]
    if len(X) < 50:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_scaled[-1].reshape(1,-1))[0]
    return pred

# -------------------------
# Streamlit Arayüz
# -------------------------
st.title("📊 BIST100 Teknik Analiz + Random Forest Tahmin")

symbol = st.text_input("🔎 Hisse Kodu", value="AEFES").upper()
period = st.selectbox("Dönem", ["6mo","1y","2y"], index=1)
interval = st.selectbox("Zaman Aralığı", ["1d","1h"], index=0)

if symbol:
    df = get_data(symbol, period=period, interval=interval)
    if df is None or len(df) < 30:
        st.warning("Yeterli veri yok veya veri çekme hatası oluştu.")
    else:
        df_feat = calculate_features(df)
        latest_ind = df_feat.iloc[-1]
        comments = interpret_indicators(latest_ind)
        close_price = latest_ind['Close']

        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        st.markdown("### 🔎 İndikatörler ve Yorumlar")
        result_df = pd.DataFrame([(k, f"{latest_ind[k]:.2f}", comments.get(k,"")) for k in latest_ind.index],
                                 columns=["İndikatör","Değer","Yorum"])
        st.dataframe(result_df, use_container_width=True)

        # RF Tahmini
        rf_pred = rf_predict(df_feat)
        st.markdown("### 📊 Tahmin")
        st.write(f"- **Random Forest Tahmini (1 gün sonrası):** {rf_pred:.2f} ₺" if rf_pred else "- Tahmin için veri yetersiz")

        # Ensemble Tavsiye (sadece RF + indikatör)
        ensemble_score = 0
        if rf_pred:
            ensemble_score += 1 if rf_pred > close_price else -1
        # İndikatör ağırlığı
        if latest_ind['RSI'] < 30: ensemble_score +=1
        elif latest_ind['RSI'] >70: ensemble_score -=1
        if latest_ind['MACD'] > latest_ind['MACD_SIGNAL']: ensemble_score +=1
        else: ensemble_score -=1
        if latest_ind['ADX'] >25: ensemble_score +=1
        else: ensemble_score -=1

        st.markdown("### 📢 Al/Sat Tavsiyesi")
        if ensemble_score > 0:
            st.write(f"- **Tavsiyesi:** AL  (Skor: {ensemble_score})")
        elif ensemble_score < 0:
            st.write(f"- **Tavsiyesi:** SAT (Skor: {ensemble_score})")
        else:
            st.write(f"- **Tavsiyesi:** NÖTR (Skor: {ensemble_score})")
