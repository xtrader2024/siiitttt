import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BIST100 15 İndikatör + RF Tahmin", layout="centered")

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
# 15 İndikatör Hesaplama
# -------------------------
def calculate_features(df):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    df_feat = pd.DataFrame(index=df.index)
    df_feat['Close'] = close

    # 1. SMA20
    df_feat['SMA20'] = close.rolling(20).mean()
    # 2. EMA20
    df_feat['EMA20'] = close.ewm(span=20, adjust=False).mean()
    # 3. RSI
    df_feat['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    # 4. MACD
    macd = ta.trend.MACD(close)
    df_feat['MACD'] = macd.macd()
    df_feat['MACD_SIGNAL'] = macd.macd_signal()
    # 5. ADX
    df_feat['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    # 6. CCI
    df_feat['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    # 7. MFI
    df_feat['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    # 8. OBV
    df_feat['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    # 9. ATR
    df_feat['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    # 10. Bollinger Band üst ve alt
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df_feat['BB_HIGH'] = bb.bollinger_hband()
    df_feat['BB_LOW'] = bb.bollinger_lband()
    # 11. Stochastic K
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df_feat['STOCH_K'] = stoch.stoch()
    # 12. Stochastic D
    df_feat['STOCH_D'] = stoch.stoch_signal()
    # 13. Williams %R
    willr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    df_feat['WILLR'] = willr.williams_r()
    # 14. ROC
    df_feat['ROC'] = ta.momentum.ROCIndicator(close, window=12).roc()
    # 15. TRIX
    df_feat['TRIX'] = ta.trend.TRIXIndicator(close, window=15).trix()

    df_feat = df_feat.dropna()
    return df_feat

# -------------------------
# İndikatör Yorumlama Al/Sat
# -------------------------
def interpret_indicators(latest):
    comments = {}
    comments['SMA20'] = "Al" if latest['Close'] > latest['SMA20'] else "Sat"
    comments['EMA20'] = "Al" if latest['Close'] > latest['EMA20'] else "Sat"
    comments['RSI'] = "Al" if latest['RSI'] < 30 else ("Sat" if latest['RSI'] > 70 else "Nötr")
    comments['MACD'] = "Al" if latest['MACD'] > latest['MACD_SIGNAL'] else "Sat"
    comments['ADX'] = "Al" if latest['ADX'] > 25 else "Sat"
    comments['CCI'] = "Al" if latest['CCI'] < -100 else ("Sat" if latest['CCI'] > 100 else "Nötr")
    comments['MFI'] = "Al" if latest['MFI'] < 20 else ("Sat" if latest['MFI'] > 80 else "Nötr")
    comments['OBV'] = "Al" if latest['OBV'] > 0 else "Sat"
    comments['ATR'] = "Al" if latest['ATR'] < 0.02*latest['Close'] else "Sat"
    comments['BB_HIGH'] = "Sat" if latest['Close'] >= latest['BB_HIGH'] else "Al" if latest['Close'] <= latest['BB_LOW'] else "Nötr"
    comments['BB_LOW'] = "Al" if latest['Close'] <= latest['BB_LOW'] else "Sat"
    comments['STOCH_K'] = "Al" if latest['STOCH_K'] < 20 else ("Sat" if latest['STOCH_K'] > 80 else "Nötr")
    comments['STOCH_D'] = "Al" if latest['STOCH_D'] < 20 else ("Sat" if latest['STOCH_D'] > 80 else "Nötr")
    comments['WILLR'] = "Al" if latest['WILLR'] < -80 else ("Sat" if latest['WILLR'] > -20 else "Nötr")
    comments['ROC'] = "Al" if latest['ROC'] > 0 else "Sat"
    comments['TRIX'] = "Al" if latest['TRIX'] > 0 else "Sat"
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
st.title("📊 BIST100 15 İndikatör + RF Tahmin")

symbol = st.text_input("🔎 Hisse Kodu", value="AEFES").upper()
period = st.selectbox("Dönem", ["6mo","1y","2y"], index=1)
interval = st.selectbox("Zaman Aralığı", ["1d","1h"], index=0)

if symbol:
    df = get_data(symbol, period=period, interval=interval)
    if df is None or len(df) < 50:
        st.warning("Yeterli veri yok veya veri çekme hatası oluştu.")
    else:
        df_feat = calculate_features(df)
        latest = df_feat.iloc[-1]
        comments = interpret_indicators(latest)
        close_price = latest['Close']

        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        st.markdown("### 🔎 15 İndikatör ve Al/Sat Yorumları")
        result_df = pd.DataFrame([(k, f"{latest[k]:.2f}", comments.get(k,"")) for k in latest.index],
                                 columns=["İndikatör","Değer","Yorum"])
        st.dataframe(result_df, use_container_width=True)

        # RF Tahmini
        rf_pred = rf_predict(df_feat)
        st.markdown("### 📊 Random Forest Tahmini (1 gün sonrası)")
        st.write(f"- Tahmin: {rf_pred:.2f} ₺" if rf_pred else "- Veri yetersiz")

        # Ensemble Tavsiye
        ensemble_score = 0
        if rf_pred:
            ensemble_score += 1 if rf_pred > close_price else -1
        # İndikatör ağırlığı
        for val in comments.values():
            if val=="Al": ensemble_score +=1
            elif val=="Sat": ensemble_score -=1

        st.markdown("### 📢 Ensemble Al/Sat Tavsiyesi")
        if ensemble_score > 0:
            st.write(f"- **Tavsiyesi:** AL  (Skor: {ensemble_score})")
        elif ensemble_score < 0:
            st.write(f"- **Tavsiyesi:** SAT (Skor: {ensemble_score})")
        else:
            st.write(f"- **Tavsiyesi:** NÖTR (Skor: {ensemble_score})")
