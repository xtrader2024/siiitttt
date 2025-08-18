import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BIST100 15 İndikatör + 1 Gün RF Tahmin", layout="centered")

# -------------------------
# Veri Çekme
# -------------------------
@st.cache_data(ttl=3600)
def get_data(symbol, period="52d", interval="4h"):
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

    # Trend
    df_feat['SMA20'] = close.rolling(20).mean()
    df_feat['EMA20'] = close.ewm(span=20, adjust=False).mean()
    # Momentum
    df_feat['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df_feat['MACD'] = macd.macd()
    df_feat['MACD_SIGNAL'] = macd.macd_signal()
    df_feat['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    df_feat['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    df_feat['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    df_feat['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df_feat['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df_feat['BB_HIGH'] = bb.bollinger_hband()
    df_feat['BB_LOW'] = bb.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df_feat['STOCH_K'] = stoch.stoch()
    df_feat['STOCH_D'] = stoch.stoch_signal()
    df_feat['WILLR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    df_feat['ROC'] = ta.momentum.ROCIndicator(close, window=12).roc()
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
st.title("📊 BIST100 15 İndikatör + 1 Gün RF Tahmin (4H Mum)")

symbol = st.text_input("🔎 Hisse Kodu", value="AEFES").upper()

if symbol:
    df = get_data(symbol, period="52d", interval="4h")
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
        st.markdown("### 📊 1 Gün Sonrası Fiyat Tahmini")
        if rf_pred:
            st.write(f"- **Random Forest Tahmini:** {rf_pred:.2f} ₺")
        else:
            st.write("- Veri yetersiz")

        # İndikatör ağırlıklı tahmin: son kapanış + indikatör sinyalleri
        ind_score = 0
        for val in comments.values():
            if val=="Al": ind_score +=1
            elif val=="Sat": ind_score -=1
        ind_pred = close_price * (1 + 0.001*ind_score)  # basit ağırlık ile tahmin
        st.write(f"- **İndikatör Ağırlıklı Tahmin:** {ind_pred:.2f} ₺")

        # Ensemble Skor ile Al/Sat Tavsiyesi
        ensemble_score = 0
        if rf_pred and rf_pred > close_price: ensemble_score +=1
        elif rf_pred: ensemble_score -=1
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
