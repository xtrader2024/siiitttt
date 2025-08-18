import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BIST100 Gelişmiş Teknik Analiz + Tahmin", layout="centered")

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
# Teknik İndikatör Hesaplama
# -------------------------
def calculate_indicators(df):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    inds = {}
    inds['Close'] = close
    inds['SMA20'] = close.rolling(20).mean()
    inds['SMA50'] = close.rolling(50).mean()
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean()
    inds['EMA50'] = close.ewm(span=50, adjust=False).mean()
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    inds['MACD'] = ta.trend.MACD(close).macd()
    inds['MACD_SIGNAL'] = ta.trend.MACD(close).macd_signal()
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    inds['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    inds['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    boll = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    inds['BB_high'] = boll.bollinger_hband()
    inds['BB_low'] = boll.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    inds['STOCH_K'] = stoch.stoch()
    inds['STOCH_D'] = stoch.stoch_signal()
    willr = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14)
    inds['WILLR'] = willr.williams_r()
    cci = ta.trend.CCIIndicator(high, low, close, window=20)
    inds['CCI'] = cci.cci()
    roc = ta.momentum.ROCIndicator(close, window=12)
    inds['ROC'] = roc.roc()
    obv_ema = inds['OBV'].ewm(span=20, adjust=False).mean()
    inds['OBV_EMA'] = obv_ema
    df_ind = pd.concat(inds.values(), axis=1, keys=inds.keys())
    df_ind.dropna(inplace=True)
    return df_ind

# -------------------------
# İndikatör Yorumlama
# -------------------------
def interpret_indicators(latest_ind):
    comments = {}
    comments['SMA20'] = "Al" if latest_ind['Close'] > latest_ind['SMA20'] else "Sat"
    comments['SMA50'] = "Al" if latest_ind['Close'] > latest_ind['SMA50'] else "Sat"
    comments['EMA20'] = "Al" if latest_ind['Close'] > latest_ind['EMA20'] else "Sat"
    comments['EMA50'] = "Al" if latest_ind['Close'] > latest_ind['EMA50'] else "Sat"
    comments['RSI'] = "Al" if latest_ind['RSI'] < 30 else ("Sat" if latest_ind['RSI'] >70 else "Nötr")
    comments['MACD'] = "Al" if latest_ind['MACD'] > latest_ind['MACD_SIGNAL'] else "Sat"
    comments['ADX'] = "Al" if latest_ind['ADX'] > 25 else "Sat"
    comments['MFI'] = "Al" if latest_ind['MFI'] < 30 else ("Sat" if latest_ind['MFI'] > 70 else "Nötr")
    comments['OBV'] = "Al" if latest_ind['OBV'] > latest_ind['OBV_EMA'] else "Sat"
    comments['ATR'] = "Sat" if latest_ind['ATR'] > 0.02*latest_ind['Close'] else "Nötr"
    comments['BB_high'] = "Sat" if latest_ind['Close'] >= latest_ind['BB_high'] else "Nötr"
    comments['BB_low'] = "Al" if latest_ind['Close'] <= latest_ind['BB_low'] else "Nötr"
    comments['STOCH_K'] = "Al" if latest_ind['STOCH_K'] < 20 else ("Sat" if latest_ind['STOCH_K'] > 80 else "Nötr")
    comments['STOCH_D'] = "Al" if latest_ind['STOCH_D'] < 20 else ("Sat" if latest_ind['STOCH_D'] > 80 else "Nötr")
    comments['WILLR'] = "Al" if latest_ind['WILLR'] < -80 else ("Sat" if latest_ind['WILLR'] > -20 else "Nötr")
    comments['CCI'] = "Al" if latest_ind['CCI'] < -100 else ("Sat" if latest_ind['CCI'] > 100 else "Nötr")
    comments['ROC'] = "Al" if latest_ind['ROC'] > 0 else "Sat"
    comments['OBV_EMA'] = "Al" if latest_ind['OBV'] > latest_ind['OBV_EMA'] else "Sat"
    return comments

# -------------------------
# Random Forest Tahmini
# -------------------------
def rf_predict(df_ind):
    features = df_ind.columns.tolist()
    X = df_ind[features].values[:-1]
    y = df_ind['Close'].values[1:]
    if len(X) < 20:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_scaled[-1].reshape(1,-1))[0]
    return pred

# -------------------------
# Streamlit Arayüz
# -------------------------
st.title("📊 BIST100 Teknik Analiz + Random Forest Tahmin")

symbol = st.text_input("🔎 Hisse Kodu", value="AEFES").upper()

if symbol:
    df = get_data(symbol)
    if df is None or len(df) < 30:
        st.warning("Yeterli veri yok veya veri çekme hatası oluştu.")
    else:
        df_ind = calculate_indicators(df)
        latest_ind = df_ind.iloc[-1]
        comments = interpret_indicators(latest_ind)
        close_price = latest_ind['Close']

        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        st.markdown("### 🔎 İndikatörler ve Yorumlar")
        result_df = pd.DataFrame([(k, f"{latest_ind[k]:.2f}", comments.get(k,"")) for k in latest_ind.index],
                                 columns=["İndikatör","Değer","Yorum"])
        st.dataframe(result_df, use_container_width=True)

        # Random Forest Tahmini
        rf_pred = rf_predict(df_ind)
        if rf_pred: 
            st.markdown(f"### 📊 Random Forest Tahmini (1 Gün Sonra): {rf_pred:.2f} ₺")
        else:
            st.markdown("- Random Forest Tahmini: Veri yetersiz")

        # -------------------------
        # İndikatörlere Göre Açıklamalı Genel Yorum
        # -------------------------
        st.markdown("### 📝 İndikatörlere Göre Açıklamalı Genel Yorum")
        explanation = []
        for k, v in comments.items():
            explanation.append(f"- **{k}:** {v}")
        st.write("\n".join(explanation))

        # Genel tavsiye
        total_score = 0
        for val in comments.values():
            if val == "Al": total_score +=1
            elif val == "Sat": total_score -=1

        if total_score > 7:
            st.write(f"💡 **Genel Tavsiye:** AL (İndikatörler çoğunlukla yükselişi işaret ediyor)")
        elif total_score < -7:
            st.write(f"💡 **Genel Tavsiye:** SAT (İndikatörler çoğunlukla düşüşü işaret ediyor)")
        else:
            st.write(f"💡 **Genel Tavsiye:** NÖTR (İndikatörler karışık sinyal veriyor)")
