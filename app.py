import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BIST100 Teknik Analiz + Tahmin", layout="centered")

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
    inds['Stoch_K'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
    inds['Stoch_D'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch_signal()
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    inds['WilliamsR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    inds['BB_high'] = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_hband()
    inds['BB_low'] = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_lband()
    df_ind = pd.concat(inds.values(), axis=1, keys=inds.keys())
    df_ind.dropna(inplace=True)
    return df_ind

# -------------------------
# İndikatör Yorumlama
# -------------------------
def interpret_indicators(latest_ind, close_price):
    comments = {}
    for ind in latest_ind.index:
        val = latest_ind[ind]
        if ind in ['SMA20','SMA50','EMA20','EMA50']:
            comments[ind] = 'Al' if close_price > val else 'Sat'
        elif ind in ['RSI','MFI']:
            comments[ind] = 'Al' if val < 30 else ('Sat' if val > 70 else 'Nötr')
        elif ind in ['MACD']:
            comments[ind] = 'Al' if val > latest_ind['MACD_SIGNAL'] else 'Sat'
        elif ind in ['ADX']:
            comments[ind] = 'Al' if val > 25 else 'Sat'
        elif ind in ['OBV']:
            comments[ind] = 'Al' if val > 0 else 'Sat'
        elif ind in ['ATR']:
            comments[ind] = 'Al' if val < 0.02*close_price else 'Sat'
        elif ind in ['Stoch_K','Stoch_D']:
            comments[ind] = 'Al' if val < 20 else ('Sat' if val > 80 else 'Nötr')
        elif ind in ['CCI']:
            comments[ind] = 'Al' if val < -100 else ('Sat' if val > 100 else 'Nötr')
        elif ind in ['WilliamsR']:
            comments[ind] = 'Al' if val < -80 else ('Sat' if val > -20 else 'Nötr')
        elif ind in ['BB_high','BB_low']:
            comments[ind] = 'Al' if close_price <= latest_ind['BB_low'] else ('Sat' if close_price >= latest_ind['BB_high'] else 'Nötr')
        else:
            comments[ind] = 'Nötr'
    return comments

# -------------------------
# Random Forest Tahmini
# -------------------------
def rf_predict(df_ind, close_price):
    X = df_ind.values[:-1]
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
period = st.selectbox("Dönem", ["52d"], index=0)
interval = st.selectbox("Zaman Aralığı", ["4h"], index=0)

if symbol:
    df = get_data(symbol, period=period, interval=interval)
    if df is None or len(df) < 30:
        st.warning("Yeterli veri yok veya veri çekme hatası oluştu.")
    else:
        df_ind = calculate_indicators(df)
        latest_ind = df_ind.iloc[-1]
        close_price = df['Close'].iloc[-1]
        comments = interpret_indicators(latest_ind, close_price)

        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        st.markdown("### 🔎 İndikatörler ve Yorumlar")
        result_df = pd.DataFrame([(k, f"{latest_ind[k]:.2f}", comments.get(k,"")) for k in latest_ind.index],
                                 columns=["İndikatör","Değer","Yorum"])
        st.dataframe(result_df, use_container_width=True)

        # 1 Gün Sonrası Fiyat Tahmini
        rf_pred = rf_predict(df_ind, close_price)
        st.markdown("### 📊 1 Gün Sonrası Tahmini Fiyat (Random Forest + İndikatörler)")
        if rf_pred:
            st.write(f"- **Tahmini Fiyat:** {rf_pred:.2f} ₺")
        else:
            st.write("- Tahmin için yeterli veri yok")

        # En alt: Al/Sat tavsiyesi
        score = 0
        for val in comments.values():
            if val == 'Al': score +=1
            elif val == 'Sat': score -=1
        st.markdown("### 📢 Genel Yorum ve Tavsiye")
        if score > 0:
            st.write(f"- **Tavsiyesi:** AL (İndikatörler ağırlıklı)")
        elif score <0:
            st.write(f"- **Tavsiyesi:** SAT (İndikatörler ağırlıklı)")
        else:
            st.write(f"- **Tavsiyesi:** NÖTR (İndikatörler ağırlıklı)")
