import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="BIST100 Gelişmiş Teknik Analiz + RF Tahmin", layout="centered")

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
    # Temel fiyat
    inds['Close'] = close
    # Trend
    inds['SMA20'] = close.rolling(20).mean()
    inds['SMA50'] = close.rolling(50).mean()
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean()
    inds['EMA50'] = close.ewm(span=50, adjust=False).mean()
    # Momentum
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    inds['MACD'] = ta.trend.MACD(close).macd()
    inds['MACD_SIGNAL'] = ta.trend.MACD(close).macd_signal()
    inds['Stoch_K'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
    inds['Stoch_D'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch_signal()
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    inds['WilliamsR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
    # Volatilite
    inds['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    inds['BB_high'] = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_hband()
    inds['BB_low'] = ta.volatility.BollingerBands(close, window=20, window_dev=2).bollinger_lband()
    # Hacim
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    inds['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    # Trend gücü
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    df_ind = pd.concat(inds.values(), axis=1, keys=inds.keys())
    df_ind.dropna(inplace=True)
    return df_ind

# -------------------------
# İndikatör Yorumlama
# -------------------------
def interpret_indicators(latest_ind):
    comments = {}
    comments['SMA20'] = "AL" if latest_ind['Close'] > latest_ind['SMA20'] else "SAT"
    comments['SMA50'] = "AL" if latest_ind['Close'] > latest_ind['SMA50'] else "SAT"
    comments['EMA20'] = "AL" if latest_ind['Close'] > latest_ind['EMA20'] else "SAT"
    comments['EMA50'] = "AL" if latest_ind['Close'] > latest_ind['EMA50'] else "SAT"
    comments['RSI'] = "AL" if latest_ind['RSI'] < 30 else ("SAT" if latest_ind['RSI'] > 70 else "NÖTR")
    comments['MACD'] = "AL" if latest_ind['MACD'] > latest_ind['MACD_SIGNAL'] else "SAT"
    comments['Stoch_K'] = "AL" if latest_ind['Stoch_K'] < 20 else ("SAT" if latest_ind['Stoch_K'] > 80 else "NÖTR")
    comments['Stoch_D'] = "AL" if latest_ind['Stoch_D'] < 20 else ("SAT" if latest_ind['Stoch_D'] > 80 else "NÖTR")
    comments['CCI'] = "AL" if latest_ind['CCI'] < -100 else ("SAT" if latest_ind['CCI'] > 100 else "NÖTR")
    comments['WilliamsR'] = "AL" if latest_ind['WilliamsR'] < -80 else ("SAT" if latest_ind['WilliamsR'] > -20 else "NÖTR")
    comments['ATR'] = "AL" if latest_ind['ATR'] < latest_ind['Close']*0.02 else "NÖTR"
    comments['BB_high'] = "SAT" if latest_ind['Close'] >= latest_ind['BB_high'] else "NÖTR"
    comments['BB_low'] = "AL" if latest_ind['Close'] <= latest_ind['BB_low'] else "NÖTR"
    comments['MFI'] = "AL" if latest_ind['MFI'] < 30 else ("SAT" if latest_ind['MFI'] > 70 else "NÖTR")
    comments['OBV'] = "AL" if latest_ind['OBV'] > 0 else "SAT"
    comments['ADX'] = "AL" if latest_ind['ADX'] > 25 else "NÖTR"
    return comments

# -------------------------
# Random Forest Tahmini
# -------------------------
def rf_predict(df_ind):
    X = df_ind.drop(columns=['Close']).values[:-1]
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
st.title("📊 BIST100 15 İndikatör + RF Tahmini (1 Gün Sonrası)")

symbol = st.text_input("🔎 Hisse Kodu", value="AEFES").upper()

if symbol:
    df = get_data(symbol, period="52d", interval="4h")
    if df is None or len(df) < 30:
        st.warning("Yeterli veri yok veya veri çekme hatası oluştu.")
    else:
        df_ind = calculate_indicators(df)
        latest_ind = df_ind.iloc[-1]
        comments = interpret_indicators(latest_ind)
        close_price = latest_ind['Close']

        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        # İndikatör Tablosu
        st.markdown("### 🔎 İndikatörler ve Yorumlar")
        result_df = pd.DataFrame([(k, f"{latest_ind[k]:.2f}", comments.get(k,"")) for k in latest_ind.index],
                                 columns=["İndikatör","Değer","Yorum"])
        st.dataframe(result_df, use_container_width=True)

        # Random Forest Tahmini
        rf_pred = rf_predict(df_ind)
        if rf_pred: 
            st.markdown(f"### 📊 1 Gün Sonrası RF Tahmini: {rf_pred:.2f} ₺")
        else:
            st.markdown("### 📊 RF Tahmini: Veri yetersiz")

        # Genel Al/Sat Tavsiyesi
        al_count = sum([1 for v in comments.values() if v=="AL"])
        sat_count = sum([1 for v in comments.values() if v=="SAT"])
        if rf_pred and rf_pred > close_price:
            al_count +=1
        elif rf_pred:
            sat_count +=1

        st.markdown("### 📢 Genel Teknik Yorum ve Tavsiye")
        if al_count > sat_count:
            st.write(f"- **Tavsiyesi:** AL  (AL:{al_count} / SAT:{sat_count})")
        elif sat_count > al_count:
            st.write(f"- **Tavsiyesi:** SAT (AL:{al_count} / SAT:{sat_count})")
        else:
            st.write(f"- **Tavsiyesi:** NÖTR (AL:{al_count} / SAT:{sat_count})")
