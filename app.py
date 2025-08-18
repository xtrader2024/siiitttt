import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet

st.set_page_config(page_title="BIST100 Gelişmiş Teknik Analiz + Tahmin", layout="centered")

# -------------------------
# Veri Çekme
# -------------------------
@st.cache_data(ttl=3600)
def get_data(symbol, period="1y", interval="1d"):
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
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean()
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    inds['MACD'] = ta.trend.MACD(close).macd()
    inds['MACD_SIGNAL'] = ta.trend.MACD(close).macd_signal()
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    inds['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    inds['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df_ind = pd.concat(inds.values(), axis=1, keys=inds.keys())
    df_ind.dropna(inplace=True)
    return df_ind

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
# LSTM Hazırlık ve Tahmin
# -------------------------
def prepare_lstm_data(df_ind, window_size=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_ind)
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i, df_ind.columns.get_loc("Close")])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def lstm_predict(df_ind):
    window_size = 10
    if len(df_ind) < window_size + 5:
        return None
    X, y, scaler = prepare_lstm_data(df_ind, window_size)
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    last_input = X[-1].reshape(1, window_size, X.shape[2])
    pred_scaled = model.predict(last_input, verbose=0)
    close_idx = df_ind.columns.get_loc("Close")
    min_val = df_ind['Close'].min()
    max_val = df_ind['Close'].max()
    pred = pred_scaled[0][0] * (max_val - min_val) + min_val
    return pred

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
# Prophet Tahmini
# -------------------------
def prophet_predict(df):
    df_prophet = df.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    if len(df_prophet) < 30:
        return None
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-1]

# -------------------------
# Streamlit Arayüz
# -------------------------
st.title("📊 BIST100 Gelişmiş Teknik Analiz + Ensemble Tahmin")

symbol = st.text_input("🔎 Hisse Kodu", value="AEFES").upper()
period = st.selectbox("Dönem", ["3mo","6mo","1y"], index=1)
interval = st.selectbox("Zaman Aralığı", ["1d","1h"], index=0)

if symbol:
    df = get_data(symbol, period=period, interval=interval)
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

        # Tahminler
        lstm_pred = lstm_predict(df_ind)
        rf_pred = rf_predict(df_ind)
        prophet_pred = prophet_predict(df)

        st.markdown("### 📊 Tahminler")
        if lstm_pred: st.write(f"- **LSTM Tahmini:** {lstm_pred:.2f} ₺")
        else: st.write("- LSTM Tahmini: Veri yetersiz")
        if rf_pred: st.write(f"- **Random Forest Tahmini:** {rf_pred:.2f} ₺")
        else: st.write("- Random Forest Tahmini: Veri yetersiz")
        if prophet_pred: st.write(f"- **Prophet Tahmini (5 gün sonrası):** {prophet_pred:.2f} ₺")
        else: st.write("- Prophet Tahmini: Veri yetersiz")

        # Ensemble Al/Sat Tavsiyesi
        scores = []
        if lstm_pred and lstm_pred > close_price: scores.append(1)
        elif lstm_pred: scores.append(-1)
        if rf_pred and rf_pred > close_price: scores.append(1)
        elif rf_pred: scores.append(-1)
        if prophet_pred and prophet_pred > close_price: scores.append(1)
        elif prophet_pred: scores.append(-1)
        # İndikatör ağırlığı
        ind_score = 0
        if latest_ind['RSI'] < 30: ind_score +=1
        elif latest_ind['RSI'] >70: ind_score -=1
        if latest_ind['MACD'] > latest_ind['MACD_SIGNAL']: ind_score +=1
        else: ind_score -=1
        if latest_ind['ADX'] >25: ind_score +=1
        else: ind_score -=1
        scores.append(ind_score)

        ensemble_score = np.sum(scores)
        st.markdown("### 📢 Ensemble Al/Sat Tavsiyesi")
        if ensemble_score > 0:
            st.write(f"- **Tavsiyesi:** AL  (Skor: {ensemble_score})")
        elif ensemble_score < 0:
            st.write(f"- **Tavsiyesi:** SAT (Skor: {ensemble_score})")
        else:
            st.write(f"- **Tavsiyesi:** NÖTR (Skor: {ensemble_score})")
