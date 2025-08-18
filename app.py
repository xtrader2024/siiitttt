import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
# 15 İndikatör Hesaplama
# -------------------------
def calculate_indicators(df):
    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    inds = {}

    # Trend: SMA ve EMA
    inds['SMA20'] = close.rolling(20).mean()
    inds['SMA50'] = close.rolling(50).mean()
    inds['EMA20'] = close.ewm(span=20, adjust=False).mean()
    inds['EMA50'] = close.ewm(span=50, adjust=False).mean()

    # Momentum: RSI, CCI, Stoch
    inds['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    inds['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    inds['STOCH_K'] = stoch.stoch()
    inds['STOCH_D'] = stoch.stoch_signal()

    # Trend Strength: ADX
    inds['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    # Volatilite: ATR, Bollinger Bands
    inds['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    boll = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    inds['BB_high'] = boll.bollinger_hband()
    inds['BB_low'] = boll.bollinger_lband()

    # Hacim: MFI ve OBV
    inds['MFI'] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
    inds['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # MACD ve Williams %R
    macd = ta.trend.MACD(close)
    inds['MACD'] = macd.macd()
    inds['MACD_SIGNAL'] = macd.macd_signal()
    inds['WILLR'] = ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()

    df_ind = pd.concat(inds.values(), axis=1, keys=inds.keys())
    df_ind.dropna(inplace=True)
    return df_ind

# -------------------------
# İndikatör Yorumlama
# -------------------------
def interpret_indicators(latest_ind):
    comments = {}
    # Al=+1, Sat=-1, Nötr=0
    comments['SMA20'] = ('Al' if latest_ind['Close'] > latest_ind['SMA20'] else 'Sat')
    comments['SMA50'] = ('Al' if latest_ind['Close'] > latest_ind['SMA50'] else 'Sat')
    comments['EMA20'] = ('Al' if latest_ind['Close'] > latest_ind['EMA20'] else 'Sat')
    comments['EMA50'] = ('Al' if latest_ind['Close'] > latest_ind['EMA50'] else 'Sat')
    comments['RSI'] = ('Al' if latest_ind['RSI'] <30 else ('Sat' if latest_ind['RSI']>70 else 'Nötr'))
    comments['CCI'] = ('Al' if latest_ind['CCI']<-100 else ('Sat' if latest_ind['CCI']>100 else 'Nötr'))
    comments['STOCH_K'] = ('Al' if latest_ind['STOCH_K']<20 else ('Sat' if latest_ind['STOCH_K']>80 else 'Nötr'))
    comments['STOCH_D'] = ('Al' if latest_ind['STOCH_D']<20 else ('Sat' if latest_ind['STOCH_D']>80 else 'Nötr'))
    comments['ADX'] = ('Al' if latest_ind['ADX']>25 else 'Sat')
    comments['ATR'] = ('Al' if latest_ind['ATR']<0.02*latest_ind['Close'] else 'Sat')
    comments['BB_high'] = ('Sat' if latest_ind['Close']>=latest_ind['BB_high'] else 'Al')
    comments['BB_low'] = ('Al' if latest_ind['Close']<=latest_ind['BB_low'] else 'Sat')
    comments['MFI'] = ('Al' if latest_ind['MFI']<30 else ('Sat' if latest_ind['MFI']>70 else 'Nötr'))
    comments['OBV'] = ('Al' if latest_ind['OBV']>0 else 'Sat')
    comments['MACD'] = ('Al' if latest_ind['MACD']>latest_ind['MACD_SIGNAL'] else 'Sat')
    comments['WILLR'] = ('Al' if latest_ind['WILLR']<-80 else ('Sat' if latest_ind['WILLR']>-20 else 'Nötr'))
    return comments

# -------------------------
# Random Forest Tahmini
# -------------------------
def rf_predict(df_ind):
    features = df_ind.columns.tolist()
    X = df_ind[features].values[:-1]
    y = df_ind['Close'].values[1:]
    if len(X)<20:
        return None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    pred = model.predict(X_scaled[-1].reshape(1,-1))[0]
    return pred

# -------------------------
# Streamlit Arayüz
# -------------------------
st.title("📊 BIST100 Teknik Analiz + 1 Gün Sonrası Tahmin")

symbol = st.text_input("🔎 Hisse Kodu", value="AEFES").upper()

if symbol:
    df = get_data(symbol, period="52d", interval="4h")
    if df is None or len(df)<52:
        st.warning("Yeterli veri yok veya veri çekme hatası oluştu.")
    else:
        df_ind = calculate_indicators(df)
        latest_ind = df_ind.iloc[-1]
        close_price = latest_ind['Close']
        comments = interpret_indicators(latest_ind)

        st.subheader(f"{symbol} - Son Analiz")
        st.write(f"📌 **Son Kapanış:** {close_price:.2f} ₺")

        st.markdown("### 🔎 İndikatörler ve AL/SAT Yorumları")
        result_df = pd.DataFrame([(k,f"{latest_ind[k]:.2f}",v) for k,v in comments.items()],
                                 columns=["İndikatör","Değer","Yorum"])
        st.dataframe(result_df, use_container_width=True)

        # Random Forest 1 Gün Sonrası Tahmin
        rf_pred = rf_predict(df_ind)
        if rf_pred:
            st.markdown(f"### 📈 1 Gün Sonrası Fiyat Tahmini (Random Forest): **{rf_pred:.2f} ₺**")
        else:
            st.write("1 Gün sonrası tahmin: Veri yetersiz")

        # Ensemble Al/Sat Tavsiyesi
        score = 0
        # İndikatörlerden puanlama
        for v in comments.values():
            if v=='Al': score+=1
            elif v=='Sat': score-=1
        # Random Forest yönü
        if rf_pred:
            score += 1 if rf_pred>close_price else -1
        st.markdown("### 📢 Genel Tavsiye (Ensemble)")
        if score>0:
            st.write(f"- **Tavsiyesi:** AL  (Skor: {score})")
        elif score<0:
            st.write(f"- **Tavsiyesi:** SAT (Skor: {score})")
        else:
            st.write(f"- **Tavsiyesi:** NÖTR (Skor: {score})")
