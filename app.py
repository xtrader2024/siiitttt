import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import timedelta

# --- optional deps ---
try:
    from prophet import Prophet  # pip install prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

try:
    import tensorflow as tf  # pip install tensorflow
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_OK = True
except Exception:
    TENSORFLOW_OK = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="BIST100 Tahmin + ML (Grafiksiz)", layout="centered")

# =========================
# DATA
# =========================
@st.cache_data(ttl=3600)
def get_data(symbol, period="2y", interval="1d"):
    try:
        df = yf.download(f"{symbol}.IS", period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

# =========================
# TECHNICALS
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close, high, low, vol = d['Close'], d['High'], d['Low'], d['Volume']

    # Trend
    d['SMA20'] = close.rolling(20).mean()
    d['EMA20'] = close.ewm(span=20, adjust=False).mean()

    # Momentum / Osc
    d['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    d['CCI'] = ta.trend.CCIIndicator(high, low, close, window=20).cci()
    d['ADX'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()

    macd = ta.trend.MACD(close)
    d['MACD'] = macd.macd()
    d['MACD_SIGNAL'] = macd.macd_signal()
    d['MACD_HIST'] = macd.macd_diff()

    # Vol / Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    d['BB_H'] = bb.bollinger_hband()
    d['BB_L'] = bb.bollinger_lband()
    d['BB_WIDTH'] = (d['BB_H'] - d['BB_L']) / close

    d['ATR'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume-based
    d['MFI'] = ta.volume.MFIIndicator(high, low, close, vol, window=14).money_flow_index()
    d['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    # Returns & volatility
    d['RET_1'] = close.pct_change()
    d['RET_5'] = close.pct_change(5)
    d['VOL_20'] = d['RET_1'].rolling(20).std()
    d['VOL_60'] = d['RET_1'].rolling(60).std()

    # Targets
    d['TARGET_RET_1'] = d['RET_1'].shift(-1)  # ertesi gün getiri
    d.dropna(inplace=True)
    return d

# =========================
# TEXT INTERPRETATION
# =========================
def indicator_summary_row(name, val, ctx):
    # ctx dict may carry thresholds if needed
    comment = ""
    if name == "RSI":
        comment = "Aşırı alım" if val > 70 else ("Aşırı satım" if val < 30 else "Nötr")
    elif name == "ADX":
        comment = "Trend güçlü" if val > 25 else "Trend zayıf"
    elif name == "MACD":
        comment = "Al sinyali" if ctx['MACD'] > ctx['MACD_SIGNAL'] else "Sat sinyali"
    elif name == "SMA20":
        comment = "Fiyat üstünde → yükseliş" if ctx['Close'] > val else "Fiyat altında → düşüş"
    elif name == "EMA20":
        comment = "Fiyat üstünde → yükseliş" if ctx['Close'] > val else "Fiyat altında → düşüş"
    elif name == "MFI":
        comment = "Para girişi güçlü" if val > 50 else "Para çıkışı baskın"
    elif name == "BB":
        if ctx['Close'] >= ctx['BB_H']: comment = "Üst banda yakın (aşırı alım riski)"
        elif ctx['Close'] <= ctx['BB_L']: comment = "Alt banda yakın (alım fırsatı)"
        else: comment = "Bant içinde"
    elif name == "ATR":
        thresh = ctx['Close'] * ctx['VOL_20'] if not np.isnan(ctx['VOL_20']) else val
        comment = "Volatilite yüksek" if val > thresh else "Volatilite normal"
    return comment

def trend_momentum_view(row):
    trend = "Yukarı" if (row['Close'] > row['EMA20']) and (row['Close'] > row['SMA20']) else "Aşağı"
    trend_strength = "Güçlü" if row['ADX'] > 25 else "Zayıf"
    momentum = "Pozitif" if (row['RSI'] > 50) and (row['MACD'] > row['MACD_SIGNAL']) else "Negatif"
    return trend, trend_strength, momentum

# =========================
# FORECASTERS
# =========================
def forecast_arima(series: pd.Series, steps=5):
    # simple SARIMAX(1,1,1) as a baseline
    try:
        model = SARIMAX(series, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.forecast(steps)
        return fc
    except Exception:
        return None

def forecast_prophet(df: pd.DataFrame, steps=5):
    if not PROPHET_OK:
        return None
    try:
        tmp = df[['Close']].reset_index().rename(columns={'Date':'ds','Close':'y'})
        # Some yfinance indexes may already be datetime; Prophet expects ds,y
        if 'ds' not in tmp.columns:
            tmp.columns = ['ds','y']
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(tmp)
        future = m.make_future_dataframe(periods=steps, freq='D')
        forecast = m.predict(future).set_index('ds').iloc[-steps:]['yhat']
        return forecast
    except Exception:
        return None

def forecast_lstm(df: pd.DataFrame, steps=5, lookback=30, epochs=8):
    if not TENSORFLOW_OK:
        return None
    try:
        series = df['Close'].astype('float32').values.reshape(-1,1)
        # scale
        mn, mx = series.min(), series.max()
        scaled = (series - mn) / (mx - mn + 1e-9)

        X, y = [], []
        for i in range(len(scaled)-lookback-1):
            X.append(scaled[i:i+lookback, 0])
            y.append(scaled[i+lookback, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([LSTM(32, input_shape=(lookback,1)), Dense(1)])
        model.compile(optimizer='adam', loss='mae')
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

        # iterative forecast
        last_seq = scaled[-lookback:, 0].tolist()
        preds = []
        for _ in range(steps):
            arr = np.array(last_seq[-lookback:]).reshape((1,lookback,1))
            p = float(model.predict(arr, verbose=0)[0,0])
            preds.append(p)
            last_seq.append(p)

        preds = np.array(preds)*(mx - mn + 1e-9) + mn
        idx = pd.date_range(df.index[-1] + timedelta(days=1), periods=steps, freq='D')
        return pd.Series(preds, index=idx)
    except Exception:
        return None

# =========================
# ML (FEATURE-BASED)
# =========================
FEATURES = [
    'RSI','MACD','MACD_SIGNAL','MACD_HIST','ADX','ATR','MFI','OBV',
    'SMA20','EMA20','BB_WIDTH','VOL_20','VOL_60','RET_1','RET_5','Volume'
]

def ml_next_return(df_feat: pd.DataFrame):
    d = df_feat.copy()
    # ensure Volume as numeric
    if 'Volume' not in d.columns:
        d['Volume'] = d['Volume']

    # Drop inf/nan
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ['TARGET_RET_1'])
    if len(d) < 200:
        return None, None, None  # not enough data

    X = d[FEATURES].values
    y = d['TARGET_RET_1'].values

    split = int(len(d)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # latest prediction for next day
    last_row = d.iloc[-1:][FEATURES].values
    next_ret_pred = float(model.predict(last_row)[0])
    return next_ret_pred, r2, mae

# =========================
# SIGNAL ENGINE
# =========================
def signal_engine(row):
    score = 0
    notes = []

    # RSI
    if row['RSI'] < 30: score += 2; notes.append("RSI aşırı satım (AL)")
    elif row['RSI'] > 70: score -= 2; notes.append("RSI aşırı alım (SAT)")

    # MACD
    if row['MACD'] > row['MACD_SIGNAL']: score += 2; notes.append("MACD yukarı kesmiş (AL)")
    else: score -= 1; notes.append("MACD aşağı kesmiş (SAT)")

    # ADX
    if row['ADX'] > 25: score += 1; notes.append("Trend güçlü")
    else: notes.append("Trend zayıf")

    # Bollinger
    if row['Close'] <= row['BB_L']: score += 1; notes.append("Alt banda yakın (AL)")
    if row['Close'] >= row['BB_H']: score -= 1; notes.append("Üst banda yakın (SAT)")

    # MFI
    if row['MFI'] < 20: score += 1; notes.append("MFI düşük (AL)")
    if row['MFI'] > 80: score -= 1; notes.append("MFI yüksek (SAT)")

    # Final decision
    if score >= 3: decision = "🟢 AL"
    elif score >= 1: decision = "🟡 BEKLE"
    else: decision = "🔴 SAT"
    return score, decision, notes

# =========================
# UI
# =========================
st.title("📊 BIST100 – Zaman Serisi Tahmini + ML (Grafiksiz)")

symbol = st.text_input("🔎 Hisse kodu girin (ör. THYAO, ASELS, GARAN)").upper().strip()
period = st.selectbox("Dönem", ["6mo","1y","2y","5y"], index=2)
interval = st.selectbox("Zaman Aralığı", ["1d"], index=0)

use_arima = st.checkbox("ARIMA (SARIMAX 1,1,1)", value=True)
use_prophet = st.checkbox("Prophet (varsa)", value=True and PROPHET_OK)
use_lstm = st.checkbox("LSTM (varsa)", value=False and TENSORFLOW_OK)
use_ml = st.checkbox("ML – RandomForest (RSI, MACD, ADX vb.)", value=True)

if not symbol:
    st.info("Lütfen bir hisse kodu girin.")
    st.stop()

df = get_data(symbol, period=period, interval=interval)
if df is None or df.empty or len(df) < 120:
    st.error("Yeterli veri yok (en az ~120 bar önerilir).")
    st.stop()

# Indicators + features
df_feat = add_indicators(df)
last = df_feat.iloc[-1]

# ---- INDICATOR TABLE (compact, text only)
st.subheader(f"{symbol} – Son Değerler & Yorumlar")
rows = []
for name in ["SMA20","EMA20","RSI","ADX","MACD","MACD_SIGNAL","ATR","MFI","BB_H","BB_L","BB_WIDTH","VOL_20"]:
    val = float(last[name])
    ctx = {
        'Close': float(last['Close']),
        'MACD': float(last['MACD']),
        'MACD_SIGNAL': float(last['MACD_SIGNAL']),
        'BB_H': float(last['BB_H']),
        'BB_L': float(last['BB_L']),
        'VOL_20': float(last['VOL_20']) if not np.isnan(last['VOL_20']) else np.nan
    }
    if name in ["BB_H","BB_L"]:
        label = "BB"  # tek yorum satırı üretelim
    else:
        label = name
    comment = indicator_summary_row(label, val, ctx)
    rows.append([name, f"{val:.4f}", comment])

st.dataframe(pd.DataFrame(rows, columns=["İndikatör", "Değer", "Yorum"]))

# ---- TREND / MOMENTUM
trend, trend_strength, momentum = trend_momentum_view(last)
st.markdown("### 📊 Genel Teknik Yorum")
st.write(f"- **Trend Yönü:** {trend}")
st.write(f"- **Trend Gücü (ADX):** {trend_strength}")
st.write(f"- **Momentum (RSI+MACD):** {momentum}")

# ---- FORECASTS
st.markdown("### 🔮 Zaman Serisi Tahminleri (5 gün)")
preds = []
weights = []
explanations = []

if use_arima:
    arima_fc = forecast_arima(df['Close'], steps=5)
    if arima_fc is not None:
        preds.append(arima_fc.values)
        weights.append(1.0)
        explanations.append("ARIMA aktif")
    else:
        st.warning("ARIMA başarısız.")

if use_prophet:
    if PROPHET_OK:
        prop_fc = forecast_prophet(df, steps=5)
        if prop_fc is not None:
            preds.append(prop_fc.values)
            weights.append(1.0)
            explanations.append("Prophet aktif")
        else:
            st.warning("Prophet tahmini başarısız.")
    else:
        st.info("Prophet kurulu değil, atlandı.")

if use_lstm:
    if TENSORFLOW_OK:
        lstm_fc = forecast_lstm(df, steps=5)
        if lstm_fc is not None:
            preds.append(lstm_fc.values)
            weights.append(1.0)
            explanations.append("LSTM aktif")
        else:
            st.warning("LSTM tahmini başarısız.")
    else:
        st.info("TensorFlow kurulu değil, LSTM atlandı.")

# ML prediction (next-day return -> price path)
ml_pred_ret, ml_r2, ml_mae = (None, None, None)
if use_ml:
    ml_pred_ret, ml_r2, ml_mae = ml_next_return(df_feat)
    if ml_pred_ret is not None:
        last_close = float(last['Close'])
        # 5 gün için basit birikimli büyüt (sabit günlük ML tahmini varsayımı)
        ml_path = [last_close * (1 + ml_pred_ret)]
        for _ in range(4):
            ml_path.append(ml_path[-1] * (1 + ml_pred_ret))
        preds.append(np.array(ml_path))
        weights.append(1.0)
        explanations.append(f"ML aktif (R2: {ml_r2:.2f}, MAE: {ml_mae:.4f})")
    else:
        st.info("ML eğitimi için yeterli temiz veri yok veya hata oluştu.")

# ENSEMBLE
if len(preds) == 0:
    st.warning("Aktif bir tahmin üretilemedi.")
    ensemble = None
else:
    arr = np.vstack(preds)
    w = np.array(weights).reshape(-1,1)
    # ağırlıklı ortalama (eşit ağırlık)
    ensemble = (arr * w).sum(axis=0) / w.sum()

last_close = float(last['Close'])
if ensemble is not None:
    st.write(f"- **Son Kapanış:** {last_close:.2f} ₺")
    st.write(f"- **Ensemble 5. Gün Tahmini:** {ensemble[-1]:.2f} ₺")
    exp_txt = ", ".join(explanations)
    st.write(f"- **Kullanılan Modeller:** {exp_txt}")
else:
    st.write(f"- **Son Kapanış:** {last_close:.2f} ₺")
    st.write("- **Tahmin üretilemedi.**")

# ---- SIGNALS & RECOMMENDATION
score, decision, notes = signal_engine(last)
st.markdown("### 📢 Sinyaller (Kurallı)")
for n in notes:
    st.write(f"- {n}")

# ML sinyali (varsa) ek etkisi
ml_boost = 0
if ml_pred_ret is not None:
    if ml_pred_ret > 0.002:    # ~%0.2+
        ml_boost = 1
        st.write("- ML: Ertesi gün pozitif getiri bekleniyor (AL etkisi)")
    elif ml_pred_ret < -0.002:
        ml_boost = -1
        st.write("- ML: Ertesi gün negatif getiri bekleniyor (SAT etkisi)")

total_score = score + ml_boost
if total_score >= 3: final_decision = "🟢 AL"
elif total_score >= 1: final_decision = "🟡 BEKLE"
else: final_decision = "🔴 SAT"

st.markdown(f"### 🧮 Skor: **{total_score}**  → **Genel Tavsiye: {final_decision}**")

# ---- RISK METRICS
st.markdown("### ⚠️ Risk / Volatilite")
vol20 = float(last['VOL_20']) if not np.isnan(last['VOL_20']) else np.nan
vol60 = float(last['VOL_60']) if not np.isnan(last['VOL_60']) else np.nan
st.write(f"- 20 günlük günlük volatilite (σ): {vol20:.4f}" if not np.isnan(vol20) else "- 20g volatilite: hesaplanamadı")
st.write(f"- 60 günlük günlük volatilite (σ): {vol60:.4f}" if not np.isnan(vol60) else "- 60g volatilite: hesaplanamadı")

st.caption("Uyarı: ARIMA/Prophet/LSTM tahminleri ve ML sonuçları istatistiksel model varsayımlarına dayanır; yatırım tavsiyesi değildir.")
