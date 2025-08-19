import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hisse Teknik Analiz (Detaylı)", layout="centered")
st.title("📊 BIST100 – Detaylı Teknik Analiz")

symbol = st.text_input("Hisse kodunu girin (örn: AEFES):").upper()

def analyze(symbol):
    try:
        df = yf.download(f"{symbol}.IS",
                         period="30d",
                         interval="1h",
                         progress=False,
                         multi_level_index=False)
        if df.empty:
            return None, f"{symbol}: veri yok ya da hatalı kod."

        df.dropna(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Teknik göstergeler
        inds = {
            'RSI': ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1],
            'MACD': ta.trend.MACD(close).macd_diff().iloc[-1],
            'SMA20': ta.trend.SMAIndicator(close, 20).sma_indicator().iloc[-1],
            'EMA20': ta.trend.EMAIndicator(close, 20).ema_indicator().iloc[-1],
            'MFI': ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index().iloc[-1],
            'ADX': ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1],
            'CCI': ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1],
            'STOCH': ta.momentum.StochasticOscillator(high, low, close, window=14).stoch().iloc[-1],
            'WILLR': ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r().iloc[-1]
        }

        # Ağırlıklı puanlama
        weights = {
            'RSI': 1.0,
            'MACD': 1.2,
            'SMA20': 1.0,
            'EMA20': 1.0,
            'MFI': 0.8,
            'ADX': 1.0,
            'CCI': 0.5,
            'STOCH': 0.5,
            'WILLR': 0.5
        }

        score = 0
        score += weights['RSI'] if inds['RSI'] > 55 else 0
        score += weights['MACD'] if inds['MACD'] > 0 else 0
        score += weights['SMA20'] if close.iloc[-1] > inds['SMA20'] else 0
        score += weights['EMA20'] if close.iloc[-1] > inds['EMA20'] else 0
        score += weights['MFI'] if inds['MFI'] > 55 else 0
        score += weights['ADX'] if inds['ADX'] > 20 else 0
        score += weights['CCI'] if inds['CCI'] > 0 else 0
        score += weights['STOCH'] if inds['STOCH'] > 50 else 0
        score += weights['WILLR'] if inds['WILLR'] > -70 else 0

        max_score = sum(weights.values())

        # Sinyal belirleme
        if score >= 0.75*max_score:
            signal = "🔼 GÜÇLÜ AL"
        elif score >= 0.5*max_score:
            signal = "⚠️ AL Sinyali"
        else:
            signal = "🔽 NÖTR"

        details = [f"{k}: {v:.2f}" for k,v in inds.items()]

        # Yorum ekleme
        yorum = []
        yorum.append(f"Toplam ağırlıklı puan: {score:.2f} / {max_score:.2f}")
        if inds['RSI'] > 70:
            yorum.append("RSI yüksek: aşırı alım bölgesi, geri çekilme riski var.")
        elif inds['RSI'] < 30:
            yorum.append("RSI düşük: aşırı satım bölgesi, alım fırsatı olabilir.")

        if close.iloc[-1] > inds['SMA20']:
            yorum.append("Fiyat SMA20 üzerinde, kısa vadeli trend pozitif.")
        else:
            yorum.append("Fiyat SMA20 altında, kısa vadeli trend negatif.")

        if close.iloc[-1] > inds['EMA20']:
            yorum.append("Fiyat EMA20 üzerinde, trend yükseliş yönünde.")
        else:
            yorum.append("Fiyat EMA20 altında, trend düşüş yönünde.")

        # 24 saatlik tahmin (Linear Regression + volatilite)
        recent_close = close[-24:].values.reshape(-1,1)
        X = np.arange(len(recent_close)).reshape(-1,1)
        model = LinearRegression()
        model.fit(X, recent_close)
        predicted_price = model.predict(np.array([[len(recent_close)]]))[0][0]
        volatility = np.std(recent_close)
        yorum.append(f"24 saat sonrası tahmini fiyat: {predicted_price:.2f} TRY "
                     f"(±{volatility:.2f} TRY aralığında)")

        return {
            "Hisse": symbol,
            "Fiyat": round(close.iloc[-1],2),
            "Puan": round(score,2),
            "Sinyal": signal,
            "Detay": details,
            "Yorum": yorum
        }, None

    except Exception as e:
        return None, f"{symbol} analiz hatası: {e}"

if symbol:
    st.info(f"{symbol} için analiz başladı...")
    result, err = analyze(symbol)
    if err:
        st.error(err)
    else:
        st.success(f"📊 {result['Hisse']} Analiz Sonucu")
        st.write(f"• Fiyat: {result['Fiyat']} TRY")
        st.write(f"• Puan: {result['Puan']}")
        st.write(f"• Sinyal: {result['Sinyal']}")
        st.markdown("#### İndikatör Detayları:")
        for d in result['Detay']:
            st.markdown(f"- {d}")
        st.markdown("#### Analiz ve Yorum:")
        for y in result['Yorum']:
            st.markdown(f"- {y}")
