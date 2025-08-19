import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import numpy as np

st.set_page_config(page_title="Hisse Teknik Analiz (Detaylı)", layout="centered")
st.title("📊 BIST100 – Detaylı Teknik Analiz")

symbol = st.text_input("Hisse kodunu girin (örn: AEFES):").upper()

def analyze(symbol):
    try:
        df = yf.download(f"{symbol}.IS",
                         period="7d",
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

        inds = {
            'RSI': ta.momentum.RSIIndicator(close).rsi().iloc[-1],
            'MACD': ta.trend.MACD(close).macd_diff().iloc[-1],
            'SMA20': ta.trend.SMAIndicator(close, 20).sma_indicator().iloc[-1],
            'EMA20': ta.trend.EMAIndicator(close, 20).ema_indicator().iloc[-1],
            'MFI': ta.volume.MFIIndicator(high, low, close, volume).money_flow_index().iloc[-1],
            'ADX': ta.trend.ADXIndicator(high, low, close).adx().iloc[-1],
            'CCI': ta.trend.CCIIndicator(high, low, close).cci().iloc[-1],
            'STOCH': ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1],
            'WILLR': ta.momentum.WilliamsRIndicator(high, low, close).williams_r().iloc[-1]
        }

        score = sum([
            inds['RSI'] > 50,
            inds['MACD'] > 0,
            close.iloc[-1] > inds['SMA20'],
            close.iloc[-1] > inds['EMA20'],
            inds['MFI'] > 50,
            inds['ADX'] > 20,
            inds['CCI'] > 0,
            inds['STOCH'] > 50,
            inds['WILLR'] > -80
        ])

        details = [
            f"RSI: {inds['RSI']:.2f}",
            f"MACD: {inds['MACD']:.4f}",
            f"SMA20: {inds['SMA20']:.2f}",
            f"EMA20: {inds['EMA20']:.2f}",
            f"MFI: {inds['MFI']:.2f}",
            f"ADX: {inds['ADX']:.2f}",
            f"CCI: {inds['CCI']:.2f}",
            f"STOCH: {inds['STOCH']:.2f}",
            f"Williams %R: {inds['WILLR']:.2f}"
        ]

        signal = ("🔼 GÜÇLÜ AL" if score >= 7 else
                  "⚠️ AL Sinyali" if score >= 5 else
                  "🔽 NÖTR")

        # Yorum ekleme
        yorum = []
        if inds['RSI'] > 70:
            yorum.append("RSI yüksek: aşırı alım bölgesi, dikkat.")
        elif inds['RSI'] < 30:
            yorum.append("RSI düşük: aşırı satım bölgesi, potansiyel alım fırsatı.")
        else:
            yorum.append("RSI orta seviyede.")

        if inds['MFI'] > 80:
            yorum.append("Hacim yoğunluğu yüksek, fiyatın yükselişi güçlü olabilir.")
        elif inds['MFI'] < 20:
            yorum.append("Hacim düşük, fiyat hareketi zayıf.")
        else:
            yorum.append("Hacim dengeli.")

        if close.iloc[-1] > inds['SMA20']:
            yorum.append("Fiyat SMA20 üzerinde, kısa vadeli trend pozitif.")
        else:
            yorum.append("Fiyat SMA20 altında, kısa vadeli trend negatif.")

        # Basit 24 saatlik tahmin (son fiyat ve MACD/ADX trendine göre)
        tahmin_degisim = 0
        if inds['MACD'] > 0 and inds['ADX'] > 20:
            tahmin_degisim = close.iloc[-1] * 0.01  # +1% tahmin
        elif inds['MACD'] < 0 and inds['ADX'] > 20:
            tahmin_degisim = -close.iloc[-1] * 0.01  # -1% tahmin

        tahmini_fiyat = close.iloc[-1] + tahmin_degisim
        yorum.append(f"24 saat sonrası tahmini fiyat: {tahmini_fiyat:.2f} TRY (basit tahmin)")

        return {
            "Hisse": symbol,
            "Fiyat": round(close.iloc[-1], 2),
            "Puan": score,
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
        st.write(f"• Fiyat: {result['Fiyat']}")
        st.write(f"• Puan: {result['Puan']} / 9")
        st.write(f"• Sinyal: {result['Sinyal']}")

        st.markdown("#### İndikatör Detayları:")
        for d in result['Detay']:
            st.markdown(f"- {d}")

        st.markdown("#### Analiz ve Yorum:")
        for y in result['Yorum']:
            st.markdown(f"- {y}")
