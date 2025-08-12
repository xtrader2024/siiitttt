import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Teknik gösterge hesaplama için ta-lib alternatifi - talib-python yoksa manual hesaplama yapılabilir
import talib

# --- Fonksiyonlar ---

def download_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"{ticker} için veri alınamadı.")
            return None
        return df
    except Exception as e:
        st.error(f"Veri indirme hatası: {e}")
        return None

def calculate_indicators(df):
    df = df.copy()
    
    # Basit hareketli ortalamalar
    df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA50'] = talib.SMA(df['Close'], timeperiod=50)
    
    # RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    
    # Bollinger Bands
    upperband, middleband, lowerband = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_upper'] = upperband
    df['BB_middle'] = middleband
    df['BB_lower'] = lowerband
    
    # ADX
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    # Stochastic RSI (momentum)
    slowk, slowd = talib.STOCHRSI(df['Close'], timeperiod=14)
    df['StochRSI_K'] = slowk
    df['StochRSI_D'] = slowd
    
    # On Balance Volume (OBV) (hacim trendi)
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    
    return df

def generate_signals(df):
    signals = []
    
    # Trend yönü - SMA
    if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]:
        signals.append("📈 Kısa vadeli trend: Yükseliş")
    else:
        signals.append("📉 Kısa vadeli trend: Düşüş")
    
    # RSI sinyali
    if df['RSI'].iloc[-1] > 70:
        signals.append("🔴 RSI: Aşırı alım (Satış sinyali)")
    elif df['RSI'].iloc[-1] < 30:
        signals.append("🟢 RSI: Aşırı satım (Alım sinyali)")
    else:
        signals.append("⚪️ RSI: Nötr")
    
    # MACD sinyali
    if df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]:
        signals.append("🟢 MACD: Alım sinyali")
    else:
        signals.append("🔴 MACD: Satış sinyali")
    
    # Bollinger Bands
    if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1]:
        signals.append("🔴 Fiyat üst Bollinger Bandında - Düzeltme beklenebilir")
    elif df['Close'].iloc[-1] < df['BB_lower'].iloc[-1]:
        signals.append("🟢 Fiyat alt Bollinger Bandında - Alım fırsatı olabilir")
    else:
        signals.append("⚪️ Bollinger Bandı: Normal bant içinde")
    
    # ADX trend gücü
    if df['ADX'].iloc[-1] > 25:
        signals.append("💪 ADX: Güçlü trend mevcut")
    else:
        signals.append("⚪️ ADX: Zayıf trend")
    
    # StochRSI Momentum
    if df['StochRSI_K'].iloc[-1] > 80:
        signals.append("🔴 StochRSI: Aşırı alım")
    elif df['StochRSI_K'].iloc[-1] < 20:
        signals.append("🟢 StochRSI: Aşırı satım")
    else:
        signals.append("⚪️ StochRSI: Nötr")
    
    # OBV hacim trendi
    if df['OBV'].iloc[-1] > df['OBV'].iloc[-2]:
        signals.append("📊 OBV: Hacim artıyor, alım gücü var")
    else:
        signals.append("📉 OBV: Hacim düşüyor, zayıflama sinyali")
    
    return signals

def plot_data(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name="Fiyat"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='blue', width=1), name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange', width=1), name='SMA50'))
    fig.update_layout(title=f"{ticker} Fiyat Grafiği ve SMA",
                      xaxis_title="Tarih", yaxis_title="Fiyat")
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit Arayüz ---

st.title("📊 BIST100 Teknik Analiz (Profesyonel)")

ticker = st.text_input("Hisse kodunu girin (örn: AEFES)").upper()

if ticker:
    st.info(f"{ticker} için veri indiriliyor...")
    data = download_data(ticker)
    if data is not None:
        st.success(f"{ticker} için veri indirildi, analiz başlatılıyor...")
        df_ind = calculate_indicators(data)
        signals = generate_signals(df_ind)
        
        st.subheader(f"{ticker} Teknik Analiz Sonuçları:")
        for s in signals:
            st.write(s)
        
        st.subheader("Grafik")
        plot_data(df_ind, ticker)
        
        # Ek bilgiler
        st.markdown("---")
        st.markdown("### Trend Yönü ve Hacim Yorumu")
        trend = "Yükseliş" if df_ind['SMA20'].iloc[-1] > df_ind['SMA50'].iloc[-1] else "Düşüş"
        volume_trend = "Artıyor" if df_ind['OBV'].iloc[-1] > df_ind['OBV'].iloc[-2] else "Azalıyor"
        st.write(f"- **Trend Yönü:** {trend}")
        st.write(f"- **Hacim:** {volume_trend}")
        
        # Basit hedef fiyat (örnek)
        recent_return = df_ind['Close'].pct_change().mean()
        target_price = df_ind['Close'].iloc[-1] * (1 + recent_return * 5)
        st.write(f"- **Basit hedef fiyat (5 gün sonrası):** {target_price:.2f} TRY")
