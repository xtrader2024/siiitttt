import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="BIST100 Trend Dönüş Analizi", layout="wide")

# ===================== BIST100 HİSSELERİ ===================== #
BIST100_SYMBOLS = [
    "ADEL.IS","AGHOL.IS","AKBNK.IS","AKSA.IS","AKSEN.IS","ALARK.IS","ALFAS.IS","ALTNY.IS","ANHYT.IS","ANSGR.IS",
    "ARCLK.IS","ASELS.IS","ASTOR.IS","AVHOL.IS","BALAT.IS","BALSU.IS","BERA.IS","BIMAS.IS","BIZIM.IS","BJKAS.IS",
    "BMSCH.IS","BRSAN.IS","BRYAT.IS","BSOKE.IS","BUCIM.IS","CANTE.IS","CCOLA.IS","CIMSA.IS","CLEBI.IS","CWENE.IS",
    "DOAS.IS","DOHOL.IS","DSTKF.IS","EFORC.IS","EGEEN.IS","EKGYO.IS","ENERY.IS","ENJSA.IS","ENKAI.IS","EREGL.IS",
    "EUPWR.IS","FENER.IS","FROTO.IS","GARAN.IS","GENIL.IS","GESAN.IS","GLRMK.IS","GRSEL.IS","GRTHO.IS","GSRAY.IS",
    "GUBRF.IS","HALKB.IS","HEKTS.IS","IEYHO.IS","IPEKE.IS","ISCTR.IS","ISMEN.IS","KCAER.IS","KCHOL.IS","KONTR.IS",
    "KOZAA.IS","KOZAL.IS","KRDMD.IS","KTLEV.IS","KUYAS.IS","LMKDC.IS","MAGEN.IS","MAVI.IS","MGROS.IS","MIATK.IS",
    "MPARK.IS","OBAMS.IS","ODAS.IS","OTKAR.IS","OYAKC.IS","PASEU.IS","PETKM.IS","PGSUS.IS","RALYH.IS","REEDR.IS",
    "SAHOL.IS","SASA.IS","SISE.IS","SKBNK.IS","SMRTG.IS","SOKM.IS","TABGD.IS","TAVHL.IS","TCELL.IS","THYAO.IS",
    "TKFEN.IS","TOASO.IS","TSKB.IS","TTKOM.IS","TTRAK.IS","TUPRS.IS","TUREX.IS","TURSG.IS","ULKER.IS","VAKBN.IS",
    "YKBNK.IS","ZOREN.IS","ZOREL.IS"
]

# ===================== TEKNİK GÖSTERGELER ===================== #
def calculate_indicators(df):
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_Line"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD_Line"].ewm(span=9, adjust=False).mean()
    
    return df

# ===================== SİNYAL ÜRET ===================== #
def generate_signals(df):
    buy_signal = (
        ((df["EMA_50"].shift(1) < df["EMA_200"].shift(1)) & (df["EMA_50"] > df["EMA_200"])) |
        ((df["RSI"].shift(1) < 30) & (df["RSI"] > 30)) |
        ((df["MACD_Line"].shift(1) < df["MACD_Signal"].shift(1)) & (df["MACD_Line"] > df["MACD_Signal"]))
    )
    df["Buy_Signal"] = np.where(buy_signal, "BUY", "")
    return df

# ===================== GRAFİK ÇİZ ===================== #
def plot_stock(df, symbol):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], label="Close")
    ax.plot(df.index, df["EMA_50"], label="EMA50")
    ax.plot(df.index, df["EMA_200"], label="EMA200")
    ax.set_title(f"{symbol} Analizi")
    ax.legend()
    return fig

# ===================== ANA PROGRAM ===================== #
def main():
    st.title("📈 BIST100 Trend Dönüş Analizi")
    st.write("Düşüş trendi sona ermiş ve yükselişe hazırlanan hisseler.")
    
    results = []
    with st.spinner("Analiz ediliyor... Bu birkaç dakika sürebilir."):
        for symbol in BIST100_SYMBOLS:
            try:
                df = yf.download(symbol, period="6mo", interval="1d", progress=False)
                if df.empty: continue
                df = calculate_indicators(df)
                df = generate_signals(df)
                
                last_signal = df["Buy_Signal"].iloc[-1]
                last_price = df["Close"].iloc[-1]
                results.append([symbol, last_price, last_signal])
                
                # Sinyali olan hisseleri grafikle göster
                if last_signal == "BUY":
                    st.subheader(f"{symbol} - {last_price:.2f} TL - {last_signal}")
                    fig = plot_stock(df, symbol)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.write(f"{symbol} işlenemedi: {e}")
    
    results_df = pd.DataFrame(results, columns=["Hisse", "Son Fiyat", "Sinyal"])
    st.subheader("📊 Tüm BIST100 Analiz Sonuçları")
    st.dataframe(results_df)
    
    # CSV indirme linki
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="BIST100_analysis.csv">📥 CSV Olarak İndir</a>'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
