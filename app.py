import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64

st.set_page_config(page_title="BIST100 Trend Dönüş Analizi", layout="wide")

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
    gain = np.where(delta > 0, delta, 0).ravel()
    loss = np.where(delta < 0, -delta, 0).ravel()
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(0)

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

# ===================== TEK HİSSE İŞLE ===================== #
def process_stock(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if df.empty: return None
        df = calculate_indicators(df)
        df = generate_signals(df)
        last_signal = df["Buy_Signal"].iloc[-1]
        last_price = df["Close"].iloc[-1]
        return {"symbol": symbol, "price": last_price, "signal": last_signal, "df": df}
    except Exception as e:
        return None

# ===================== ANA PROGRAM ===================== #
def main():
    st.title("📈 BIST100 Trend Dönüş Analizi (Hata Düzeltildi)")

    results = []
    with st.spinner("BIST100 taranıyor..."):
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_stock, s) for s in BIST100_SYMBOLS]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)

    buy_stocks = [r for r in results if r["signal"] == "BUY"]

    st.subheader("📊 Trend Dönüş Sinyali Olan Hisseler")
    for stock in buy_stocks:
        st.write(f"{stock['symbol']} - {stock['price']:.2f} TL - {stock['signal']}")
        fig = plot_stock(stock["df"], stock["symbol"])
        st.pyplot(fig)

    results_df = pd.DataFrame([{"Hisse": r["symbol"], "Son Fiyat": r["price"], "Sinyal": r["signal"]} for r in results])
    st.subheader("📋 Tüm BIST100 Analiz Sonuçları")
    st.dataframe(results_df)

    # CSV indirme
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="BIST100_analysis.csv">📥 CSV Olarak İndir</a>'
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
