import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from decimal import Decimal, getcontext
import base64

getcontext().prec = 50

# Teknik gösterge parametreleri
BOLLINGER_WINDOW = 20
RSI_TIME_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
STOCH_FASTK_PERIOD = 14
STOCH_SLOWK_PERIOD = 3

# BIST100 hisseleri (eksiksiz)
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
"YKBNK.IS","KRDMD.IS","ALCAR.IS","AYGAZ.IS","ASELS.IS","BANVT.IS","BRISA.IS","DGGYO.IS","ECILC.IS","ECIL.IS",
"EKGYO.IS","ENJSA.IS","GUBRF.IS","ISBTR.IS","IPEKE.IS","KARSN.IS","KARTN.IS","KONYA.IS","KORDS.IS","KRONT.IS",
"KOZAA.IS","KOZAL.IS","KONTR.IS","LOGO.IS","MGROS.IS","NTHOL.IS","NETAS.IS","ODEAB.IS","OTKAR.IS","PETUN.IS",
"PRKME.IS","PGSUS.IS","QUAGR.IS","SAHOL.IS","SASA.IS","SISE.IS","SOKM.IS","TAVHL.IS","TCELL.IS","TCHOL.IS",
"TKFEN.IS","TOASO.IS","TTRAK.IS","TUPRS.IS","ULKER.IS","VAKBN.IS","VESTL.IS","YKBNK.IS","ZOREN.IS","ZOREL.IS",
"ADANA.IS","ADANA.IS","AFYON.IS","AGHOL.IS","AKBNK.IS","AKCNS.IS","AKSA.IS","ALARK.IS","ALBRK.IS","ALFAS.IS",
"ALTIN.IS","ALTNY.IS","ANSGR.IS","ARCLK.IS","ASELS.IS","ASTOR.IS","AYGAZ.IS","AVHOL.IS","BAGFS.IS","BALAT.IS",
"BALSU.IS","BIMAS.IS","BJKAS.IS","BMSCH.IS","BRSAN.IS","BRYAT.IS","BSOKE.IS","BUCIM.IS","CANTE.IS","CCOLA.IS",
"CIMSA.IS","CLEBI.IS","CIMSA.IS","CWENE.IS","DOAS.IS","DOHOL.IS","DAGI.IS","ECILC.IS","EFORC.IS","EGEEN.IS",
"EKGYO.IS","ENJSA.IS","ENKAI.IS","EREGL.IS","FENER.IS","FROTO.IS","GARAN.IS","GENIL.IS","GESAN.IS","GLRMK.IS",
"GRSEL.IS","GRTHO.IS","GSRAY.IS","GUBRF.IS","HALKB.IS","HEKTS.IS","IEYHO.IS","IPEKE.IS","ISCTR.IS","ISMEN.IS",
"KCHOL.IS","KCAER.IS","KOZAA.IS","KOZAL.IS","KRDMD.IS","KUYAS.IS","LMKDC.IS","MAGEN.IS","MAVI.IS","MGROS.IS",
"MIATK.IS","MPARK.IS","OBAMS.IS","ODAS.IS","OTKAR.IS","OYAKC.IS","PASEU.IS","PETKM.IS","PGSUS.IS","RALYH.IS",
"REEDR.IS","SAHOL.IS","SASA.IS","SISE.IS","SKBNK.IS","SMRTG.IS","SOKM.IS","TABGD.IS","TAVHL.IS","TCELL.IS",
"THYAO.IS","TKFEN.IS","TOASO.IS","TTRAK.IS","TUPRS.IS","TUREX.IS","TURSG.IS","ULKER.IS","VAKBN.IS","YKBNK.IS",
"ZOREN.IS","ZOREL.IS"
]

# Dil seçenekleri
TEXTS = {
    'en': {'title':'BIST100 Stock Analysis', 'time_interval':'Time Interval', 'start_analysis':'Start Analysis',
           'current_price':'Current Price','expected_price':'Expected Price','expected_increase_percentage':'Expected Increase Percentage',
           'rsi_14':'RSI 14','macd_line':'MACD Line','macd_signal':'MACD Signal','entry_price':'Entry Price','take_profit_price':'Take Profit Price',
           'stop_loss_price':'Stop Loss Price','signal_comment':'Signal Comment','download_csv':'Download CSV Results','debug_info':'Debug Info'},
    'tr': {'title':'BIST100 Hisse Analizi', 'time_interval':'Zaman Aralığı','start_analysis':'Analiz Başlat',
           'current_price':'Mevcut Fiyat','expected_price':'Beklenen Fiyat','expected_increase_percentage':'Beklenen Artış Yüzdesi',
           'rsi_14':'RSI 14','macd_line':'MACD Çizgisi','macd_signal':'MACD Sinyali','entry_price':'Giriş Fiyatı','take_profit_price':'Kar Alma Fiyatı',
           'stop_loss_price':'Zarar Durdur Fiyatı','signal_comment':'Sinyal Yorumu','download_csv':'CSV Sonuçlarını İndir','debug_info':'Debug Bilgisi'}
}

# Teknik göstergeler ve diğer fonksiyonlar (calculate_indicators, calculate_support_resistance, generate_signals, forecast_next_price, calculate_expected_price, calculate_trade_levels, plot_to_png, process_stock)
# ... (önceki kodla aynı, buraya olduğu gibi eklenir) ...

# Streamlit main fonksiyonu
def main():
    global language
    language = st.selectbox('Select Language / Dil Seçin',['en','tr'])
    st.title(TEXTS[language]['title'])
    interval = st.selectbox(TEXTS[language]['time_interval'],['1d','1wk'],index=0)
    rsi_lower = st.slider('RSI Lower Bound / RSI Alt Sınır',30,70,30)
    rsi_upper = st.slider('RSI Upper Bound / RSI Üst Sınır',50,90,70)
    min_expected_increase = st.slider('Min Expected Increase % / Minimum Beklenen Artış %',0,20,5)
    start_button = st.button(TEXTS[language]['start_analysis'])
    if not start_button: return

    results=[]
    with st.spinner('Analyzing...'):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_stock,s,interval,rsi_lower,rsi_upper,min_expected_increase) for s in BIST100_SYMBOLS]
            for future in as_completed(futures):
                res=future.result()
                if res: results.append(res)

    if results:
        # AL sinyali olanlar üstte, diğerleri aşağı
        signals = [r for r in results if r['signal_comment']=='BUY']
        no_signals = [r for r in results if r['signal_comment']!='BUY']
        sorted_results = signals + no_signals

        st.write(f"Total stocks analyzed: {len(results)}")
        for res in sorted_results:
            color = "red" if res['signal_comment']=='BUY' else "black"
            with st.expander(f"<span style='color:{color}'>{res['coin_name']} - {res['signal_comment']}</span>", unsafe_allow_html=True):
                st.write(f"{TEXTS[language]['current_price']}: {res['price']}")
                st.write(f"{TEXTS[language]['expected_price']}: {res['expected_price']}")
                st.write(f"{TEXTS[language]['expected_increase_percentage']}: {res['expected_increase_percentage']:.2f}%")
                st.write(f"{TEXTS[language]['rsi_14']}: {res['rsi_14']:.2f}, {TEXTS[language]['macd_line']}: {res['macd_line']:.4f}, {TEXTS[language]['macd_signal']}: {res['macd_signal']:.4f}")
                st.image(res['plot'])
                with st.expander(TEXTS[language]['debug_info']):
                    st.json(res['debug'])

        # CSV indirme
        df_res=pd.DataFrame(results)
        csv=df_res.to_csv(index=False)
        b64=base64.b64encode(csv.encode()).decode()
        href=f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">{TEXTS[language]["download_csv"]}</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.write("No stocks analyzed.")

if __name__=="__main__":
    main()
