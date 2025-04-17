import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("C:/Users/CH585/Documents/T6/CDS/50.038 project") 
PREDICTION_PATH = PROJECT_ROOT / "final_predictions.csv"

# 50 tickers
TICKERS = [
    'ADA/USDT', 'ALGO/USDT', 'ANKR/USDT', 'ARDR/USDT', 'ARPA/USDT', 'ATOM/USDT',
    'BAL/USDT', 'BAND/USDT', 'BAT/USDT', 'BCH/USDT', 'BNB/USDT', 'BNT/USDT',
    'BTC/USDT', 'CELR/USDT', 'CHR/USDT', 'CHZ/USDT', 'COMP/USDT', 'COS/USDT',
    'COTI/USDT', 'CRV/USDT', 'CTSI/USDT', 'CTXC/USDT', 'CVC/USDT', 'DASH/USDT',
    'DATA/USDT', 'DCR/USDT', 'DENT/USDT', 'DGB/USDT', 'DOGE/USDT', 'DOT/USDT',
    'DUSK/USDT', 'ENJ/USDT', 'EOS/USDT', 'ETC/USDT', 'ETH/USDT', 'EUR/USDT',
    'FET/USDT', 'FTT/USDT', 'FUN/USDT', 'HBAR/USDT', 'HIVE/USDT', 'HOT/USDT',
    'ICX/USDT', 'IOST/USDT', 'IOTA/USDT', 'IOTX/USDT', 'JST/USDT', 'KAVA/USDT',
    'KMD/USDT', 'KNC/USDT', 'LINK/USDT', 'LRC/USDT', 'LSK/USDT', 'LTC/USDT',
    'LTO/USDT', 'LUNA/USDT', 'MANA/USDT', 'MBL/USDT', 'MDT/USDT', 'MKR/USDT',
    'MTL/USDT', 'NEO/USDT', 'NKN/USDT', 'NMR/USDT', 'NULS/USDT', 'OGN/USDT',
    'ONE/USDT', 'ONG/USDT', 'ONT/USDT', 'PAXG/USDT', 'QTUM/USDT', 'RLC/USDT',
    'RSR/USDT', 'RVN/USDT', 'SAND/USDT', 'SC/USDT', 'SNX/USDT', 'SOL/USDT',
    'STORJ/USDT', 'STPT/USDT', 'STX/USDT', 'SXP/USDT', 'TFUEL/USDT', 'THETA/USDT',
    'TROY/USDT', 'TRX/USDT', 'TUSD/USDT', 'USDC/USDT', 'VET/USDT', 'VTHO/USDT',
    'WAN/USDT', 'WIN/USDT', 'XLM/USDT', 'XRP/USDT', 'XTZ/USDT', 'YFI/USDT',
    'ZEC/USDT', 'ZEN/USDT', 'ZIL/USDT', 'ZRX/USDT'
]

def load_predictions():
    df = pd.read_csv(PREDICTION_PATH)
    if 'symbol' not in df.columns or 'percent_of_coins' not in df.columns:
        raise ValueError("CSV must contain 'symbol' and 'percent_of_coins' columns.")
    return df

predictions_df = load_predictions()

# UI
st.title("Crypto Purchase Predictor")
st.markdown("Select any number of tickers to view predicted purchase percentages for March 24, 2025.")

# Ticker selector
selected_tickers = st.multiselect("Select one or more tickers", options=TICKERS)

if selected_tickers:
    st.subheader("Suggested Purchase % on March 24, 2025")
    
    # Filter the dataframe based on selected tickers
    filtered_df = predictions_df[predictions_df['symbol'].isin(selected_tickers)].copy()

    # normalised data
    total = filtered_df['percent_of_coins'].sum()
    filtered_df['rescaled_percent'] = (filtered_df['percent_of_coins'] / total) * 100
    
    filtered_df = filtered_df.sort_values(by='rescaled_percent', ascending=False)
    
    st.dataframe(
        filtered_df[['symbol', 'rescaled_percent']]
        .rename(columns={'rescaled_percent': 'Suggested % to Purchase'})
        .style.format({'Suggested % to Purchase': "{:.2f}"})
    )

else:
    st.info("Please select at least one ticker to see predictions.")
