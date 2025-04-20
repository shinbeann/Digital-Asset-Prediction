# app.py
import streamlit as st
import pandas as pd
import torch
import numpy as np

from src.models import CryptoInformer  # Your model class
from src.dataset import Normalizer, CryptoDataset

# Constants
TEST_PATH = "data/processed/test_set.csv"
MODEL_PATH = "saved_models/CryptoInformer/Best_R2.pth"
SEQUENCE_LENGTH = 14
# Normalization stats (from training set)
NORMALIZER_STATS = {
    'mean': torch.tensor([
        4.7167e+02, 4.8263e+02, 4.6039e+02, 4.7177e+02,
        4.3276e+08, 6.2134e+07, 2.2808e+00, 4.3239e+03,
        -4.2273e-01, 4.6805e+01, 1.9130e+03
    ]),
    'std': torch.tensor([
        3.6570e+03, 3.7372e+03, 3.5740e+03, 3.6608e+03,
        4.5931e+09, 4.1551e+08, 6.3604e+02, 4.3362e+02,
        3.1840e-01, 2.0331e+01, 1.3954e+02
    ])
}

# Load model
@st.cache_resource
def load_model():
    model = CryptoInformer()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

# Load data
@st.cache_data
def load_dataframe():
    df = pd.read_csv(TEST_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    return df

# Instantiate Normalizer
normalizer = Normalizer()
normalizer.mean = NORMALIZER_STATS['mean']
normalizer.std = NORMALIZER_STATS['std']

# Streamlit UI
st.title("Crypto Next-Day Price Predictor")

model, device = load_model()
df = load_dataframe()

available_dates = pd.to_datetime(df['date']).sort_values().unique()
valid_dates = available_dates[SEQUENCE_LENGTH:]  # Ensure enough past days

selected_date = st.date_input(
    "Select a prediction date",
    value=pd.to_datetime(valid_dates[-1]),
    min_value=valid_dates[0],
    max_value=valid_dates[-1]
)

if selected_date:
    selected_date = pd.to_datetime(selected_date)
    input_end_date = selected_date - pd.Timedelta(days=1)
    input_start_date = input_end_date - pd.Timedelta(days=SEQUENCE_LENGTH - 1)

    crypto_inputs = []
    crypto_symbols = []

    for symbol in df['symbol'].unique():
        crypto_df = df[df['symbol'] == symbol].sort_values('date')
        window = crypto_df[
            (crypto_df['date'] >= input_start_date) & 
            (crypto_df['date'] <= input_end_date)
        ]

        if len(window) == SEQUENCE_LENGTH:
            next_day = crypto_df[crypto_df['date'] == selected_date]
            if next_day.empty:
                continue  # No actual close price for day t

            feature_cols = [col for col in df.columns if col not in ['date', 'symbol']]
            sequence_features = window[feature_cols].values
            X = torch.tensor(sequence_features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, num_features)
            X = normalizer(X)
            
            model.eval()
            with torch.no_grad():
                pred = model(X).squeeze(0)

            actual_close = next_day.iloc[0]['close']

            crypto_inputs.append({
                'prev_close': window.iloc[-1]['close'],
                'symbol': symbol,
                'predicted_close': pred,
                'actual_close': actual_close,
            })

    # Display results
    if crypto_inputs:
        results_df = pd.DataFrame(crypto_inputs)
        results_df = results_df.sort_values(by='predicted_close', ascending=False)

        # Calculate percentage increase from previous close
        results_df['percentage_increase'] = (results_df['predicted_close'] - results_df['prev_close']) / results_df['prev_close'] * 100

        # Calculate portfolio allocation proportions
        total_percentage_increase = results_df['percentage_increase'].sum()
        results_df['proportion'] = results_df['percentage_increase'] / total_percentage_increase
        results_df['percent_of_coins'] = (results_df['proportion'] * 100).round().astype(int)
        
        # Adjust for rounding errors to ensure total equals 100%
        total_coins = results_df['percent_of_coins'].sum()
        if total_coins != 100:
            # Find the symbol with the largest rounding error
            difference = 100 - total_coins
            # Adjust the first row (which has the highest predicted close)
            results_df.iloc[0, results_df.columns.get_loc('percent_of_coins')] += difference

        st.subheader(f"Recommended Portfolio Allocation for {selected_date.strftime('%Y-%m-%d')}")
        
        # Display only symbol and allocation percentage
        st.dataframe(
            results_df[['symbol', 'percent_of_coins']]
            .rename(columns={
                'symbol': 'Cryptocurrency',
                'percent_of_coins': 'Allocation %'
            })
            .style.format({'Allocation %': '{:.0f}%'})
        )
        
        # Display total allocation (should always be 100%)
        st.write(f"Total allocation: {results_df['percent_of_coins'].sum()}%")
    else:
        st.warning("Not enough data to make predictions for the selected date.")