# app.py
import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import r2_score
from src.models import CryptoInformer  # Your model class
from src.dataset import Normalizer, CryptoDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
                'prev_close': float(window.iloc[-1]['close']),
                'symbol':str(symbol),
                'predicted_close': float(pred),
                'actual_close': float(actual_close),
            })

    # Display results
    if crypto_inputs:
        results_df = pd.DataFrame(crypto_inputs)
        results_df = results_df.sort_values(by='predicted_close', ascending=False)
        r2 = r2_score(results_df['actual_close'], results_df['predicted_close'])

        y_true = results_df['actual_close']
        y_pred = results_df['predicted_close']

        mae = mean_absolute_error(y_true, y_pred)
    

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


        results_df = results_df[results_df['percent_of_coins'] > 0]
        results_df = results_df.sort_values(by='predicted_close', ascending=False)

        st.subheader(f"Recommended Portfolio Allocation for {selected_date.strftime('%Y-%m-%d')}")
        
        col_table, col_chart = st.columns([1, 2])
        
        with col_table:
            # Display the formatted table
            st.dataframe(
                results_df[['symbol', 'percent_of_coins']]
                .rename(columns={
                    'symbol': 'Cryptocurrency',
                    'percent_of_coins': 'Allocation %'
                })
                .style.format({'Allocation %': '{:.0f}%'})
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('text-align', 'center')]
                }])
            )
            st.write(f"*Total allocation: {results_df['percent_of_coins'].sum()}%*")

        with col_chart:
            # Configure plot style
            rcParams['font.size'] = 10
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create custom colors
            colors = plt.cm.tab20c(np.linspace(0, 1, len(results_df)))
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                results_df['percent_of_coins'],
                labels=None,  # We'll use legend instead
                colors=colors,
                startangle=90,
                wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
                pctdistance=0.8,
                autopct=lambda p: f'{p:.1f}%' if p >= 5 else ''
            )
            
            # Style percentage labels
            plt.setp(autotexts, size=9, weight="bold", color='black')
            
            # Add legend with symbols and allocations
            legend_labels = [
                f"{row['symbol']} ({row['percent_of_coins']}%)" 
                for _, row in results_df.iterrows()
            ]
            
            ax.legend(
                wedges,
                legend_labels,
                title="Cryptocurrencies",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False
            )
            
            ax.set_title('Portfolio Allocation', pad=20)
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Model Performance")
    
        st.markdown(f"""
        **R² Score:** {r2:.3f}  
        **MAE (Mean Absolute Error):** {mae:.2f}  

        *Lower MAE indicate better predictions. R² closer to 1 is ideal.*
        """)

    else:
        st.warning("Not enough data to make predictions for the selected date.")