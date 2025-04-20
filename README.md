# Digital Asset Prediction
The primary objective of this project is to forecast the future closing prices of digital assets in order to construct an optimal cryptocurrency portfolio. Specifically, we aim to develop a predictive model that, given a historical sequence of a cryptocurrency's data, estimates its closing price for the following day. Accurate forecasting of asset prices and market dynamics enables dynamic portfolio allocation strategies that seek to maximize returns while minimizing associated risks. To enhance prediction performance, the model integrates asset-specific features (such as open, high, low, and close prices, trading volume, and market capitalization) with macroeconomic indicators (including the S\&P 500 index and treasury yield spread) and sentiment metrics (such as the Fear and Greed Index), enabling more informed investment decisions.

## Table of Contents
- [Project Overview](#project-overview)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Project Overview
- `data/`: Contains raw (scraped) and processed datasets.
- `demo_results/`: Model results obtained from running `notebooks/demo_inference.ipynb`.
- `notebooks/`: Various notebooks that serve as environments for different processes.
  - `data_collection.ipynb`: Runs our functions for collecting raw data (but we have already provided the raw datasets in this directory).
  - `data_processing.ipynb`: Compile the raw datasets and preprocess them, before splitting them into train, validation, and test sets (can be found in `data/processed`). 
  - `model_training.ipynb`: Contains code for model training.
  - `demo_inference.ipynb`: Demo notebook for loading in the provided trained model parameters and running them on the test set.
- `results/`: Training metrics and performance on test set obtained from `model_training.ipynb` automatically saved here by default. 
- `saved_models/`: Contains trained model parameters (from running `model_training.ipynb`).
- `src/`:
    - `data_collection/`: Contains code for downloading raw datasets from online programmatically.
    - `dataset.py`: Contains code for our PyTorch dataset object.
    - `models.py`: Contains code for four deep learning architectures for sequence processing implemented in PyTorch - GRU, LSTM, Transformer, and Informer.
    - `train_eval.py`: Contains code for training and evaluation of models.
    - `utils.py`: Helper functions.
- `app.py`: GUI App code.

## Usage
This code was developed using Python 3.12.8.

1. Clone the repository:
    ```bash
    git clone https://github.com/shinbeann/Digital-Asset-Prediction
    cd Digital-Asset-Prediction
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the various notebooks for data collection, processing, training, and inference (see above).
4. The GUI can be ran using `streamlit run app.py`.
