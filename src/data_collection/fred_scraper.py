####################
# Required Modules #
####################

# Generic/Built-in
import os
import requests
import pandas as pd


def fetch_fred_data(
    series_id: str, 
    start_date: str = "2022-03-24", 
    end_date: str = "2025-03-24", 
    output_file: str ="fred_data.csv",
    api_key: str = os.getenv("FRED_API_KEY")
):
    """
    Fetches historical economic data from the FRED API and saves it as a CSV file.

    Args:
        series_id (str): The FRED series ID for the dataset.
        start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to "2022-03-24".
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to "2025-03-24".
        output_file (str, optional): Name of the output CSV file. Defaults to "fred_data.csv".
        api_key (str, optional): Your FRED API key. Defaults to fetching the "FRED_API_KEY" environment variable value.
    """
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "api_key": api_key,
        "file_type": "json"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "observations" in data:
        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors='coerce')
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
    else:
        print("Failed to retrieve data", data)