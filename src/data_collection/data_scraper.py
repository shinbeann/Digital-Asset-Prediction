####################
# Required Modules #
####################

# Generic/Built-in
import os
import requests
import pandas as pd
import time
from datetime import datetime
from typing import *


########################################
# Federal Reserve Economic Data (FRED) #
########################################

def fetch_data_from_fred(
    series_id: str, 
    start_date: str = "2022-03-24", 
    end_date: str = "2025-03-24", 
    output_filename: str = "fred_data.csv",
    api_key: str = os.getenv("FRED_API_KEY")
) -> pd.DataFrame:
    """
    Fetches historical economic data from the FRED API, saves it as a CSV file, and returns the data as a pandas 
    DataFrame.

    Args:
        series_id (str): The FRED series ID for the dataset.
        start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to "2022-03-24".
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to "2025-03-24".
        output_filename (str, optional): Name of the output CSV file. Defaults to "fred_data.csv".
        api_key (str, optional): Your FRED API key. Defaults to fetching the "FRED_API_KEY" environment variable value.

    Returns:
        pd.DataFrame: Retrieved data.
    """
    
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "api_key": api_key,
        "file_type": "json"
    }
    
    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        data = response.json()

        # Check if the response contains the expected data
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors='coerce')

            # Save the data to CSV
            df.to_csv(output_filename, index=False)
            print(f"Data saved to {output_filename}")
            return df
        else:
            print(f"Failed to retrieve data: {data}")
            return pd.DataFrame()  # Return an empty DataFrame in case of failure

    except requests.exceptions.RequestException as e:
        # Handle errors in the HTTP request
        print(f"Error fetching data from FRED API: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
        

#############
# CoinGecko #
#############

def get_top_assets_from_coingecko(limit: int = 100) -> List[str]:
    """
    Fetches the top cryptocurrencies by market cap.

    Args:
        limit (int, optional): Number of assets to fetch. Defaults to 100.

    Returns:
        List[str]: List of top cryptocurrency IDs.
    """

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd", # specific currency doesn't matter, we just need ordering
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Raise an error for non-2xx status codes
        
        # Extract coin IDs
        data = response.json()
        asset_list = [coin["id"] for coin in data]
        return asset_list
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching top assets: {e}")
        return []


def get_daily_market_data_of_asset_from_coingecko(
    asset_id: str,
    vs_currency: str = "usd",
    days: int = 365,
    output_file: Optional[str] = None,
    max_retries: int = 5,  
    retry_wait_time: int = 60
) -> Optional[pd.DataFrame]:
    """
    Fetches daily market data for a specific cryptocurrency.

    Args:
        asset_id (str): The coin's unique ID (e.g., 'bitcoin').
        vs_currency (str, optional): The currency in which to express the data. Defaults to "usd".
        days (int, optional): The number of days of data to fetch. Defaults to 365.
        output_file (Optional[str], optional): Path to output the data as a CSV file. Defaults to None (won't save).
        max_retries (int, optional): The maximum number of retry attempts in case of rate limits. Defaults to 5.
        retry_wait_time (int, optional): Wait time in seconds before retrying the request after rate limit exceeded.
            Defaults to 60.

    Returns:
        Optional[pd.DataFrame]: DataFrame with daily market data (price, volume, market cap, returns), or None if data 
            couldn't be fetched.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}

    retry_count = 0

    while retry_count < max_retries:
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raise an error for non-2xx status codes
            
            data = response.json()
            if not data or "prices" not in data:
                print(f"No price data for {asset_id}")
                return None

            # Get price, volume, and market cap data
            prices = data["prices"]
            volumes = data["total_volumes"]
            market_caps = data["market_caps"]
            
            # Create DataFrames
            df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
            df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            df_market_caps = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
            
            # Convert timestamps to datetime and set them as the index
            df_prices["date"] = pd.to_datetime(df_prices["timestamp"], unit="ms").dt.date
            df_prices.set_index("date", inplace=True)
            
            df_volumes["date"] = pd.to_datetime(df_volumes["timestamp"], unit="ms").dt.date
            df_volumes.set_index("date", inplace=True)
            
            df_market_caps["date"] = pd.to_datetime(df_market_caps["timestamp"], unit="ms").dt.date
            df_market_caps.set_index("date", inplace=True)
            
            # Merge DataFrames
            df = pd.DataFrame({
                "price": df_prices["price"],
                "volume": df_volumes["volume"],
                "market_cap": df_market_caps["market_cap"]
            })
            
            # Compute daily returns
            df["daily_return"] = df["price"].pct_change()
            
            # Optionally save to a CSV file
            if output_file:
                df.to_csv(output_file)
                print(f"Data saved to {output_file}")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {asset_id} data: {e}")
            if response.status_code == 429:  # Rate limit exceeded
                retry_count += 1
                print(f"Rate limit exceeded. Retrying in {retry_wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(retry_wait_time)  # Wait before retrying
            else:
                return None

    print(f"Max retries reached for {asset_id}. Could not fetch data.")
    return None

def fetch_top_crypto_data_from_coingecko(
    limit: int = 100,
    vs_currency: str = "usd",
    days: int = 365,
    output_filename: Optional[str] = None,
    max_retries: int = 5,  
    retry_wait_time: int = 10 
) -> Optional[pd.DataFrame]:
    """
    Fetches daily market data for the top cryptocurrency assets by market cap for a specified number of days and saves 
    the data to a CSV file.

    Args:
        limit (int, optional): The number of top assets to fetch. Defaults to 100.
        vs_currency (str, optional): The currency for market data. Defaults to "usd".
        days (int, optional): Number of days of historical data to fetch. Defaults to 365.
        output_filename (Optional[str], optional): Name of the output CSV file. Defaults to None, which will generate a 
            filename based on the current date.
        max_retries (int, optional): The maximum number of retry attempts in case of rate limits. Defaults to 5.
        retry_wait_time (int, optional): Wait time in seconds before retrying the request after rate limit exceeded.
            Defaults to 10.

    Returns:
        Optional[pd.DataFrame]: Combined data as a DataFrame if data is collected, else None.
    """
    # Fetch the IDs of top assets
    print(f"Fetching top {limit} cryptocurrency assets by market cap...")
    top_assets = get_top_assets_from_coingecko(limit=limit)
    print(f"Found {len(top_assets)} assets")

    # Dictionary to store data
    all_data: Dict[str, pd.DataFrame] = {}

    # Fetch daily data for each asset
    for i, asset_id in enumerate(top_assets):
        try:
            print(f"[{i+1}/{len(top_assets)}] Fetching daily data for {asset_id}...")
            df = get_daily_market_data_of_asset_from_coingecko(
                asset_id=asset_id, 
                vs_currency=vs_currency,
                days=days,
                max_retries=max_retries,
                retry_wait_time=retry_wait_time
            )
            if df is not None:
                df["asset"] = asset_id  # Add asset ID as a feature directly in the fetched data
                all_data[asset_id] = df
                print(f"✅ Successfully fetched {len(df)} days of data for {asset_id}")
            else:
                print(f"❌ No data retrieved for {asset_id}")
            
            # Adjust sleep time based on position in the list
            sleep_time = 6 if i % 10 == 0 else 3  # Longer pause every 10 requests
            print(f"Waiting {sleep_time} seconds to avoid API rate limits...")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error fetching {asset_id}: {e}")

    # Combine all data into a single DataFrame
    print("\nCombining data from all assets...")
    
    df_list = list(all_data.values())       
    if df_list:
        final_df = pd.concat(df_list)
        
        # Save to CSV
        if output_filename is None:
            output_filename = f"data/raw/top_crypto_daily_data_{datetime.now().strftime('%Y%m%d')}.csv"
        
        final_df.to_csv(output_filename)
        print(f"\n✅ Daily data collection complete! Data saved as '{output_filename}'.")
        
        # Print summary
        asset_count = len(all_data)
        total_records = len(final_df)
        print(f"\nSummary:")
        print(f"- Successfully collected data for {asset_count} cryptocurrencies")
        print(f"- Total records: {total_records}")
        print(f"- Date range: {final_df.index.min()} to {final_df.index.max()}")
        return final_df
    else:
        print("\n❌ No data was collected. Please check your internet connection and API limits.")
        return None