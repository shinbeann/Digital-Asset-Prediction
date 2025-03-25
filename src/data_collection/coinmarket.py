import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# Function to get the top 100 cryptocurrencies by market cap
def get_top_assets():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Error fetching top assets:", response.json())
        return []
    data = response.json()
    
    # Extract coin IDs
    asset_list = [coin["id"] for coin in data]
    return asset_list

# Function to fetch daily price history (not OHLC)
def get_daily_market_data(asset_id, vs_currency="usd", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching {asset_id}: {response.status_code}")
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting longer...")
            time.sleep(60)  # Wait longer if rate limited
        return None
    
    data = response.json()
    if not data or "prices" not in data:
        print(f"No price data for {asset_id}")
        return None
    
    # Get price data (timestamps and prices)
    prices = data["prices"]
    volumes = data["total_volumes"]
    market_caps = data["market_caps"]
    
    # Create DataFrame
    df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
    df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
    df_market_caps = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
    
    # Convert timestamps and set as index
    df_prices["date"] = pd.to_datetime(df_prices["timestamp"], unit="ms").dt.date
    df_prices.set_index("date", inplace=True)
    
    df_volumes["date"] = pd.to_datetime(df_volumes["timestamp"], unit="ms").dt.date
    df_volumes.set_index("date", inplace=True)
    
    df_market_caps["date"] = pd.to_datetime(df_market_caps["timestamp"], unit="ms").dt.date
    df_market_caps.set_index("date", inplace=True)
    
    # Merge dataframes
    df = pd.DataFrame({
        "price": df_prices["price"],
        "volume": df_volumes["volume"],
        "market_cap": df_market_caps["market_cap"]
    })
    
    # Compute daily returns
    df["daily_return"] = df["price"].pct_change()
    
    return df

# Get the top 100 assets
print("Fetching top 100 cryptocurrency assets by market cap...")
top_assets = get_top_assets()
print(f"Found {len(top_assets)} assets")

# Dictionary to store data
all_data = {}

# Fetch daily data for each asset
for i, asset in enumerate(top_assets):
    try:
        print(f"[{i+1}/{len(top_assets)}] Fetching daily data for {asset}...")
        df = get_daily_market_data(asset)
        if df is not None:
            all_data[asset] = df
            print(f"✓ Successfully fetched {len(df)} days of data for {asset}")
        else:
            print(f"✗ No data retrieved for {asset}")
        
        # Adjust sleep time based on position in the list
        sleep_time = 6 if i % 10 == 0 else 3  # Longer pause every 10 requests
        print(f"Waiting {sleep_time} seconds to avoid API rate limits...")
        time.sleep(sleep_time)
    except Exception as e:
        print(f"Error fetching {asset}: {e}")

# Combine all data into a single DataFrame
print("\nCombining data from all assets...")
df_list = []
for asset, df in all_data.items():
    df_with_asset = df.copy()
    df_with_asset["asset"] = asset
    df_list.append(df_with_asset)

if df_list:
    final_df = pd.concat(df_list)
    
    # Save to CSV
    output_filename = f"top_crypto_daily_data_{datetime.now().strftime('%Y%m%d')}.csv"
    final_df.to_csv(output_filename)
    print(f"\n✅ Daily data collection complete! Data saved as '{output_filename}'.")
    
    # Print summary
    asset_count = len(all_data)
    total_records = len(final_df)
    print(f"\nSummary:")
    print(f"- Successfully collected data for {asset_count} cryptocurrencies")
    print(f"- Total records: {total_records}")
    print(f"- Date range: {final_df.index.min()} to {final_df.index.max()}")
else:
    print("\n❌ No data was collected. Please check your internet connection and API limits.")
    start_date = datetime.strptime('2024-03-24', '%Y-%m-%d').date()
    end_date = datetime.strptime('2025-03-24', '%Y-%m-%d').date()
    all_data = {}
    for i, asset in enumerate(top_assets):
        try:
            print(f"[{i+1}/{len(top_assets)}] Fetching daily data for {asset}...")
            df = get_daily_market_data(asset, days=(end_date - start_date).days)
            if df is not None:
                all_data[asset] = df
                print(f"✓ Successfully fetched {len(df)} days of data for {asset}")
            else:
                print(f"✗ No data retrieved for {asset}")
            
            # Adjust sleep time based on position in the list
            sleep_time = 6 if i % 10 == 0 else 3  # Longer pause every 10 requests
            print(f"Waiting {sleep_time} seconds to avoid API rate limits...")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error fetching {asset}: {e}")