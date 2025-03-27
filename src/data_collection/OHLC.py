import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BinanceOHLCScraper:
    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize Binance exchange connection
        
        Args:
            api_key (str, optional): Binance API key from .env
            api_secret (str, optional): Binance API secret from .env
        """
        # Fetch keys from environment variables if not provided
        api_key = api_key or os.getenv('KEY_1')
        api_secret = api_secret or os.getenv('KEY_2')
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'  # Use spot market by default
            }
        })

    def fetch_ohlcv_data(self, 
                          symbol, 
                          start_date='2022-03-24', 
                          end_date='2025-03-24', 
                          timeframe='1d'):
        """
        Fetch OHLCV data for a given symbol and date range
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            timeframe (str): Candle timeframe (default: daily)
        
        Returns:
            pandas.DataFrame: OHLCV data
        """
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Initialize empty list to store all OHLCV data
        ohlcv_data = []
        
        # Fetch data in chunks to avoid API limitations
        current_start = start_timestamp
        while current_start < end_timestamp:
            try:
                # Fetch 500 candles at a time (Binance limit)
                candles = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe, 
                    since=current_start,
                    limit=500
                )
                
                # Break if no more data
                if not candles:
                    break
                
                # Add to data list
                ohlcv_data.extend(candles)
                
                # Update start timestamp for next iteration
                current_start = candles[-1][0] + 1
                
                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter date range
        df = df.loc[start_date:end_date]
        
        return df

    def fetch_top_symbols(self, limit=100):
        """
        Fetch top trading symbols by volume
        
        Args:
            limit (int): Number of top symbols to return
        
        Returns:
            list: Top trading symbols
        """
        try:
            # Load markets
            self.exchange.load_markets()
            
            # Sort markets by daily volume
            markets = sorted(
                self.exchange.markets.values(), 
                key=lambda x: x.get('quote', 'USDT') == 'USDT' and x.get('active', False),
                reverse=True
            )
            
            # Filter USDT pairs and get top symbols
            usdt_pairs = [
                market['symbol'] for market in markets 
                if market['quote'] == 'USDT' and market['active']
            ]
            
            return usdt_pairs[:limit]
        
        except Exception as e:
            print(f"Error fetching top symbols: {e}")
            return []

def main():
    # Initialize scraper using environment variables
    scraper = BinanceOHLCScraper()
    
    # Get top trading symbols
    top_symbols = scraper.fetch_top_symbols(limit=100)
    print(f"Fetching data for {len(top_symbols)} top symbols")
    
    # Dictionary to store all data
    all_data = {}
    
    # Fetch data for each symbol
    for symbol in top_symbols:
        try:
            print(f"Fetching data for {symbol}")
            df = scraper.fetch_ohlcv_data(symbol)
            
            if not df.empty:
                all_data[symbol] = df
                print(f"✓ Collected {len(df)} days of data for {symbol}")
            
            # Optional: Add a small delay between symbol fetches
            time.sleep(1)
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Combine all data
    if all_data:
        # Create a MultiIndex DataFrame
        final_df = pd.concat(all_data.values(), keys=all_data.keys(), names=['symbol'])
        
        # Save to CSV
        output_filename = f"binance_ohlc_data_{datetime.now().strftime('%Y%m%d')}.csv"
        final_df.to_csv(output_filename)
        
        print(f"\n✅ Data collection complete. Saved to {output_filename}")
        print(f"Symbols collected: {len(all_data)}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()