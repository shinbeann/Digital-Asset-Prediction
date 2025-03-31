from dotenv import load_dotenv
import os
import requests
import pandas as pd
import logging
from datetime import datetime
from time import sleep
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv(dotenv_path='../.env')
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("CryptoPanic_API_KEY")
BASE_URL = "https://cryptopanic.com//api/v1/posts/"

retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=['GET']
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

def fetch_news_data(start_date, end_date, page=1):
    """Fetch news data from CryptoPanic API."""
    url = f"{BASE_URL}?auth_token=b6b6faa06a397aa66e09e34f62682fed7aa8ac32&filter=crypto&before={end_date}&after={start_date}&page={page}&public=true"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None

def parse_news_data(data):
    """Parse news data to extract relevant information (sentiment, panic score, etc.)."""
    news_data = []
    
    for post in data['results']:
        title = post.get('title', '')
        url = post.get('url', '')
        sentiment = post.get('sentiment', '')
        panic_score = post.get('panic_score', None)  
        published_at = post.get('published_at', '')
        
        
        try:
            published_at = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%S%z').strftime('%Y-%m-%d %H:%M:%S')
        except:
            logging.warning(f"Error parsing date: {e}")
            published_at = None
        
        # extract cryptocurrency
        currencies = post.get('currencies', [])
        crypto_types = [currency.get('code', '') for currency in currencies]

        news_data.append({
            "Title": title,
            "URL": url,
            "Sentiment": sentiment,
            "PanicScore": panic_score,
            "Published At": published_at,
            "Cryptocurrencies": ", ".join(crypto_types)
        })
    
    return news_data

def save_to_csv(news_data, filename="crypto_news.csv"):
    """Save the news data to a CSV file."""
    df = pd.DataFrame(news_data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    logging.info(f"News data saved to {filename}")

def scrape_crypto_news(start_date, end_date):
    """Main function to scrape CryptoPanic news and save data."""
    page = 1
    all_news_data = []
    
    while True:
        data = fetch_news_data(start_date, end_date, page)
        
        if data and data['results']:
            news_data = parse_news_data(data)
            all_news_data.extend(news_data)
            page += 1  
            sleep(1)
        else:
            break  
        
    if all_news_data:
        save_to_csv(all_news_data)
    else:
        logging.info("No news data found.")

start_date = "2022-03-24"  
end_date = "2025-03-24"    


scrape_crypto_news(start_date, end_date)
