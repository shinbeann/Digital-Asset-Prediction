from dotenv import load_dotenv
import os
import time
import logging
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def setup_driver():
    """initialise Selenium WebDriver instance."""
    options = Options()
    options.add_argument("--headless")  
    options.add_argument("--start-maximized")  
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-popup-blocking")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver


def scrape_google_news(query, start_date, end_date, output_file):
    """Scrape Google News"""
    logging.info(f"Scraping Google News for query: {query}")

    google_news_url = f"https://www.google.com/search?q={query}+after:{start_date}+before:{end_date}&tbm=nws"

    driver = setup_driver()
    driver.get(google_news_url)
    time.sleep(5) 

    articles = driver.find_elements(By.XPATH, "//div[@class='SoaBEf']")
    news_data = []

    for article in articles:
        try:
            title_element = article.find_element(By.XPATH, ".//div[@role='heading']")
            title = title_element.text if title_element else ""
            
            link_element = article.find_element(By.TAG_NAME, "a")
            link = link_element.get_attribute("href") if link_element else ""

            snippet_element = article.find_element(By.CLASS_NAME, "GI74Re") 
            snippet = snippet_element.text.strip() if snippet_element else ""

            source_element = article.find_element(By.CLASS_NAME, "NUnG9d")  
            source = source_element.text.strip()  if source_element else ""

            date_element = article.find_element(By.CLASS_NAME, "LfVVr")  
            date = date_element.text.strip() if date_element else ""

            news_data.append({
                "Title": title,
                "Snippet": snippet,
                "Source": source,
                "Publication Date": date,
                "URL": link
            })
        except Exception as e:
            logging.error(f"Error extracting article data: {e}")

    driver.quit()

    df = pd.DataFrame(news_data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"Google News data saved to {output_file}")


def scrape_fear_greed_index(output_file):
    """Scrape Fear & Greed Index from alternative.me"""
    logging.info("Scraping Fear & Greed Index")

    url = "https://api.alternative.me/fng/?limit=0"  
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()["data"]  

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")  

        df = df.rename(columns={"value": "index_value", "value_classification": "sentiment", "timestamp": "date"})

        df.to_csv(output_file, index=False)
        logging.info(f"Fear & Greed Index data saved to {output_file}")
    else:
        logging.error(f"Error fetching data: {response.status_code}")


def scrape_social_media(query, start_date, end_date, output_file):
    """Scrape social media sentiment from Twitter API"""
    logging.info(f"Scraping Twitter for query: {query}")

    auth = tweepy.OAuthHandler(os.getenv("TWITTER_CONSUMER_KEY"), os.getenv("TWITTER_CONSUMER_SECRET"))
    auth.set_access_token(os.getenv("TWITTER_ACCESS_TOKEN"), os.getenv("TWITTER_ACCESS_TOKEN_SECRET"))
    api = tweepy.API(auth)

    tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", since=start_date, until=end_date).items(100)

    social_media_data = []
    for tweet in tweets:
        try:
            tweet_text = tweet.text
            tweet_sentiment = analyze_sentiment(tweet_text) 
            
            social_media_data.append({
                "Tweet": tweet_text,
                "Sentiment": tweet_sentiment,
                "Date": tweet.created_at,
                "User": tweet.user.screen_name
            })
        except Exception as e:
            logging.error(f"Error processing tweet: {e}")

    df = pd.DataFrame(social_media_data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"Social media data saved to {output_file}")



def main():
    # scrape_google_news(query="crypto market", start_date="2022-03-24", end_date="2025-03-24", output_file="google_news_crypto.csv")
    # scrape_fear_greed_index(output_file="fear_greed_index.csv")
    scrape_social_media(query="cryptocurrency", start_date="2022-03-24", end_date="2025-03-24", output_file="social_media_sentiment_crypto.csv")


if __name__ == "__main__":
    main()