import pandas as pd
import nltk
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download("vader_lexicon")

file_path = os.path.abspath("Digital-Asset-Prediction/data/raw/google_news_crypto.csv")
df = pd.read_csv(file_path)
# print(df.head())

sia = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    score = sia.polarity_scores(text)
    if score["compound"] >= 0.05:
        return "Positive"
    elif score["compound"] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


df["VADER Sentiment"] = df["Title"].apply(analyze_sentiment_vader)
df["TextBlob Sentiment"] = df["Title"].apply(analyze_sentiment_textblob)

df.to_csv("google_news_crypto_with_sentiment.csv", index=False, encoding="utf-8-sig")

print("Sentiment analysis completed and saved to 'google_news_crypto_with_sentiment.csv'")
