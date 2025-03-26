import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyse_sentiment(text):

    score = sia.polarity_scores(text)['compound']
    sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    return sentiment, score

def process_sentiment(input_file, output_file):
    
    df = pd.read_csv(input_file)
    df["Sentiment"], df["Sentiment Score"] = zip(*df["Text"].apply(analyse_sentiment))
    
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Sentiment analysis completed. Results saved to {output_file}.")


if __name__ == "__main__":
    input_file = "crypto_social_media.csv"
    output_file = "crypto_sentiment_analysis.csv"
    process_sentiment(input_file, output_file)
