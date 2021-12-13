import os 
import pandas as pd
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score

load_dotenv()
cog_endpoint = os.getenv("COG_SERVICE_ENDPOINT")
cog_key = os.getenv("COG_SERVICE_KEY")

data = pd.read_csv("../02-Movie-reviews/reviews.csv")
data["clean_reviews"] = data.reviews.str.replace(r"[^A-Za-z]", " ").str.lower().str.split().apply(" ".join)
data = data[(data.clean_reviews.str.len()) < 5120]

credential = AzureKeyCredential(cog_key)
client = TextAnalyticsClient(cog_endpoint, credential)

def analyze_sentiment_df(string: str):
    return client.analyze_sentiment([string])[0].sentiment

data["pred"] = data.clean_reviews.apply(analyze_sentiment_df)

print(accuracy_score(data.target, data.pred.map({"negative": "neg", "positive": "pos"})))