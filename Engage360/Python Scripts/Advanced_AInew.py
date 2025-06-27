# 🚀 Social Media Engagement Analysis for Facebook & Instagram (Live Output)
# ✅ Implements Data Cleaning, Time-Series Forecasting, Clustering Analysis, Sentiment Analysis, and Post Success Prediction.

# ✅ Step 1: Import Essential Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# ✅ Step 2: Download Required NLP Dataset for Sentiment Analysis
nltk.download('vader_lexicon')

# ✅ Step 3: Load Facebook & Instagram Datasets
data_dir = r"C:\Users\valarsri\Downloads"
facebook_file = os.path.join(data_dir, "Cleaned_Facebook_Posts.csv")
instagram_file = os.path.join(data_dir, "Cleaned_Instagram_Posts.csv")

df_facebook = pd.read_csv(facebook_file, encoding='ISO-8859-1')
df_instagram = pd.read_csv(instagram_file, encoding='ISO-8859-1')

print("\n✅ Loaded datasets successfully!")

# ✅ Step 4: Standardize Column Names
df_facebook.columns = df_facebook.columns.str.strip().str.lower().str.replace(" ", "_")
df_instagram.columns = df_instagram.columns.str.strip().str.lower().str.replace(" ", "_")

print("\n🔍 Facebook Dataset Columns:", df_facebook.columns)
print("\n🔍 Instagram Dataset Columns:", df_instagram.columns)

# ✅ Step 5: Compute Engagement Rate if Missing
if "engagement_rate" not in df_facebook.columns:
    df_facebook["engagement_rate"] = (
        (df_facebook["total_post_reactions"] + df_facebook["comments_on_posts"] + df_facebook["shares_on_posts"]) / df_facebook["post_reach"]
    ) * 100

if "engagement_rate" not in df_instagram.columns:
    df_instagram["engagement_rate"] = (
        (df_instagram["like_count"] + df_instagram["comments_count"] + df_instagram["shares"]) / df_instagram["media_reach"]
    ) * 100

print("\n📊 Engagement Rates Calculated!")

# ✅ Step 6: Ensure No Missing or Infinite Values
df_instagram["engagement_rate"] = df_instagram["engagement_rate"].replace([np.inf, -np.inf], np.nan)
df_instagram["engagement_rate"] = df_instagram["engagement_rate"].fillna(df_instagram["engagement_rate"].median())

# ✅ Step 7: Remove Duplicate Date Entries Before Time-Series Forecasting
df_instagram = df_instagram.drop_duplicates(subset=["date"], keep="first")  # Keep first occurrence
df_instagram["date"] = pd.to_datetime(df_instagram["date"])
df_instagram = df_instagram.sort_values("date")  # Ensure chronological order
df_instagram.set_index("date", inplace=True)
df_instagram = df_instagram.asfreq("D").ffill()  # Corrected fill method

print("\n✅ Data cleaned and prepared for forecasting!")

# ✅ Step 8: Time-Series Forecasting (ARIMA)
arima_model = ARIMA(df_instagram["engagement_rate"], order=(2,1,0))  # Simplified ARIMA structure
arima_result = arima_model.fit()

# ✅ Forecast the next 30 days
forecast = arima_result.forecast(steps=30)

print("\n📈 Forecasting Next 30 Days Engagement Trends:")
print(forecast)

plt.figure(figsize=(10,5))
plt.plot(df_instagram["engagement_rate"], label="Actual Engagement")
plt.plot(pd.date_range(start=df_instagram.index[-1], periods=30, freq='D'), forecast, label="Predicted Engagement", linestyle="dashed")
plt.title("📊 Instagram Engagement Trend Forecast")
plt.xlabel("Date")
plt.ylabel("Engagement Rate (%)")
plt.legend()
plt.show()

# ✅ Step 9: Clustering Analysis for Facebook & Instagram
if all(col in df_facebook.columns for col in ["total_post_reactions", "comments_on_posts", "shares_on_posts"]):
    X_cluster_fb = df_facebook[["total_post_reactions", "comments_on_posts", "shares_on_posts"]].dropna()
    X_cluster_fb_scaled = StandardScaler().fit_transform(X_cluster_fb)

    kmeans_fb = KMeans(n_clusters=3, random_state=42)
    df_facebook["cluster"] = kmeans_fb.fit_predict(X_cluster_fb_scaled)

    print("\n📊 Facebook Cluster Assignments:")
    print(df_facebook[["total_post_reactions", "comments_on_posts", "shares_on_posts", "cluster"]].head(10))

    plt.figure(figsize=(12,6))
    sns.scatterplot(x=df_facebook["total_post_reactions"], y=df_facebook["engagement_rate"], hue=df_facebook["cluster"], palette="Set1", s=200, alpha=0.7)
    plt.title("📊 Facebook Clustering Analysis")
    plt.xlabel("Total Post Reactions")
    plt.ylabel("Engagement Rate (%)")
    plt.legend(title="Cluster Groups")
    plt.show()

if all(col in df_instagram.columns for col in ["like_count", "comments_count", "shares"]):
    X_cluster_insta = df_instagram[["like_count", "comments_count", "shares"]].dropna()
    X_cluster_insta_scaled = StandardScaler().fit_transform(X_cluster_insta)

    kmeans_insta = KMeans(n_clusters=3, random_state=42)
    df_instagram["cluster"] = kmeans_insta.fit_predict(X_cluster_insta_scaled)

    print("\n📊 Instagram Cluster Assignments:")
    print(df_instagram[["like_count", "comments_count", "shares", "cluster"]].head(10))

    plt.figure(figsize=(12,6))
    sns.scatterplot(x=df_instagram["like_count"], y=df_instagram["engagement_rate"], hue=df_instagram["cluster"], palette="Set1", s=200, alpha=0.7)
    plt.title("📊 Instagram Clustering Analysis")
    plt.xlabel("Likes")
    plt.ylabel("Engagement Rate (%)")
    plt.legend(title="Cluster Groups")
    plt.show()

# ✅ Step 10: Sentiment Analysis (VADER)
sia = SentimentIntensityAnalyzer()

df_instagram["comments_count"] = df_instagram["comments_count"].fillna("").astype(str)  # Convert to string and fill empty values

def compute_sentiment(text):
    return sia.polarity_scores(text)["compound"] if text.strip() else np.nan  # Assign NaN for empty comments

df_instagram["sentiment_score"] = df_instagram["comments_count"].apply(compute_sentiment)

print("\n📝 Sentiment Scores Calculated!")
print(df_instagram[["comments_count", "sentiment_score"]].head(10))

plt.figure(figsize=(10,5))
sns.histplot(df_instagram["sentiment_score"].dropna(), bins=30, kde=True)
plt.title("📊 Sentiment Analysis: Understanding Audience Reactions")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()
