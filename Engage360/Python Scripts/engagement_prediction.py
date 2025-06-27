# ✅ Step 1: Import Essential Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ✅ Step 2: Load Cleaned Facebook & Instagram Datasets
data_dir = r"C:\Users\valarsri\Downloads"  # 🔹 Define the directory where datasets are stored
facebook_file = os.path.join(data_dir, "Cleaned_Facebook_Posts.csv")
instagram_file = os.path.join(data_dir, "Cleaned_Instagram_Posts.csv")

# ✅ Load CSV files with proper encoding handling
df_facebook = pd.read_csv(facebook_file, encoding='ISO-8859-1')
df_instagram = pd.read_csv(instagram_file, encoding='ISO-8859-1')

# ✅ Step 3: Standardize Column Names
df_facebook.columns = df_facebook.columns.str.strip().str.lower().str.replace(" ", "_")
df_instagram.columns = df_instagram.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Step 4: Ensure Engagement Rate Exists in Both Datasets
if "engagement_rate" not in df_facebook.columns:
    df_facebook["engagement_rate"] = (
        (df_facebook["total_post_reactions"] + df_facebook["comments_on_posts"] + df_facebook["shares_on_posts"]) / df_facebook["post_reach"]
    ) * 100

if "engagement_rate" not in df_instagram.columns:
    df_instagram["engagement_rate"] = (
        (df_instagram["like_count"] + df_instagram["comments_count"] + df_instagram["shares"]) / df_instagram["media_reach"]
    ) * 100

# ✅ Step 5: Compute Feature Correlations for Selection
corr_fb = df_facebook.select_dtypes(include=['number']).corr()["engagement_rate"].sort_values(ascending=False)
corr_insta = df_instagram.select_dtypes(include=['number']).corr()["engagement_rate"].sort_values(ascending=False)

# ✅ Step 6: Select Best Features Automatically
best_features_fb = corr_fb[corr_fb > 0.5].index.tolist()
best_features_insta = corr_insta[corr_insta > 0.5].index.tolist()

# ✅ Step 7: Prepare Data for Prediction Model
X_fb_selected = df_facebook[best_features_fb].dropna()
X_insta_selected = df_instagram[best_features_insta].dropna()

# ✅ Step 8: Remove Infinity & Clip Extreme Values
X_fb_selected = X_fb_selected.replace([np.inf, -np.inf], np.nan)
X_insta_selected = X_insta_selected.replace([np.inf, -np.inf], np.nan)

X_fb_selected = X_fb_selected.fillna(X_fb_selected.median())
X_insta_selected = X_insta_selected.fillna(X_insta_selected.median())

X_fb_selected = X_fb_selected.clip(lower=X_fb_selected.quantile(0.01), upper=X_fb_selected.quantile(0.99), axis=1)
X_insta_selected = X_insta_selected.clip(lower=X_insta_selected.quantile(0.01), upper=X_insta_selected.quantile(0.99), axis=1)

# ✅ Step 9: Scale Features for Better Predictions
scaler = StandardScaler()
X_fb_scaled = scaler.fit_transform(X_fb_selected)
X_insta_scaled = scaler.fit_transform(X_insta_selected)

# ✅ Step 10: Train & Evaluate Linear Regression Models
X_train_fb, X_test_fb, y_train_fb, y_test_fb = train_test_split(X_fb_scaled, df_facebook["engagement_rate"].dropna(), test_size=0.2, random_state=42)
X_train_insta, X_test_insta, y_train_insta, y_test_insta = train_test_split(X_insta_scaled, df_instagram["engagement_rate"].dropna(), test_size=0.2, random_state=42)

# ✅ Ensure y_train_insta contains valid numeric values
y_train_insta = y_train_insta.replace([np.inf, -np.inf], np.nan)
y_train_insta = y_train_insta.fillna(y_train_insta.median())
y_train_insta = y_train_insta.clip(lower=y_train_insta.quantile(0.01), upper=y_train_insta.quantile(0.99))

# ✅ Train Models
model_fb = LinearRegression()
model_insta = LinearRegression()

model_fb.fit(X_train_fb, y_train_fb)
model_insta.fit(X_train_insta, y_train_insta)

df_facebook["predicted_engagement"] = model_fb.predict(X_fb_scaled)
df_instagram["predicted_engagement"] = model_insta.predict(X_insta_scaled)

# ✅ Step 11: FINAL FIX - Remove Infinite Values from Engagement Rate
df_instagram["engagement_rate"] = df_instagram["engagement_rate"].replace([np.inf, -np.inf], np.nan)
df_instagram["engagement_rate"] = df_instagram["engagement_rate"].fillna(df_instagram["engagement_rate"].median())
df_instagram["engagement_rate"] = df_instagram["engagement_rate"].clip(
    lower=df_instagram["engagement_rate"].quantile(0.01),
    upper=df_instagram["engagement_rate"].quantile(0.99)
)

# ✅ Step 12: Debugging Check Before Model Evaluation
print("\n🔍 Final Debugging Check Before Model Evaluation...")
print("\n📊 Missing Values in `engagement_rate`:", df_instagram["engagement_rate"].isnull().sum())
print("\n📊 Infinite Values in `engagement_rate`:", df_instagram["engagement_rate"].replace([np.inf, -np.inf], np.nan).isnull().sum())

print("\n📊 Maximum Engagement Rate:", df_instagram["engagement_rate"].max())
print("\n📊 Minimum Engagement Rate:", df_instagram["engagement_rate"].min())

# ✅ Step 13: Evaluate Model Accuracy (Mean Absolute Error)
mae_fb = mean_absolute_error(df_facebook["engagement_rate"].dropna(), df_facebook["predicted_engagement"])
mae_insta = mean_absolute_error(df_instagram["engagement_rate"].dropna(), df_instagram["predicted_engagement"])

print("\n📊 Model Accuracy (Mean Absolute Error):")
print(f"Facebook Prediction Error: {mae_fb:.3f}")
print(f"Instagram Prediction Error: {mae_insta:.3f}")

# ✅ Step 14: Visualization - Actual vs. Predicted Engagement
plt.figure(figsize=(12, 6))

sns.scatterplot(x=df_facebook["engagement_rate"], y=df_facebook["predicted_engagement"], color="red", label="Facebook")
sns.scatterplot(x=df_instagram["engagement_rate"], y=df_instagram["predicted_engagement"], color="blue", label="Instagram")

# ✅ Add Trend Lines for Better Interpretation
z_fb = np.polyfit(df_facebook["engagement_rate"], df_facebook["predicted_engagement"], 1)
p_fb = np.poly1d(z_fb)
plt.plot(df_facebook["engagement_rate"], p_fb(df_facebook["engagement_rate"]), color="red", linestyle="dashed")

z_insta = np.polyfit(df_instagram["engagement_rate"], df_instagram["predicted_engagement"], 1)
p_insta = np.poly1d(z_insta)
plt.plot(df_instagram["engagement_rate"], p_insta(df_instagram["engagement_rate"]), color="blue", linestyle="dashed")

plt.title("📊 Actual vs. Predicted Engagement Rates (Facebook & Instagram)")
plt.xlabel("Actual Engagement Rate (%)")
plt.ylabel("Predicted Engagement Rate (%)")
plt.legend()
plt.show()
