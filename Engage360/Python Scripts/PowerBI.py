import os
import pandas as pd
import numpy as np

# 📂 Auto-detect file paths for scalability
data_dir = r"C:\Users\valarsri\Downloads"  # ✅ Define dataset directory

# ✅ Create full file paths using 'os.path.join' for compatibility
facebook_file = os.path.join(data_dir, "Facebook_Analytics.xlsx")
instagram_file = os.path.join(data_dir, "Instagram_Analytics.xlsx")

# ✅ Step 1: Load Facebook Analytics Sheets
fb_profile = pd.read_excel(facebook_file, sheet_name="Facebook Profile Overview")

# ✅ Load Facebook Post Engagement (Fix header issue)
fb_engagement = pd.read_excel(facebook_file, sheet_name="Facebook Post Engagement")  # Skip first two rows
#fb_engagement_raw.columns = fb_engagement_raw.iloc[300]  # Use row 301 (index 299) as the header
#fb_engagement = fb_engagement_raw.iloc[301:].reset_index(drop=True)  # Remove row 301

# ✅ Load Instagram Analytics Sheets
insta_posts = pd.read_excel(instagram_file, sheet_name="Instagram Post Engagement")
insta_demographics = pd.read_excel(instagram_file, sheet_name="Instagram Profile Overview")

print("\n✅ Loaded all datasets successfully!")

# ✅ Step 2: Standardize Column Names
for df in [fb_profile, fb_engagement, insta_posts, insta_demographics]:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\n🔍 Standardized column names for consistency.")

# ✅ Step 3: Print Data Types Before Conversion
print("\n🔍 Checking Data Types Before Conversion:")
print("\nFacebook Profile Overview:", fb_profile.dtypes)
print("\nFacebook Post Engagement:", fb_engagement.dtypes)
print("\nInstagram Post Engagement:", insta_posts.dtypes)
print("\nInstagram Demographics:", insta_demographics.dtypes)

# ✅ Step 4: Convert Columns Used in Calculations to Numeric
fb_profile_cols = ["new_likes", "unlikes", "%_of_reach_from_organic", "%_of_reach_from_paid", "page_post_engagements", "total_impressions"]
fb_engagement_cols = ["post_reach", "post_impressions", "total_post_reactions", "comments_on_posts", "organic_video_views", "video_views"]
insta_posts_cols = ["like_count", "comments_count", "shares", "media_impressions", "unique_saves"]
insta_demographics_cols = ["new_followers"]

for df, cols in zip([fb_profile, fb_engagement, insta_posts, insta_demographics], [fb_profile_cols, fb_engagement_cols, insta_posts_cols, insta_demographics_cols]):
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

print("\n✅ Data Type Conversion Completed Successfully!")

# ✅ Step 5: Apply Custom Metrics

# 📌 Facebook Profile Overview Metrics
fb_profile["net_follower_growth"] = fb_profile["new_likes"] - fb_profile["unlikes"]
fb_profile["organic_vs_paid_ratio"] = fb_profile["%_of_reach_from_organic"] / fb_profile["%_of_reach_from_paid"]
fb_profile["engagement_rate"] = (fb_profile["page_post_engagements"] / fb_profile["total_impressions"]) * 100

# 📌 Facebook Post Engagement Metrics
fb_engagement["post_efficiency_score"] = fb_engagement["post_reach"] / fb_engagement["post_impressions"]
fb_engagement["reaction_comment_ratio"] = fb_engagement["total_post_reactions"] / fb_engagement["comments_on_posts"]
fb_engagement["video_view_retention"] = fb_engagement["organic_video_views"] / fb_engagement["video_views"]

# 📌 Instagram Post Engagement Metrics
insta_posts["like_to_comment_ratio"] = insta_posts["like_count"] / insta_posts["comments_count"]
insta_posts["save_rate"] = insta_posts["unique_saves"] / insta_posts["media_impressions"]
insta_posts["post_virality_score"] = (insta_posts["shares"] / insta_posts["media_impressions"]) * 100

# 📌 Instagram Demographics Metrics
insta_demographics["profile_growth_rate"] = insta_demographics["new_followers"] / insta_demographics["new_followers"].shift(30)

print("\n✅ Custom metrics applied successfully!")

# ✅ Step 6: Handle Missing Values
for df in [fb_profile, fb_engagement, insta_posts, insta_demographics]:
    df.fillna(0, inplace=True)  # Replace NaN values with zero if needed

print("\n🔍 Checked & cleaned missing values!")

# ✅ Step 7: Reprint Data Types After Conversion
print("\n🔍 Checking Data Types After Conversion:")
print("\nFacebook Profile Overview:", fb_profile.dtypes)
print("\nFacebook Post Engagement:", fb_engagement.dtypes)
print("\nInstagram Post Engagement:", insta_posts.dtypes)
print("\nInstagram Demographics:", insta_demographics.dtypes)

# ✅ Step 8: Save Processed Data for Power BI
fb_profile.to_excel(os.path.join(data_dir, "Enhanced_Facebook_Profile_Overview.xlsx"), index=False)
fb_engagement.to_excel(os.path.join(data_dir, "Enhanced_Facebook_Post_Engagement.xlsx"), index=False)
insta_posts.to_excel(os.path.join(data_dir, "Enhanced_Instagram_Post_Engagement.xlsx"), index=False)
insta_demographics.to_excel(os.path.join(data_dir, "Enhanced_Instagram_Demographics.xlsx"), index=False)

print("\n✅ Final datasets saved successfully for Power BI integration!")
