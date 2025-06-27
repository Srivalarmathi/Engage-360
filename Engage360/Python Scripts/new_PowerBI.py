import os
import pandas as pd
import numpy as np


instagram_file = r"C:\Users\valarsri\Downloads\Instagram_Analytics.xlsx"
xls = pd.ExcelFile(instagram_file)
print(xls.sheet_names)


# 📂 Set the working directory and file
data_dir = r"C:\Users\valarsri\Downloads"
instagram_file = os.path.join(data_dir, "Instagram_Analytics.xlsx")

# ✅ Step 1: Load new Instagram sheets
insta_age_gender = pd.read_excel(instagram_file, sheet_name="Instagram Age Gender Demographi")
insta_location = pd.read_excel(instagram_file, sheet_name="Instagram Top Cities Regions")
supermetrics_meta = pd.read_excel(instagram_file, sheet_name="SupermetricsQueries")

# ✅ Step 2: Standardize column names
for df in [insta_age_gender, insta_location, supermetrics_meta]:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Step 3: Convert to numeric where needed
insta_age_gender["profile_followers"] = pd.to_numeric(insta_age_gender["profile_followers"], errors="coerce")
insta_location["profile_followers"] = pd.to_numeric(insta_location["profile_followers"], errors="coerce")

# ✅ Step 4: Create calculated insights

# 📌 1. Age group dominance and follower distribution
insta_age_gender["age_group"] = insta_age_gender["age"].astype(str)
insta_age_gender["audience_percentage"] = (
    insta_age_gender["profile_followers"] / insta_age_gender["profile_followers"].sum()
).round(4) * 100

# 📌 2. Gender-based analysis
gender_summary = insta_age_gender.groupby("gender")["profile_followers"].sum().reset_index()
gender_summary["gender_audience_share"] = (
    gender_summary["profile_followers"] / gender_summary["profile_followers"].sum()
).round(4) * 100

# 📌 3. Top cities & region concentration
insta_location["city_audience_percentage"] = (
    insta_location["profile_followers"] / insta_location["profile_followers"].sum()
).round(4) * 100

# ✅ Step 5: Clean missing values
insta_age_gender.fillna(0, inplace=True)
insta_location.fillna(0, inplace=True)

# ✅ Step 6: Export cleaned & enriched files
insta_age_gender.to_excel(os.path.join(data_dir, "Enhanced_Instagram_Age_Gender.xlsx"), index=False)
gender_summary.to_excel(os.path.join(data_dir, "Enhanced_Instagram_Gender_Share.xlsx"), index=False)
insta_location.to_excel(os.path.join(data_dir, "Enhanced_Instagram_Location.xlsx"), index=False)
supermetrics_meta.to_excel(os.path.join(data_dir, "Enhanced_Supermetrics_Metadata.xlsx"), index=False)

print("\n✅ All Instagram demographic sheets enhanced and exported for Power BI integration!")
