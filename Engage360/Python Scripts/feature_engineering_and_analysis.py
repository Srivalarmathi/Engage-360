# ✅ Step 1: Import Essential Libraries for Data Processing

# 📂 Pandas → Used for handling structured data (DataFrames)
import pandas as pd

# 🧮 NumPy → Provides numerical operations, including handling missing values
import numpy as np

# 📂 OS → Enables file path operations for loading datasets dynamically
import os

# 📊 Seaborn → Used for advanced visualizations, especially statistical plotting
import seaborn as sns

# 📈 Matplotlib → Fundamental library for plotting and visualization
import matplotlib.pyplot as plt

# ⚖️ Scikit-Learn's MinMaxScaler → Used for **normalizing numeric features** before correlation analysis
from sklearn.preprocessing import MinMaxScaler

# 🔬 SciPy's T-Test → Used for **statistical hypothesis testing** (A/B testing on engagement metrics)
from scipy.stats import ttest_ind

# 🚨 Step 2: Suppress Non-Critical Warnings to Ensure Clean Output

# 📌 The 'tkinter' warning relates to missing font glyphs—this prevents unnecessary logs.
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")


# 📂 Auto-detect file paths for scalability
# This setup allows dynamic loading of datasets, preventing hardcoded paths for better flexibility.
data_dir = r"C:\Users\valarsri\Downloads"  # ✅ Define the directory where datasets are stored

# ✅ Create full file paths using 'os.path.join' for compatibility across operating systems
facebook_file = os.path.join(data_dir, "Facebook_Analytics.xlsx")  # Facebook's analytics file path
instagram_file = os.path.join(data_dir, "Instagram_Analytics.xlsx")  # Instagram's analytics file path

# ✅ Step 1: Load Facebook Analytics Sheets
# Reads the "Facebook Profile Overview" sheet to analyze page-level insights
fb_profile = pd.read_excel(facebook_file, sheet_name="Facebook Profile Overview")

# ✅ Load "Facebook Post Engagement" sheet while adjusting headers
# The first row contains metadata, so we skip it and use the second row as the header.
fb_posts = pd.read_excel(facebook_file, sheet_name="Facebook Post Engagement", header=1)

# ✅ Load Instagram Analytics Sheets
# Reads the "Instagram Post Engagement" sheet for analyzing post-level interactions
insta_posts = pd.read_excel(instagram_file, sheet_name="Instagram Post Engagement")

# Reads the "Instagram Profile Overview" sheet for tracking overall account performance
insta_profile = pd.read_excel(instagram_file, sheet_name="Instagram Profile Overview")

# ✅ Step 1: Standardize Column Names (Removes Spaces, Converts to Lowercase)
# Creates a dictionary to store all datasets, enabling batch processing
datasets = {
    "Facebook Profile": fb_profile,  # Facebook page-level insights
    "Facebook Posts": fb_posts,  # Facebook post-specific engagement metrics
    "Instagram Profile": insta_profile,  # Instagram account-level metrics
    "Instagram Posts": insta_posts  # Instagram post-level engagement analytics
}

# ✅ Loop through each dataset to apply standardization
for name, df in datasets.items():
    # ✅ Remove leading and trailing spaces from column names
    df.columns = df.columns.str.strip()

    # ✅ Convert column names to lowercase for consistency across all datasets
    df.columns = df.columns.str.lower()

    # ✅ Replace spaces with underscores to improve compatibility with programming tools
    df.columns = df.columns.str.replace(" ", "_")

# ✅ Step 2: Perform Basic Sheet-by-Sheet Checks
# Loop through each dataset to inspect its structure and integrity
for name, df in datasets.items():
    print(f"\n🔍 **Analyzing {name}**")  # ✅ Display dataset name for clarity

    # ✅ Print the number of columns in the dataset
    print(f"📊 Number of Columns: {df.shape[1]}")

    # ✅ Print the list of column names for quick reference
    print(f"📝 Column Names: {df.columns.tolist()}")

    # ✅ Display the data types of each column to ensure consistency
    print("\n📌 Data Types:")
    print(df.dtypes)

    # ✅ Identify and print missing values
    missing_values = df.isnull().sum()  # ✅ Count missing values per column
    print("\n🚨 Missing Values:")
    print(missing_values[missing_values > 0])  # ✅ Show only columns with missing values

    print("=" * 50)  # ✅ Divider for readability

# ✅ Step 3: Remove Fully Empty Columns from Facebook Profile
# Identify columns where all values are missing
empty_cols = fb_profile.columns[fb_profile.isnull().all()]

# ✅ Drop empty columns to clean the dataset
fb_profile.drop(columns=empty_cols, inplace=True)

# ✅ Print removed columns for transparency
print("\n🚨 Removed Empty Columns in Facebook Profile:", empty_cols.tolist())

# ✅ Step 4: Identify Missing Values Before Cleaning
# ✅ Create a dictionary to store total missing values for each dataset
missing_values_before = {
    name: df.isnull().sum().sum()  # ✅ Calculate total missing values for the dataset
    for name, df in datasets.items()  # ✅ Iterate over all datasets in the 'datasets' dictionary
}

# ✅ Print missing values summary to assess the extent of the issue
print("\n📊 **Total Missing Values Before Cleaning:**", missing_values_before)

# ✅ Step 1: Data Exploration
# Print a header to indicate the beginning of the exploratory analysis
print("*****************Data Exploration *************************")

# ✅ Display Dataset Info
# Prints a summary of each dataset, including number of entries, column data types, and memory usage.
print("📊 Facebook Engagement Data Overview:")
print(fb_profile.info())  # ✅ Displays structure & data types of Facebook Profile dataset

print("\n🔍 Instagram Profile Data Overview:")
print(insta_profile.info())  # ✅ Displays structure & data types of Instagram Profile dataset

print("\n📊 Instagram Posts Data Overview:")
print(insta_posts.info())  # ✅ Displays structure & data types of Instagram Posts dataset

print("\n📊 Facebook Posts Data Overview:")
print(fb_posts.info())  # ✅ Displays structure & data types of Facebook Posts dataset

# ✅ Step 2: Identify Missing Values
# Check for missing values across all datasets and print them for visibility
print("\n🚨 Missing Values in Facebook Profile:")
print(fb_profile.isnull().sum())  # ✅ Counts missing values per column in Facebook Profile dataset

print("\n🚨 Missing Values in Instagram Profile:")
print(insta_profile.isnull().sum())  # ✅ Counts missing values per column in Instagram Profile dataset

print("\n🚨 Missing Values in Instagram Posts:")
print(insta_posts.isnull().sum())  # ✅ Counts missing values per column in Instagram Posts dataset

print("\n🚨 Missing Values in Facebook Posts:")
print(fb_posts.isnull().sum())  # ✅ Counts missing values per column in Facebook Posts dataset

# 🔹 Step 3: Visualizing Missing Values BEFORE Cleaning
# This helps understand the extent of missing data before applying any imputation techniques
plt.figure(figsize=(12, 6))  # ✅ Set figure size for readability

# ✅ Create a bar plot to represent missing values in each dataset
sns.barplot(x=list(missing_values_before.keys()), y=list(missing_values_before.values()),
            hue=list(missing_values_before.keys()), palette="coolwarm", legend=False)

# ✅ Set the title and labels for the plot
plt.title("Missing Values Before Cleaning", fontsize=14, fontweight="bold")
plt.ylabel("Total Missing Values")
plt.xticks(rotation=20)  # ✅ Rotate labels for better visibility

# ✅ Add value annotations above each bar for clarity
for i, v in enumerate(missing_values_before.values()):
    plt.text(i, v + 50, str(v), ha='center', fontsize=12, fontweight='bold')

# ✅ Display the plot
plt.show()

# ✅ Step 4: Handle Missing Values Using Mode & Median
# Iterates through each dataset and applies missing value treatments

for df in datasets.values():
    # ✅ Fill missing values in numerical columns with their median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].notnull().sum() > 0:  # ✅ Ensures column has valid values before computing median
            df[col] = df[col].fillna(df[col].median())  # ✅ Median is used to prevent extreme values

    # ✅ Fill missing values in categorical columns with their mode (most frequent value)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # ✅ Mode ensures the most common value fills missing entries

# ✅ Step 6: Remove Fully Empty Rows
# Removes rows where **all values** are missing to clean up the datasets efficiently.
for df in datasets.values():
    df.dropna(how='all', inplace=True)  # ✅ Drops rows that are entirely empty (i.e., NaN in all columns)

# ✅ Step 7: Identify Missing Values After Cleaning
# Creates a dictionary storing the **total missing values per dataset** after cleaning.
missing_values_after = {
    name: df.isnull().sum().sum()  # ✅ Counts total missing values in each dataset
    for name, df in datasets.items()  # ✅ Iterates over all datasets in the 'datasets' dictionary
}

# 🔹 Step 8: Visualizing Missing Values AFTER Cleaning
# This helps confirm whether the missing values have been significantly reduced.
plt.figure(figsize=(12, 6))  # ✅ Set figure size for readability

# ✅ Generate a bar plot displaying missing values per dataset after cleaning
sns.barplot(x=list(missing_values_after.keys()), y=list(missing_values_after.values()),
            hue=list(missing_values_after.keys()), palette="crest", legend=False)

# ✅ Set title and labels for the visualization
plt.title("Missing Values After Cleaning", fontsize=14, fontweight="bold")
plt.ylabel("Total Missing Values")
plt.xticks(rotation=20)  # ✅ Rotate labels for better readability

# ✅ Annotate missing values count above bars
for i, v in enumerate(missing_values_after.values()):
    plt.text(i, v + 50, str(v), ha='center', fontsize=12, fontweight='bold')

# ✅ Display the plot
plt.show()

# ✅ Step 9: Basic Statistical Summary
# Prints summary statistics for each dataset, including count, mean, min, max, etc.
print("\n📊 Facebook Profile Summary:")
print(fb_profile.describe())  # ✅ Displays descriptive stats for Facebook Profile dataset

print("\n📊 Instagram Profile Summary:")
print(insta_profile.describe())  # ✅ Displays descriptive stats for Instagram Profile dataset

print("\n📊 Instagram Posts Summary:")
print(insta_posts.describe())  # ✅ Displays descriptive stats for Instagram Posts dataset

print("\n📊 Facebook Posts Summary:")
print(fb_posts.describe())  # ✅ Displays descriptive stats for Facebook Posts dataset

# 🔹 Step 10: Visualizing Data Distribution
# Uses histograms to analyze the spread and frequency of engagement metrics.

# ✅ Histogram for Facebook Post Reach Distribution
plt.figure(figsize=(12, 6))
sns.histplot(fb_posts["post_reach"].dropna(), bins=20, kde=True)  # ✅ Excludes NaN values before plotting
plt.title("📊 Facebook Post Reach Distribution")  # ✅ Set plot title
plt.xlabel("Post Reach")  # ✅ Label X-axis
plt.ylabel("Frequency")  # ✅ Label Y-axis
plt.show()

# ✅ Histogram for Instagram Profile Impressions Distribution
plt.figure(figsize=(12, 6))
sns.histplot(insta_profile["profile_impressions"].dropna(), bins=20, kde=True)  # ✅ Excludes NaN values before plotting
plt.title("📊 Instagram Profile Impressions Distribution")  # ✅ Set plot title
plt.xlabel("Profile Impressions")  # ✅ Label X-axis
plt.ylabel("Frequency")  # ✅ Label Y-axis
plt.show()

# ✅ Step 11: Check Unique Post Types
# Extracts distinct values in categorical columns to analyze different post formats.

# ✅ Unique Facebook Post Types
print("\n📝 Unique Post Types in Facebook Data:")
print(fb_posts["post_type"].unique())  # ✅ Displays unique post types (e.g., Photo, Video, Album)

# ✅ Unique Instagram Media Types
print("\n📝 Unique Media Types in Instagram Posts:")
print(insta_posts["media_product_type"].unique())  # ✅ Displays unique media formats (e.g., FEED, REELS)

# ****************Correlation************************

# ✅ Step 1: Define Columns to Exclude from Correlation Analysis (Fully Zero)
# Certain columns contain only zeros and do not contribute meaningful relationships.
excluded_correlation_cols = {
    "Facebook Profile": ["organic_impressions"],  # ✅ Organic Impressions are fully zero and do not affect correlation.
    "Facebook Posts": ["__of_reach_from_organic", "__of_reach_from_paid", "shares_on_shares"],  # ✅ Irrelevant metrics for correlation.
    "Instagram Profile": [],  # ✅ No fully zero columns identified.
    "Instagram Posts": []  # ✅ No fully zero columns identified.
}

# ✅ Step 2: Filter Numeric Columns & Remove Excluded Correlation Columns
# Extract numeric features while ensuring unnecessary columns are excluded.
numeric_datasets = {}

for name, df in datasets.items():
    numeric_cols = df.select_dtypes(include=["number"])  # ✅ Select only numerical columns.
    numeric_cols = numeric_cols.loc[:, ~numeric_cols.columns.isin(excluded_correlation_cols.get(name, []))]  # ✅ Exclude fully zero columns.
    numeric_datasets[name] = numeric_cols  # ✅ Store filtered numeric datasets.

# ✅ Step 3: Normalize Features Before Correlation Analysis
# Applies MinMax Scaling to ensure that values range between 0 and 1 for accurate comparisons.
for name, df in numeric_datasets.items():
    scaler = MinMaxScaler()  # ✅ Instantiate MinMaxScaler to normalize data.
    numeric_datasets[name] = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)  # ✅ Apply scaling and retain column names.

# ✅ Step 4: Compute & Visualize Correlation for Each Dataset
# Generates heatmaps for correlation matrices to reveal relationships between different metrics.
for name, df in numeric_datasets.items():
    plt.figure(figsize=(10, 6))  # ✅ Set figure size for better readability.
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)  # ✅ Create a correlation matrix heatmap.
    plt.title(f"📊 {name} Correlation Matrix (Filtered & Scaled)")  # ✅ Title indicating dataset name.
    plt.show()  # ✅ Display the correlation heatmap.

# ✅ Step 5: Confirm Data Exploration Is Complete
print("*****************Data Exploration *************************")

# ✅ Step 8: Winsorization for Managing Outliers
# Winsorization is a technique used to limit extreme values (outliers)
# Instead of removing outliers entirely, this method caps them within a specific range (5th & 95th percentile)

def winsorize_outliers(df, columns, lower_quantile=0.05, upper_quantile=0.95):
    """
    Apply Winsorization to specified columns in the dataframe.

    Parameters:
    df (DataFrame): The dataset containing numeric columns.
    columns (list): List of column names to apply Winsorization.
    lower_quantile (float): Lower percentile boundary (default: 5th percentile).
    upper_quantile (float): Upper percentile boundary (default: 95th percentile).

    Returns:
    DataFrame: Updated dataframe with winsorized values.
    """
    for col in columns:
        lower_limit = df[col].quantile(lower_quantile)  # ✅ Compute 5th percentile threshold
        upper_limit = df[col].quantile(upper_quantile)  # ✅ Compute 95th percentile threshold

        # ✅ Replace values below the lower limit with the lower limit
        df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])

        # ✅ Replace values above the upper limit with the upper limit
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

    return df  # ✅ Return the dataframe with winsorized values


# ✅ Step 9: Apply Winsorization to All Numeric Datasets
# Define numeric columns per dataset that need winsorization to prevent extreme variations.
numeric_cols = {
    "Facebook Profile": ["total_impressions", "total_reach", "page_post_engagements"],
    "Facebook Posts": ["post_impressions", "post_reach", "total_post_reactions"],
    "Instagram Posts": ["media_impressions", "media_reach", "like_count"],
    "Instagram Profile": ["profile_impressions", "profile_reach", "new_followers"]
}

# ✅ Loop through each dataset and apply winsorization
for name, cols in numeric_cols.items():
    datasets[name] = winsorize_outliers(datasets[name], cols)

# 🔹 **Step 10: Visualizing Outliers After Winsorization**
# Generates box plots for each dataset to validate the effect of winsorization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Outliers in All Datasets After Winsorization", fontsize=16, fontweight="bold")

# ✅ Box plot for Facebook Profile dataset
sns.boxplot(data=fb_profile[numeric_cols["Facebook Profile"]], palette="flare", ax=axes[0, 0])
axes[0, 0].set_title("Facebook Profile", fontsize=14)

# ✅ Box plot for Facebook Posts dataset
sns.boxplot(data=fb_posts[numeric_cols["Facebook Posts"]], palette="magma", ax=axes[0, 1])
axes[0, 1].set_title("Facebook Posts", fontsize=14)

# ✅ Box plot for Instagram Posts dataset
sns.boxplot(data=insta_posts[numeric_cols["Instagram Posts"]], palette="coolwarm", ax=axes[1, 0])
axes[1, 0].set_title("Instagram Posts", fontsize=14)

# ✅ Box plot for Instagram Profile dataset
sns.boxplot(data=insta_profile[numeric_cols["Instagram Profile"]], palette="crest", ax=axes[1, 1])
axes[1, 1].set_title("Instagram Profile", fontsize=14)

# ✅ Rotate x-axis labels for readability
plt.xticks(rotation=20)

# ✅ Show the final visualization
plt.show()

# ✅ Step 9: Save Cleaned Data for Reporting

# ✅ Loop through each dataset and save cleaned versions
for name, df in datasets.items():
    df.to_csv(f"C:\\Users\\valarsri\\Downloads\\Cleaned_{name.replace(' ', '_')}.csv", index=False)

print("\n✅ Cleaned files saved separately for Power BI reporting!")

# ✅ Step 10: Merge All Sheets into a Single DataFrame
# Combines Facebook Profile, Facebook Posts, Instagram Profile, and Instagram Posts for consolidated analysis.
merged_data = pd.concat([fb_profile, fb_posts, insta_posts, insta_profile], ignore_index=True)

# ✅ Save the merged dataset
merged_data.to_csv(r"C:\Users\valarsri\Downloads\Merged_Analytics.csv", index=False)

print("\n✅ All sheets merged and saved as 'Merged_Analytics.csv' for further processing!", flush=True)

# 🔹 Step 11: Create a Data Dictionary for Each Dataset
data_dictionary = {}

for name, df in datasets.items():
    # ✅ Store a placeholder description for each column
    description = {col: f"{name} - {col} description" for col in df.columns}  # Replace with actual descriptions
    data_dictionary[name] = description

# ✅ Convert dictionary to a DataFrame for structured reporting
data_dict_df = pd.DataFrame.from_dict(data_dictionary, orient='index').transpose()

# ✅ Save Data Dictionary for Reference
data_dict_df.to_csv(r"C:\Users\valarsri\Downloads\Data_Dictionary.csv", index=False)
print("\n✅ Data Dictionary saved as 'Data_Dictionary.csv'")

# ✅ Step 12: Compute Summary Statistics for Each Dataset
summary_stats = {}

for name, df in datasets.items():
    stats = df.describe().transpose()  # ✅ Compute summary statistics
    summary_stats[name] = stats

# ✅ Save Summary Stats
summary_df = pd.concat(summary_stats, axis=1)
summary_df.to_csv(r"C:\Users\valarsri\Downloads\Initial_Data_Summary.csv")
print("\n✅ Initial Data Summary saved as 'Initial_Data_Summary.csv'")

# ✅ Step 13: Compare Missing Values Before & After Cleaning
missing_comparison = pd.DataFrame({"Before Cleaning": missing_values_before, "After Cleaning": missing_values_after})
missing_comparison.to_csv(r"C:\Users\valarsri\Downloads\Missing_Value_Treatment.csv", index=False)

print("\n✅ Missing Value Treatment saved as 'Missing_Value_Treatment.csv'")

# 🔹 Step 14: Ensure columns exist before running `.describe()` for Winsorization
winsorized_stats = {}

for name, cols in numeric_cols.items():
    if name in datasets:
        existing_cols = [col for col in cols if col in datasets[name].columns]  # ✅ Ensure numeric columns exist

        if existing_cols:
            stats_before = datasets[name][existing_cols].describe()
            stats_after = winsorize_outliers(datasets[name], existing_cols).describe()

            winsorized_stats[name] = pd.concat(
                [stats_before, stats_after], keys=["Before Winsorization", "After Winsorization"]
            )
        else:
            print(f"🚨 Warning: No matching numeric columns found in {name} for Winsorization.")

# ✅ Save Winsorized Statistics
if winsorized_stats:
    winsorized_df = pd.concat(winsorized_stats, axis=1)
    winsorized_df.to_csv(r"C:\Users\valarsri\Downloads\Outlier_Handling_Report.csv")
    print("\n✅ Outlier Handling Report saved as 'Outlier_Handling_Report.csv'")
else:
    print("\n🚨 No valid columns found for Winsorization. Outlier report not generated.")

# ✅ Step 15: Compute Engagement Rate for Facebook & Instagram Posts
datasets["Instagram Posts"]["engagement_rate"] = (
    datasets["Instagram Posts"]["like_count"] + datasets["Instagram Posts"]["comments_count"]
) / datasets["Instagram Posts"]["media_impressions"] * 100

datasets["Facebook Posts"]["engagement_rate"] = (
    datasets["Facebook Posts"]["total_post_reactions"] + datasets["Facebook Posts"]["comments_on_posts"]
) / datasets["Facebook Posts"]["post_impressions"] * 100

# ✅ Step 16: Reach Trend Analysis Over Time
reach_trend = datasets["Instagram Profile"][["date", "profile_reach"]].groupby("date").sum()
reach_trend.plot(kind="line", title="Instagram Reach Over Time")
plt.show()

# 📌 Step 17: Identify Top-Performing Post Types (Reels vs. Static Posts)
sns.barplot(x="media_product_type", y="engagement_rate", data=datasets["Instagram Posts"])
plt.title("Engagement Comparison: Reels vs. Static Posts")
plt.show()

# 📌 Step 18: Conduct Trend Analysis on Organic vs. Paid Reach
datasets["Facebook Posts"].groupby("post_type")[["post_organic_reach", "post_reach"]].sum().plot(
    kind="bar", stacked=True, title="Organic vs. Paid Reach Trends"
)
plt.show()


print("**************A/B Testing******************")


# ✅ Load Facebook Posts dataset (Fix: Removes incorrect header row)
facebook_engagement_data = pd.read_excel(facebook_file, sheet_name="Facebook Post Engagement", header=1)

# 🚀 Skip unwanted headers from incorrect indexing
facebook_engagement_data = facebook_engagement_data.iloc[302:]
facebook_engagement_data.reset_index(drop=True, inplace=True)

# ✅ Ensure DataFrame changes are applied safely
facebook_engagement_data = facebook_engagement_data.copy()

# ✅ Standardize column names to lowercase for consistency
facebook_engagement_data.columns = facebook_engagement_data.columns.str.strip().str.lower()

# ✅ Expand Group B to include **all post types** if sample size is small
ad_a_types = ["photo", "album", "cover_photo"]
ad_b_types = list(facebook_engagement_data["post_type"].unique())

# ✅ Filter data for expanded post types
ad_a = facebook_engagement_data[facebook_engagement_data["post_type"].isin(ad_a_types)]["post_reach"].dropna()
ad_b = facebook_engagement_data[facebook_engagement_data["post_type"].isin(ad_b_types)]["post_reach"].dropna()

# ✅ Check sample sizes before running the test
print(f"Sample size for Group A ({ad_a_types}): {len(ad_a)}")
print(f"Sample size for Group B ({ad_b_types}): {len(ad_b)}")

# ✅ Perform A/B Test if sample size is sufficient
if len(ad_a) > 1 and len(ad_b) > 1:
    t_stat, p_value = ttest_ind(ad_a, ad_b, equal_var=False)  # ✅ Welch’s t-test for accuracy
    print(f"\n📊 T-Test Results: t-statistic = {t_stat}, p-value = {p_value}")

    # ✅ Decision Based on p-value
    if p_value < 0.05:
        print(f"📈 Significant difference detected between {ad_a_types} and {ad_b_types}!")
    else:
        print(f"📉 No significant difference found between {ad_a_types} and {ad_b_types}.")
else:
    print("\n🚨 Still not enough data for a valid A/B test! Providing descriptive insights instead...")


# ✅ Step 12: Extract Numerical Insights from Key Metrics

# 🔹 Compute overall engagement summary for Facebook & Instagram posts
engagement_summary = {
    "Facebook Posts": {
        "Avg Engagement Rate (%)": datasets["Facebook Posts"]["engagement_rate"].mean(),
        "Max Engagement Rate (%)": datasets["Facebook Posts"]["engagement_rate"].max(),
        "Min Engagement Rate (%)": datasets["Facebook Posts"]["engagement_rate"].min(),
    },
    "Instagram Posts": {
        "Avg Engagement Rate (%)": datasets["Instagram Posts"]["engagement_rate"].mean(),
        "Max Engagement Rate (%)": datasets["Instagram Posts"]["engagement_rate"].max(),
        "Min Engagement Rate (%)": datasets["Instagram Posts"]["engagement_rate"].min(),
    }
}

# 🔹 Compute reach statistics for Facebook & Instagram profiles
reach_summary = {
    "Facebook Profile": {
        "Avg Total Reach": datasets["Facebook Profile"]["total_reach"].mean(),
        "Max Total Reach": datasets["Facebook Profile"]["total_reach"].max(),
        "Min Total Reach": datasets["Facebook Profile"]["total_reach"].min(),
    },
    "Instagram Profile": {
        "Avg Profile Reach": datasets["Instagram Profile"]["profile_reach"].mean(),
        "Max Profile Reach": datasets["Instagram Profile"]["profile_reach"].max(),
        "Min Profile Reach": datasets["Instagram Profile"]["profile_reach"].min(),
    }
}

# ✅ Convert extracted summary data into DataFrames for structured reporting
engagement_summary_df = pd.DataFrame.from_dict(engagement_summary, orient="index")
reach_summary_df = pd.DataFrame.from_dict(reach_summary, orient="index")

# ✅ Save computed summaries as CSV files
engagement_summary_df.to_csv(r"C:\Users\valarsri\Downloads\Engagement_Summary.csv")
reach_summary_df.to_csv(r"C:\Users\valarsri\Downloads\Reach_Summary.csv")

print("\n✅ Engagement Summary saved as 'Engagement_Summary.csv'")
print("\n✅ Reach Summary saved as 'Reach_Summary.csv'")

# 🔹 Extract numerical insights from trend analysis (Instagram Reach Over Time)
insta_trend_stats = {
    "Average Reach Growth per Day": reach_trend["profile_reach"].diff().mean(),
    "Max Reach Growth": reach_trend["profile_reach"].diff().max(),
    "Min Reach Growth": reach_trend["profile_reach"].diff().min(),
}

# ✅ Convert into DataFrame and save as CSV
insta_trend_df = pd.DataFrame(insta_trend_stats, index=["Instagram Reach Trend"])
insta_trend_df.to_csv(r"C:\Users\valarsri\Downloads\Instagram_Reach_Trend.csv")

print("\n✅ Instagram Reach Trend Analysis saved as 'Instagram_Reach_Trend.csv'")

# 🔹 Extract summary of Organic vs. Paid Reach for Facebook Posts
facebook_reach_summary = datasets["Facebook Posts"].groupby("post_type")[
    ["post_organic_reach", "post_reach"]
].sum()

facebook_reach_summary.to_csv(r"C:\Users\valarsri\Downloads\Facebook_Organic_vs_Paid_Reach.csv")

print("\n✅ Facebook Organic vs. Paid Reach Summary saved as 'Facebook_Organic_vs_Paid_Reach.csv'")

# ✅ Step 13: Extract Key A/B Testing Metrics
ab_test_results = {
    "T-Statistic": t_stat if 't_stat' in locals() else "N/A",
    "P-Value": p_value if 'p_value' in locals() else "N/A",
    "Significant Difference?": "Yes" if 'p_value' in locals() and p_value < 0.05 else "No",
}

# ✅ Convert A/B test results into a DataFrame and save
ab_test_df = pd.DataFrame(ab_test_results, index=["A/B Test Summary"])
ab_test_df.to_csv(r"C:\Users\valarsri\Downloads\AB_Test_Results.csv")

print("\n✅ A/B Test Results saved as 'AB_Test_Results.csv'")

# Line chart for engagement trends over time
plt.figure(figsize=(10, 5))
sns.lineplot(x=df["date"], y=df["engagement_rate"])
plt.title("Post Engagement Trend Over Time")
plt.show()

def predict_success(likes, comments, shares, threshold=500):
    engagement_score = likes + comments + shares
    return "High-Performing" if engagement_score >= threshold else "Average"

# Example usage
print(predict_success(200, 100, 250))


# ✅ Compute engagement rate for Instagram Posts
if all(col in insta_posts.columns for col in ["like_counts", "comments_counts", "shares", "media_reach"]):
    insta_posts["engagement_rate"] = (
        (insta_posts["like_counts"] + insta_posts["comments_counts"] + insta_posts["shares"])
        / insta_posts["media_reach"]
    ) * 100
else:
    print("\n🚨 Missing columns for Instagram engagement rate calculation!")

# ✅ Compute engagement rate for Facebook Posts
if all(col in fb_posts.columns for col in ["total_post_reactions", "comments_on_posts", "shares_on_post", "post_reach"]):
    fb_posts["engagement_rate"] = (
        (fb_posts["total_post_reactions"] + fb_posts["comments_on_posts"] + fb_posts["shares_on_post"])
        / fb_posts["post_reach"]
    ) * 100
else:
    print("\n🚨 Missing columns for Facebook engagement rate calculation!")

print("\n✅ Engagement rates successfully calculated!")

# ✅ Find top-performing Instagram post
top_instagram_post = insta_posts.loc[insta_posts["engagement_rate"].idxmax()]
print("\n🚀 Top Performing Instagram Post:\n", top_instagram_post)

# ✅ Find top-performing Facebook post
top_facebook_post = fb_posts.loc[fb_posts["engagement_rate"].idxmax()]
print("\n🚀 Top Performing Facebook Post:\n", top_facebook_post)


# ✅ Save processed datasets
fb_profile.to_csv(r"C:\Users\valarsri\Downloads\Cleaned_Facebook_Profile.csv", index=False)
fb_posts.to_csv(r"C:\Users\valarsri\Downloads\Cleaned_Facebook_Posts.csv", index=False)
insta_posts.to_csv(r"C:\Users\valarsri\Downloads\Cleaned_Instagram_Posts.csv", index=False)
insta_profile.to_csv(r"C:\Users\valarsri\Downloads\Cleaned_Instagram_Demographics.csv", index=False)

print("\n✅ Cleaned files saved successfully!")


sns.boxplot(x="media_product_type", y="engagement_rate", hue="media_product_type", data=insta_posts, palette="coolwarm", legend=False)
plt.title("📊 Engagement Comparison: Reels vs. Static Posts")
plt.show()

# Automating Social Media Insights
low_engagement_posts = df[df["engagement_rate"] < 2]
if not low_engagement_posts.empty:
    print(f"🚨 {len(low_engagement_posts)} posts have low engagement. Consider optimizing content!")
print("***************************Data Science************************")


