# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
file_path = "C:/Users/Anurag Sable/Downloads/archive (2)/ai_ghibli_trend_dataset_v2.csv"
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
data.info()

# Display the first few rows of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(data.head())

# Check for missing values
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Examine data types
print("\nData Types of Each Column:")
print(data.dtypes)

# Display summary statistics for numerical columns
print("\nSummary Statistics for Numerical Columns:")
print(data.describe())

# Initial data cleaning: Strip whitespace from column names
data.columns = data.columns.str.strip()

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicate_rows}")

# Drop duplicate rows if any
if duplicate_rows > 0:
    data = data.drop_duplicates()
    print("Duplicate rows dropped.")

# Display the shape of the cleaned dataset
print("\nShape of the Dataset After Initial Cleaning:")
print(data.shape)

# Analyze the distribution of different prompt themes
data['prompt_theme'] = data['prompt'].str.extract(r'(Ghibli-style|Studio Ghibli|Spirited Away|Anime-style|Mysterious|Magical|Cozy|Serene)', expand=False)

# Count the occurrences of each theme
theme_counts = data['prompt_theme'].value_counts()

# Plot the distribution of themes
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=theme_counts.index, y=theme_counts.values, palette="viridis")
plt.title("Distribution of Prompt Themes", fontsize=16)
plt.xlabel("Prompt Theme", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Identify the most popular themes based on engagement metrics
engagement_metrics = ['likes', 'shares', 'comments']
data['total_engagement'] = data[engagement_metrics].sum(axis=1)

# Group by theme and calculate average engagement
theme_engagement = data.groupby('prompt_theme')['total_engagement'].mean().sort_values(ascending=False)

# Plot the average engagement for each theme
plt.figure(figsize=(10, 6))
sns.barplot(x=theme_engagement.index, y=theme_engagement.values, palette="coolwarm")
plt.title("Average Engagement by Prompt Theme", fontsize=16)
plt.xlabel("Prompt Theme", fontsize=12)
plt.ylabel("Average Engagement", fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Explore relationships between themes and engagement metrics
theme_metrics = data.groupby('prompt_theme')[engagement_metrics].mean()

# Heatmap of engagement metrics by theme
plt.figure(figsize=(10, 6))
sns.heatmap(theme_metrics, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True)
plt.title("Engagement Metrics by Prompt Theme", fontsize=16)
plt.xlabel("Engagement Metric", fontsize=12)
plt.ylabel("Prompt Theme", fontsize=12)
plt.show()

from wordcloud import WordCloud

ghibli_text = ' '.join(data[data['prompt_theme'] == 'Ghibli-style']['prompt'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ghibli_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Words in Ghibli-Style Prompts')
plt.show()