# Task 1: Importing All Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Task 2: Loading Dataset
df = pd.read_csv("C:\\Users\\Javed\\Downloads\\Electric_Vehicle_Population_Data.csv")

# Task 3: Initial Exploration
print(df.head())
print(df.tail())
print("Shape of Dataset:", df.shape)
print(df.info())
print(df.describe())

# Task 4: Data Cleaning
print("Missing Values:\n", df.isnull().sum())

df.dropna(subset=['Make', 'Model Year', 'Electric Range', 'Electric Vehicle Type', 'City', 'County'], inplace=True)

df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')
df['Electric Range'] = pd.to_numeric(df['Electric Range'], errors='coerce')

print("Duplicate Rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicate Rows after drop:", df.duplicated().sum())

print(df.dtypes)

# Task 5: Data Analysis (EDA)

# 5.1: EV Registration Growth Trend
yearly_ev_trend = df.groupby('Model Year').size().reset_index(name='EV Count')
yearly_ev_trend = yearly_ev_trend.sort_values(by='Model Year')

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_ev_trend, x='Model Year', y='EV Count', marker='o', color='blue')
plt.title('Growth Trend of EV Registrations Over Time', fontsize=14)
plt.xlabel('Model Year')
plt.ylabel('Number of EV Registrations')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5.2: Most Popular EV Makes
popular_makes = df['Make'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=popular_makes.values, y=popular_makes.index, palette='magma')
plt.title("Top 10 EV Makes")
plt.xlabel("Number of Registrations")
plt.ylabel("EV Make")
plt.tight_layout()
plt.show()

# 5.3: Most Popular Cities for EVs
top_cities = df['City'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='cool')
plt.title("Top 10 Cities by EV Count")
plt.xlabel("Number of EVs")
plt.ylabel("City")
plt.tight_layout()
plt.show()

# 5.4: EV Type Distribution
ev_type_counts = df['Electric Vehicle Type'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=ev_type_counts.index, y=ev_type_counts.values, palette='Set2')
plt.title("EV Type Distribution")
plt.ylabel("Number of EVs")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 5.5: Top 15 Cities/Counties with Highest EVs
location_ev_counts = df.groupby(['City', 'County']).size().reset_index(name='EV Count')
top_locations = location_ev_counts.sort_values(by='EV Count', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_locations, x='EV Count', y='City', hue='County', palette='viridis')
plt.title('Top 15 Cities with Highest EV Counts (Potential Charging Station Need)', fontsize=14)
plt.xlabel('Number of EVs')
plt.ylabel('City')
plt.legend(title='County', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Task 6: Extra Visuals

# 6.1: Boxplot for Electric Range (Outlier Detection)
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Electric Range')
plt.title("Electric Range Outlier Detection")
plt.tight_layout()
plt.show()

# 6.2: Histogram for Electric Range Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Electric Range'], bins=30, kde=True, color='green')
plt.title("Electric Range Distribution")
plt.xlabel("Electric Range (miles)")
plt.tight_layout()
plt.show()

# 6.3: Correlation Heatmap of Numerical Features
numerical_cols = df.select_dtypes(include=[np.number])
corr_matrix = numerical_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(data=corr_matrix, annot=True, cmap='magma', fmt=".2f", linewidths=0.5)
plt.title("Heatmap: Correlation of EV Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()
