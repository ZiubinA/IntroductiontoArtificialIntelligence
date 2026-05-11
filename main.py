import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

df = pd.read_csv("heart.csv")

def get_numeric_summary(df, cols):
    summary_data = []
    
    for col in cols:
        stats = {
            "Column": col,
            "Total_Values": df[col].count(),
            "%_Missing": round(df[col].isnull().mean() * 100, 2),
            "Cardinality": df[col].nunique(),
            "Min": round(df[col].min(), 2),
            "Max": round(df[col].max(), 2),
            "Q1": round(df[col].quantile(0.25), 2),
            "Q3": round(df[col].quantile(0.75), 2),
            "Average": round(df[col].mean(), 2),
            "Median": round(df[col].median(), 2),
            "Std_Dev": round(df[col].std(), 2)
        }
        summary_data.append(stats)

    summary_df = pd.DataFrame(summary_data)
    return summary_df

continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

numeric_table = get_numeric_summary(df, continuous_cols)
print("NUMERIC ATTRIBUTES TABLE")
print(numeric_table.to_string(index=False))
print("\n" + "="*50 + "\n")

def get_categorical_summary(df, cols):
    summary_data = []
    
    for col in cols:
        counts = df[col].value_counts()
        total = len(df)
        
        mode1, freq1, pct1 = "N/A", 0, 0
        mode2, freq2, pct2 = "N/A", 0, 0
        
        if len(counts) > 0:
            mode1 = counts.index[0]
            freq1 = counts.iloc[0]
            pct1 = (freq1 / total) * 100
            
        if len(counts) > 1:
            mode2 = counts.index[1]
            freq2 = counts.iloc[1]
            pct2 = (freq2 / total) * 100
            
        stats = {
            "Column": col,
            "Total_Values": df[col].count(),
            "%_Missing": round(df[col].isnull().mean() * 100, 2),
            "Cardinality": df[col].nunique(),
            "Mode": mode1,
            "Mode_Freq": freq1,
            "Mode_%": round(pct1, 2),
            "2nd_Mode": mode2,
            "2nd_Mode_Freq": freq2,
            "2nd_Mode_%": round(pct2, 2)
        }
        summary_data.append(stats)

    summary_df = pd.DataFrame(summary_data)
    return summary_df

categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]

cat_table = get_categorical_summary(df, categorical_cols)
print("CATEGORICAL ATTRIBUTES TABLE")
print(cat_table.to_string(index=False))

continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]

print("Generating Histograms")
n = len(df)
#formula to decide the optimal number of bars for a histogram based on the dataset size (n).
num_bins = int(1 + 3.22 * math.log10(n))

plt.figure(figsize=(15, 10))
for i, col in enumerate(continuous_cols):
    plt.subplot(2, 3, i + 1) 
    sns.histplot(df[col], bins=num_bins, kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Filled missing values in {col} with mode: {mode_val}")

for col in continuous_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    #Calculates safe limits
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_count = len(df)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    removed_count = original_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers from {col}")

# These are the columns where outliers were actually removed
cleaned_cols = ["trestbps", "chol", "thalach", "oldpeak"]

plt.figure(figsize=(12, 8))
plt.suptitle("Histograms AFTER Outlier Removal", fontsize=16)

for i, col in enumerate(cleaned_cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(df[col], kde=True, color='green') 
    plt.title(f"Cleaned Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

#Age vs Max Heart Rate
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='age', y='thalach', hue='target')
plt.title("Scatter: Age vs Thalach (Max Heart Rate)")
plt.show()

#Chol vs Trestbps
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='chol', y='trestbps', hue='target')
plt.title("Scatter: Cholesterol vs Resting BP")
plt.show()

sns.pairplot(df[continuous_cols])
plt.suptitle("SPLOM (Scatter Plot Matrix)", y=1.02)
plt.show()

#Chest Pain Type (cp) vs Heart Disease (target)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='cp', hue='target')
plt.title("Relationship: Chest Pain Type vs Heart Disease Target")
plt.show()

#Sex vs Heart Disease
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sex', hue='target')
plt.title("Relationship: Sex vs Heart Disease Target")
plt.show()

#Age grouped by Chest Pain Type (Categorical)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='cp', y='age')
plt.title("Box Plot: Age distribution by Chest Pain Type")
plt.show()

#Thalach (Continuous) grouped by Target (Categorical)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='target', y='thalach')
plt.title("Box Plot: Max Heart Rate by Heart Disease Target")
plt.show()

print("COVARIANCE MATRIX")
#Calculate Covariance for numeric columns only
cov_matrix = df[continuous_cols].cov()
print(cov_matrix)
print("\n" + "="*50 + "\n")

print("CORRELATION MATRIX")
#Calculate Correlation
corr_matrix = df[continuous_cols].corr()
print(corr_matrix)
print("\n" + "="*50 + "\n")

#Graphically represent the correlation matrix (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

print("DATA NORMALIZATION [0, 1]")

for col in continuous_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    df[col] = (df[col] - min_val) / (max_val - min_val)

print("Normalization Completed.")
print("Statistics AFTER Normalization:")
norm_table = get_numeric_summary(df, continuous_cols)
print(norm_table.to_string(index=False))
print("\n" + "="*50 + "\n")

print("PART 9: CONVERT CATEGORICAL TO NUMERIC")
display_cols = ['cp', 'thal', 'slope', 'sex', 'target']
print(df[display_cols].head(10).to_string(index=False))

print("\n" + "="*50 + "\n")