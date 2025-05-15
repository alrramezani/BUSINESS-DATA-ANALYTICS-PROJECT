import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def make_file(path):
    # Load the user (telecom_churn) dataset from a CSV file
    df = pd.read_csv(path)
    # Display the first few rows of the DataFrame to get an initial view of the data
    print(df.head())
    # Print information about the DataFrame, including data types and missing values
    print(df.info())
    # Generate descriptive statistics for the DataFrame, such as mean, std, min, max, etc.
    print(df.describe())

    # Handle missing values (example)
    df.replace(" ", np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Convert target variable
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Normalize numerical features
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Print DataFrame information to verify changes and ensure data integrity
    print(df.info())
    # Save the cleaned data to a file for future analysis or reuse.
    df.to_csv('./data/clean_data.csv', index=False)
    
def load_data(path):
    make_file(path)
    # Load the cleaned dataset
    df = pd.read_csv("./data/clean_data.csv")
    return df

def EDA():
    # Create a folder for figures if it doesn't exist
    os.makedirs("./figs", exist_ok=True)

    # Churn distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df)
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("./figs/churn_distribution.png")
    plt.close()

    # Correlation heatmap (only for numerical features)
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig("./figs/feature_correlation.png")
    plt.close()

    # Distribution of numerical features
    numeric_cols = numeric_df.columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"./figs/distribution_{col}.png")
        plt.close()

    # Boxplots to compare churned vs non-churned customers
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Churn', y=col, data=df)
        plt.title(f"{col} vs Churn")
        plt.tight_layout()
        plt.savefig(f"./figs/boxplot_{col}.png")
        plt.close()

    # Categorical features vs Churn
    cat_cols = df.select_dtypes(include='object').columns.drop('Churn')
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, hue='Churn', data=df)
        plt.title(f"{col} by Churn")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"./figs/{col}_churn_countplot.png")
        plt.close()