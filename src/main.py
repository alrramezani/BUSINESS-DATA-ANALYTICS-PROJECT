import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder


# Load the telecom_churn dataset from a CSV file located in the './data/' directory
df = pd.read_csv('./data/telecom_churn.csv')
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
