import pandas as pd

# Load the telecom_churn dataset from a CSV file located in the './data/' directory
df = pd.read_csv('./data/telecom_churn.csv')
# Display the first few rows of the DataFrame to get an initial view of the data
print(df.head())
# Print information about the DataFrame, including data types and missing values
print(df.info())
# Generate descriptive statistics for the DataFrame, such as mean, std, min, max, etc.
print(df.describe())