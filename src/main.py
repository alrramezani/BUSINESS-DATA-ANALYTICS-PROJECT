import os
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import clean_data_maker

# Check if the CSV file exists and load the cleaned dataset from './data/'
if not os.path.exists("./data/clean_data.csv"):
    clean_data_maker.make_file()
df = pd.read_csv("./data/clean_data.csv")

# Churn distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.savefig("./figs/churn_distribution")
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.savefig("./figs/feature_correlation")

