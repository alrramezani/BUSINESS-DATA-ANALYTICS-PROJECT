import os
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Check if the CSV file exists and load the cleaned dataset from './data/'
if not os.path.exists("./data/clean_data.csv"):
    utils.make_file()
df = pd.read_csv("./data/clean_data.csv")

#Exploratory Data Analysis
utils.EDA(df)

# Split the dataset into features (X) and target variable (y)
X = df.drop('Churn', axis=1)  # Features
y = df['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # Allocate 30% of the data for testing
    random_state=42  # Ensure reproducibility of the split
)

# Train a logistic regression model to predict the target variable
lr = LogisticRegression()  # Create a logistic regression model instance
lr.fit(X_train, y_train)  # Train the model using the training data
# Evaluate the model's performance by making predictions on unseen data
y_pred_lr = lr.predict(X_test)  # Predict outcomes for the test set

# Train a decision tree classifier to predict the target variable
dt = DecisionTreeClassifier()  # Create a decision tree classifier instance
dt.fit(X_train, y_train)  # Train the model using the training data
# Evaluate the model's performance by making predictions on unseen data
y_pred_dt = dt.predict(X_test)  # Predict outcomes for the test set

# Train a random forest classifier to predict the target variable
rf = RandomForestClassifier(n_estimators=100)  # Create a random forest classifier with 100 trees
rf.fit(X_train, y_train)  # Train the model using the training data
# Evaluate the model's performance by making predictions on unseen data
y_pred_rf = rf.predict(X_test)  # Predict outcomes for the test set


# Function to evaluate the performance of a model using key metrics
def evaluate(y_true, y_pred, model_name):
    """
    Prints key performance metrics for a given model.
    
    Parameters:
    - y_true: Actual target values
    - y_pred: Predicted target values
    - model_name: Name of the model being evaluated
    """
    print(f"--- {model_name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))  # Overall correctness of predictions
    print("Precision:", precision_score(y_true, y_pred))  # How precise are positive predictions?
    print("Recall:", recall_score(y_true, y_pred))  # How well are positive instances detected?
    print("F1 Score:", f1_score(y_true, y_pred))  # Balanced measure of precision and recall
    print()  # Empty line for readability

# Evaluate the performance of each trained model
evaluate(y_test, y_pred_lr, "Logistic Regression")
evaluate(y_test, y_pred_dt, "Decision Tree")
evaluate(y_test, y_pred_rf, "Random Forest")
