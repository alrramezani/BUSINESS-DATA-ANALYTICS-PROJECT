import os
import utils
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, auc
)

# Ensure the cleaned dataset exists; if not, generate it
if not os.path.exists("./data/clean_data.csv"):
    utils.make_file()

# Load the cleaned dataset
df = pd.read_csv("./data/clean_data.csv")

# Run exploratory data analysis from the utils module
utils.EDA()

# Define features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize classifiers
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)

# Train classifiers
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Function to compute evaluation metrics
def evaluate(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

# Evaluate all models and collect results
results = [
    evaluate(y_test, y_pred_lr, "Logistic Regression"),
    evaluate(y_test, y_pred_dt, "Decision Tree"),
    evaluate(y_test, y_pred_rf, "Random Forest")
]

# Visualize the performance of all models using a bar chart
def plot_model_metrics(results):
    df_results = pd.DataFrame(results)
    df_results.set_index("Model", inplace=True)
    df_results.plot(kind='bar', figsize=(10, 6), colormap='Set2')
    plt.title("Model Comparison - Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

plot_model_metrics(results)

# Display confusion matrices for each model
for model, name in zip([lr, dt, rf], ["Logistic Regression", "Decision Tree", "Random Forest"]):
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.show()

# Plot ROC curves and calculate AUC scores for each model
def plot_roc(models, names):
    plt.figure(figsize=(8, 6))
    for model, name in zip(models, names):
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_roc([lr, dt, rf], ["Logistic Regression", "Decision Tree", "Random Forest"])