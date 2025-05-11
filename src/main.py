# main.py
import utils
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load and clean data
df = utils.load_data('./data/telecom_churn.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify numerical and categorical features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Models and hyperparameter grids
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {'clf__C': [0.01, 0.1, 1, 10]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(),
        {'clf__max_depth': [3, 5, 10, None]}
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [5, 10, None]
        }
    )
}

# Function to evaluate models
def evaluate_models(X, y):
    results = []
    final_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (model, param_grid) in models.items():
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('clf', model)
        ])
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X, y)

        best_model = grid.best_estimator_
        final_models[name] = best_model

        # Cross-validated predictions
        y_pred = best_model.predict(X)
        y_proba = best_model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_score = auc(fpr, tpr)

        results.append({
            "Model": name,
            "Best Params": grid.best_params_,
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "AUC": auc_score
        })

    return results, final_models

# Evaluate and get best models
results, final_models = evaluate_models(X, y)

# Visualize model metrics
def plot_model_metrics(results):
    df_results = pd.DataFrame(results).set_index("Model")
    df_results[["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]].plot(
        kind='bar', figsize=(12, 6), colormap='Set2'
    )
    plt.title("Model Comparison - Evaluation Metrics (5-Fold CV)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

plot_model_metrics(results)

# Display confusion matrix
for name, model in final_models.items():
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y, cmap='Blues')
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.show()

# Plot ROC curves
def plot_roc(models_dict):
    plt.figure(figsize=(8, 6))
    for name, model in models_dict.items():
        if hasattr(model.named_steps['clf'], "predict_proba"):
            y_probs = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_probs)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve - Full Dataset")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_roc(final_models)
