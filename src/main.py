import utils
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, roc_curve, auc
)

# -----------------------------
# Load and prepare data
# -----------------------------
df = utils.load_data('./data/telecom_churn.csv')
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# StratifiedKFold for reproducible CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluation function
def evaluate_model(name, model):
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    accuracy = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy').mean()
    precision = cross_val_score(pipeline, X, y, cv=skf, scoring='precision').mean()
    recall = cross_val_score(pipeline, X, y, cv=skf, scoring='recall').mean()
    f1 = cross_val_score(pipeline, X, y, cv=skf, scoring='f1').mean()
    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Run evaluations
results = [evaluate_model(name, model) for name, model in models.items()]

# Visualize model comparison
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

# -----------------------------
# Optional: Confusion matrix & ROC curve for final fit
# -----------------------------
# Fit on full data and plot confusion matrix and ROC curve
from sklearn.model_selection import train_test_split

# Final train-test split to visualize metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, cmap='Blues')
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.show()

# Plot ROC curve for each
def plot_roc(models_dict):
    plt.figure(figsize=(8, 6))
    for name, model in models_dict.items():
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_probs = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_roc(models)