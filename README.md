# 📊 Telecom Customer Churn Prediction

## 🗂️ Project Structure

```text
📦 telecom-churn-prediction/
├── 📁 data/        # Raw and processed datasets
├── 📁 notebooks/   # Jupyter notebooks for EDA and analysis
├── 📁 src/
│   ├── main.py     # Main script to run the project
│   └── utils.py    # Helper functions
└── README.md       # Project documentation (this file)
```
---

## 📁 Dataset

- **Source**: Publicly available on [Kaggle](https://www.kaggle.com/)
- **Size**: 7,043 customer records
- **Features**: 21 columns including customer demographics, account info, contract details, and service usage
- **Quality**: Well-structured, minimal cleaning needed
- **Feature Types**: Mix of categorical and numerical
- **Use Case**: Widely used for churn prediction tasks

---

## 📌 Problem Definition

Customer churn (i.e., customers leaving the service) significantly impacts revenue in the telecom industry. The aim is to build a predictive model that identifies customers likely to churn, enabling proactive retention strategies.

---

## ⚠️ Challenges & Impact

- Determine key features that lead to churn
- Evaluate and compare various machine learning models
- Help businesses retain high-risk customers by:
  - Offering targeted discounts
  - Recommending better service plans

---

## 🔍 Analytical Approach

### 🔧 Preprocessing (in `utils.py`)
- Handle missing values
- Encode categorical features (One-Hot or Label Encoding)
- Normalize/scale numerical features

### 📊 Exploratory Data Analysis (in `notebooks/eda.ipynb`)
- Visualize churn distribution
- Analyze trends across customer attributes (e.g., contract type, tenure, payment method)

### 🤖 Modeling (in `main.py`)
Models trained and evaluated:
- **Logistic Regression** (baseline model)
- **Decision Tree**
- **Random Forest**

### 📈 Evaluation Metrics
Models will be evaluated using the following:
- Accuracy
- Precision
- Recall
- F1 Score

> 📌 Evaluation will guide model selection and business recommendations.

### ⭐ Feature Importance
- Analyze which features most strongly influence customer churn (e.g., contract type, tenure, monthly charges)

---

## ✅ Expected Outcomes

- Accurate churn prediction model
- Insights into customer behavior and churn risks
- Business-friendly recommendations for reducing churn

---

## 📦 Deliverables

- 📁 Cleaned and preprocessed dataset (in `data/`)
- 🐍 Python scripts for data pipeline and modeling (in `src/`)
- 📓 Interactive EDA and modeling notebooks (in `notebooks/`)
- 📄 Final report covering:
  - Business insights
  - Data analysis
  - Model evaluation
  - Actionable recommendations

---

_Developed as part of my MSc final project._