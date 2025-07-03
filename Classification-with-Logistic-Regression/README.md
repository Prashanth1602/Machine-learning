# Logistic Regression Binary Classification

This project demonstrates a complete machine learning pipeline using **Logistic Regression** for binary classification. It includes data preprocessing, missing value imputation, feature scaling, model training, evaluation, and threshold tuning.

## 📊 Dataset

The dataset used is a binary classification dataset (e.g., Breast Cancer Wisconsin Dataset), provided as a CSV file. Make sure your dataset has a column for the target (e.g., `target`, `diagnosis`, etc.).

## 🔧 Tools & Libraries

- pandas
- numpy
- scikit-learn
- matplotlib

## ✅ Workflow

1. Load the dataset from CSV
2. Handle missing values using `SimpleImputer`
3. Split data into training and testing sets
4. Standardize features using `StandardScaler`
5. Train a logistic regression model
6. Make predictions and calculate class probabilities
7. Evaluate model using:
   - Confusion matrix
   - Classification report (Precision, Recall, F1-score)
   - ROC-AUC curve
8. Tune classification threshold manually

## 🛠 How to Run

1. Make sure your dataset (e.g., `breast_cancer.csv`) is in the project folder.
2. Run the Python script:
   ```bash
   python logistic_regression_classifier.py
