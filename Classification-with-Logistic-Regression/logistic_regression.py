# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt

# Step 2: Load CSV Dataset
df = pd.read_csv("diagnosis.csv")  
print(df.head()) 

X = df.drop('diagnosis',axis=1)
y = df['diagnosis']

# Step 4: Handle Missing Values Using Imputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Predict Labels and Probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 9a: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 9b: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9c: ROC Curve and AUC
y_test_numerical = y_test.map({'B': 0, 'M': 1})
fpr, tpr, thresholds = roc_curve(y_test_numerical, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Step 10: Threshold Tuning
threshold = 0.7
y_custom = (y_proba >= threshold).astype(int)
y_test_numerical = y_test.map({'B': 0, 'M': 1})
print(confusion_matrix(y_test_numerical, y_custom))
print(classification_report(y_test_numerical, y_custom))