import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("cancer.csv")
df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df[['radius_mean', 'texture_mean']].values  # input features
y = df['diagnosis'].values                      # target labels

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

y_pred = svm_linear.predict(X_test)

print("Accuracy (Linear):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

svm_rbf = SVC(kernel='rbf', C=1, gamma=0.5)
svm_rbf.fit(X_train, y_train)

y_pred_rbf = svm_rbf.predict(X_test)
print("Accuracy (RBF):", accuracy_score(y_test, y_pred_rbf))
print(classification_report(y_test, y_pred_rbf))

def plot_decision_boundary(clf, X, y, title):
    h = 0.01
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.show()

# Plot for both
plot_decision_boundary(svm_linear, X_train, y_train, "Linear Kernel SVM")
plot_decision_boundary(svm_rbf, X_train, y_train, "RBF Kernel SVM")

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Test Accuracy with Best Model:", grid.best_estimator_.score(X_test, y_test))

scores = cross_val_score(grid.best_estimator_, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Average Accuracy:", scores.mean())
