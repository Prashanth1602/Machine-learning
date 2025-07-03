import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])
X = df.drop("Species", axis=1)  
y = df["Species"]  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_initial = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k_initial)
knn_classifier.fit(X_train_scaled, y_train)
y_pred = knn_classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
target_names = y.unique()
target_names.sort()

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for K={k_initial}')
plt.show()

k_values = list(range(1, 31)) # Test K from 1 to 30
accuracies = []
conf_matrices = {} # To store confusion matrices for different K if needed

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
    conf_matrices[k] = confusion_matrix(y_test, y_pred_k)

plt.plot(k_values, accuracies, marker='o', linestyle='-', color='skyblue')
plt.title('KNN Accuracy vs. K Value')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

optimal_k_index = np.argmax(accuracies)
optimal_k = k_values[optimal_k_index]
highest_accuracy = accuracies[optimal_k_index]

print(f"\nOptimal K found: {optimal_k} with accuracy: {highest_accuracy:.4f}")

X_2d = X[['SepalLengthCm', 'PetalLengthCm']]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)


knn_2d = KNeighborsClassifier(n_neighbors=optimal_k)
knn_2d.fit(X_train_2d_scaled, y_train_2d)

x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis)


plt.scatter(X_train_2d_scaled[:, 0], X_train_2d_scaled[:, 1], c=y_train_2d,
            cmap=plt.cm.viridis, edgecolor='k', s=20, label='Training points')

plt.scatter(X_test_2d_scaled[:, 0], X_test_2d_scaled[:, 1], c=y_test_2d,
            cmap=plt.cm.viridis, edgecolor='k', s=50, marker='X', label='Test points')
