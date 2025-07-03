# SVM Breast Cancer Classification

This project uses **Support Vector Machines (SVM)** for binary classification to predict whether a tumor is **malignant** or **benign** using the Breast Cancer dataset.

---

## 📊 Dataset

- `cancer.csv` (Breast Cancer Wisconsin Dataset)
- Uses only two features for 2D visualization:
  - `radius_mean`
  - `texture_mean`

---

## 🔍 Task Overview

1. Load and preprocess the dataset
2. Train two SVM models:
   - Linear Kernel
   - RBF (Radial Basis Function) Kernel
3. Plot decision boundaries
4. Tune hyperparameters `C` and `gamma` using `GridSearchCV`
5. Evaluate with cross-validation

---

## 🛠️ Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

---

## 🧪 How to Run

1. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the script**:

    ```bash
    python svm.py
    ```

3. You will see:
    - Accuracy and classification reports for both SVM models
    - Decision boundary plots for linear and RBF kernel
    - Best parameters from GridSearchCV
    - Cross-validation scores

---

## 📈 Output Example

- Accuracy (Linear): ~95%
- Accuracy (RBF): ~97%
- Best Parameters: e.g., `{'C': 10, 'gamma': 0.1}`
- Average Cross-Validation Accuracy: ~96%

---

## 📌 Notes

- Only 2 features used for simplicity and visualization.
- You can expand the model with more features for higher accuracy.
# support-vector-machines
