# 🛍️ Mall Customer Segmentation using K-Means Clustering

This project performs **unsupervised learning** using the **K-Means clustering algorithm** to segment customers based on their **Annual Income** and **Spending Score**. It's a classic machine learning application in marketing analytics to better understand customer behavior.

---

## 🚀 Objective

To group mall customers into distinct clusters that reveal useful marketing insights such as:
- High-spending vs low-spending customers
- Conservative vs impulsive buyers
- Potential premium members

---

## 🧰 Tech Stack & Libraries

- Python
- pandas
- matplotlib
- seaborn
- scikit-learn

---

## 📂 Dataset

**Mall Customers Dataset**  
You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)

Filename: `Mall_Customers.csv`

---

## 📊 Steps Performed

1. **Data Loading and Preprocessing**  
   Cleaned and selected relevant features: `Annual Income (k$)` and `Spending Score (1-100)`.

2. **Elbow Method**  
   Used to determine the optimal number of clusters `K`.

3. **K-Means Clustering**  
   Applied KMeans algorithm to segment customers into distinct groups.

4. **Visualization**  
   Plotted clustered data using scatter plots.

5. **Evaluation**  
   Calculated **Silhouette Score** to assess clustering performance.
   > Achieved score: `~0.554`, indicating well-defined clusters.

---

## 📸 Output

- Elbow curve to determine K
- Scatter plots showing clusters
- Cluster centers
- Silhouette score printed in console

---

## 📝 How to Run

1. Clone the repository or download the project files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
