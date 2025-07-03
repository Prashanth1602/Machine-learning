import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("data.csv")
print(df.describe())
print(df.info())
print(df.isnull().sum())

df['Age'].hist(bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()

correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

sns.pairplot(df)
plt.show()

fig = px.histogram(df, x="Age")
fig.show()