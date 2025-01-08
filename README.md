import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, header=None, names=columns)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(iris_data.head())

# Basic statistics
print("\nBasic statistics:")
print(iris_data.describe())

# Check for missing values
print("\nMissing values in each column:")
print(iris_data.isnull().sum())

# Data visualization
# Pairplot to visualize relationships between features
sns.pairplot(iris_data, hue='species')
plt.title('Pairplot of Iris Dataset')
plt.show()

# Boxplot to visualize the distribution of sepal length by species
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=iris_data)
plt.title('Boxplot of Sepal Length by Species')
plt.show()

# Correlation matrix
correlation_matrix = iris_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Grouping data by species and calculating the mean
mean_values = iris_data.groupby('species').mean()
print("\nMean values by species:")
print(mean_values)

# Save the processed data to a new CSV file
mean_values.to_csv('iris_mean_values.csv')
print("\nMean values saved to 'iris_mean_values.csv'")
