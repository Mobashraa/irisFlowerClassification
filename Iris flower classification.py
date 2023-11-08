#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification Dataset
# 
# The Iris flower classification dataset comprises 150 samples of Iris flowers, categorized into three species:
# Iris setosa
# Iris versicolor
# Iris virginica
# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# Attribute Information:
# 
# Sepal length in cm
# Sepal width in cm
# Petal length in cm
# Petal width in cm
# Class
# The dataset is widely used as a benchmark in machine learning for supervised classification tasks aiming to accurately classify Iris flowers based on their measurements

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("Iris.csv")
df.head()


# # Processing the dataset

# In[4]:


# Delete the column
df=df.drop(columns="Id")


# In[5]:


df.head()


# In[6]:


# to display stats about the data
df.describe()


# In[7]:


# basic  info of datatype in dataset
df.info()


# In[8]:


# Display the number of samples on each class
df["Species"].value_counts()


# In[9]:


# Check for the null values
df.isnull().sum()


# # Exploratory Data Analysis

# In[10]:


# histograms
df['SepalLengthCm'].hist()


# In[11]:


df['SepalWidthCm'].hist()


# In[12]:


df['PetalLengthCm'].hist()


# In[13]:


df['PetalWidthCm'].hist()


# # Pairplot
# 

# In[15]:


sns.pairplot(df,hue='Species')


# # Relationship between species and sepal length

# In[23]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Species',y='SepalLengthCm',data=df.sort_values('SepalLengthCm',ascending=False))


# # Relationship between species and sepal width

# In[24]:


df.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm')


# # Relationship between sepal width and sepal length

# In[25]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, size=5)


# # Pairplot

# In[26]:


sns.pairplot(df, hue="Species", size=3)


# # Model training

# In[18]:


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


# # Model 1

# In[19]:


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)
accuracy_logreg = model1.score(x_test, y_test) * 100
print("Accuracy (Logistic Regression): ", accuracy_logreg)


# 
# 
# 
# # Model 2

# In[20]:


# K-nearest Neighbours Model (KNN)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier()
model2.fit(x_train, y_train)
accuracy_knn = model2.score(x_test, y_test) * 100
print("Accuracy (KNN): ", accuracy_knn)


# # Model 3

# In[21]:


# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train)
accuracy_decision_tree = model3.score(x_test, y_test) * 100
print("Accuracy (Decision Tree): ", accuracy_decision_tree)


# # Project Report

# In[22]:


# Model Comparison - Visualization
models = ['Logistic Regression', 'KNN', 'Decision Tree']
accuracies = [accuracy_logreg, accuracy_knn, accuracy_decision_tree]

plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison - Accuracy")
plt.ylim([0, 100])
plt.show()


# In[ ]:




