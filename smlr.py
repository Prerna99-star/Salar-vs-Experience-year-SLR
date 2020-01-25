# -*- coding: utf-8 -*-
"""
Simle Linear Regression 
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 1/3, random_state = 0 )

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting simple linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(XTrain, yTrain)

#Predicting the test set results
y_pred = regressor.predict(XTest)

#visualizing the Training set
plt.scatter(XTrain, yTrain, color = 'red')
plt.plot(XTrain, regressor.predict(XTrain), color = 'green')
plt.title('Salary vs Experience (Training Set)')
plt.xlable('Year of Experience')
plt.ylable('Salary')
plt.show()

#visualizing the Test set
plt.scatter(XTest, yTest, color = 'green')
plt.plot(XTrain, regressor.predict(XTrain), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlable('Year of Experience')
plt.ylable('Salary')
plt.show()
