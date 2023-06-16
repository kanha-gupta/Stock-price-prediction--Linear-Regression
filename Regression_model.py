import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# read dataset
dataset = pd.read_csv("dataset/TSLA.csv")
dataset.head()

# null value management
dataset.isnull().sum()
dataset.isna().any()
dataset.info

dataset.describe()

print(len(dataset))

# dataset plot
dataset['Open'].plot(figsize=(16,6))

x = dataset[['Open','High','Low','Volume']]
y = dataset['Close']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)

x_train.shape

x_test.shape


regressor = LinearRegression()

regressor.fit(x_train, y_train)

# printing regression values
print(regressor.coef_)

print(regressor.intercept_)

predicted = regressor.predict(x_test)
print(x_test)

predicted.shape

# tells Actual price & predicted price. Predicted price is derived from sklearn LinearRegression.predict method
dframe = pd.DataFrame(y_test, predicted)
dfr = pd.DataFrame({'Actual Price':y_test, 'Predicted Price': predicted})
print(dfr)


regressor.score(x_test, y_test)

print('Mean absolute error:' ,metrics.mean_absolute_error(y_test, predicted))
print('mean squared error:',metrics.mean_squared_error(y_test, predicted))
print('root mean squared error:', math.sqrt(metrics.mean_squared_error(y_test, predicted)))

graph = dfr.head(20)
graph.plot(kind='bar')

