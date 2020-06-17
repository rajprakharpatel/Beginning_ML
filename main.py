"""
For testing models
"""
from models import LinearRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = pd.read_csv("data\\housing_ext.csv")
X = dataset['RM'].values.reshape(-1, 1)
y = dataset['MEDV'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
Lr = LinearRegressor.LinearRegressor(lr=0.0229, num_iter=2000)
Lr.train(X_train, y_train)

y_pred = Lr.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df.head(25).plot(kind='bar', figsize=(16, 9))
plt.grid(which='major', linestyle='-', linewidth='.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

