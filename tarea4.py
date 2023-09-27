# Tarea 4: Script para regresion lineal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('clean_dataset.csv')

X = df['applies'].fillna(0).values.reshape(-1, 1)
Y = df['max_salary'].fillna(0).values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('applies')
plt.ylabel('max_salary')
plt.show()
