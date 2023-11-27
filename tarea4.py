# Tarea 4: Script para regresion lineal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('clean_dataset.csv')

X = df['views'].fillna(0).values.reshape(-1, 1)
Y = df['applies'].fillna(0).values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

print(f'a = {linear_regressor.coef_}\nb = {linear_regressor.intercept_}\nR^2 = {linear_regressor.score(X, Y)}')

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('views')
plt.ylabel('applies')
plt.show()
