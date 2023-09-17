# Tarea 2: Script para estadistica descriptiva
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt 

df = pd.read_csv('clean_dataset.csv')
df.hist()
plt.show()

for column in df:
    if is_numeric_dtype(df[column]):
        print("\n*** Estadistica descriptiva para la columna:", column, "***\n")
        print("Promedio:", df[column].mean())
        print("Mediana:", df[column].median())
        print("Máximo:", df[column].max())
        print("Mínimo:", df[column].min())
