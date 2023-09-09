# Tarea 1: Script para limpiar datos
import pandas as pd

df = pd.read_csv('job_postings.csv')

# Eliminar todas las filas donde no haya información sobre el mínimo y máximo salario
df = df.dropna(subset=['min_salary', 'max_salary'])[['title', 'min_salary', 'med_salary', 'max_salary', 'location', 'remote_allowed', 'applies', 'views', 'listed_time', 'formatted_work_type']]

# Convertir la columna listed_time a objeto Datetime desde UNIX timestamp
df['listed_time'] = pd.to_datetime(df['listed_time'], unit='ms')

# Convertir la columna de remote allowed de enteros a booleanos
df['remote_allowed'] = df['remote_allowed'].apply(lambda val: True if (val == 1) else False)

print(df)
print(df.shape[0])

df.to_csv('clean_dataset.csv', encoding='utf-8', index=False)
