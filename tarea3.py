# Tarea 3: Script para ANOVA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv('clean_dataset.csv')

# gb = df.groupby('formatted_work_type')
# df = pd.DataFrame([gb.get_group(x)['max_salary'] for x in gb.groups]).transpose()
# df.columns = [x for x in gb.groups]

# Diagramas de caja para ver la distribución por tratamientos
ax = sns.boxplot(x='formatted_work_type', y='max_salary', data=df, color='#99c2a2')
ax = sns.swarmplot(x="formatted_work_type", y="max_salary", data=df, color='#7d0013')
plt.show()

# Entrenar modelo de ANOVA
model = ols('max_salary ~ formatted_work_type', data=df).fit()

# Mostrar tabla de ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
