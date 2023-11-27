# Tarea 5: Script para k-nearest neighbors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def scatter_group_by(file_path, df, x_column, y_column, label_column):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    ax.set_ylabel("Salario máximo")
    ax.set_xlabel("Ubicación")
    plt.xticks(rotation='vertical')
    plt.savefig(file_path)
    plt.show()

df = pd.read_csv('clean_dataset.csv')

df = df.assign(title=df['title'].astype('category').cat.codes)
df = df.fillna(0)

print(df)
scatter_group_by("knnGroups.png", df, "location", "max_salary", "formatted_work_type")

df = df.assign(location=df['location'].astype('category').cat.codes)

X = df[['location', 'max_salary']]
Y = df['formatted_work_type']
MinMaxScaler = MinMaxScaler()
X_minmax = MinMaxScaler.fit_transform(X)
data = pd.DataFrame(X_minmax,columns=['location', 'max_salary'])

X_train, X_test, y_train, y_test = train_test_split(data, Y,test_size=0.2, random_state = 1)
knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train,y_train)
ypred=knn_clf.predict(X_test)

cm = confusion_matrix(y_test, ypred)
print("Confusion Matrix:")
print(cm)
cr = classification_report(y_test, ypred)
print("Classification Report:",)
print(cr)
acc = accuracy_score(y_test,ypred)
print("Accuracy:", acc)

df_cm = pd.DataFrame(cm, ['Contract', 'Full-time', 'Internship', 'Part-time', 'Temporary'], ['Contract', 'Full-time', 'Internship', 'Part-time', 'Temporary'])
sns.heatmap(df_cm)
plt.show()

