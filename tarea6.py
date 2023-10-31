# Tarea 6: Script para k-means clustering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.cluster import KMeans
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
    plt.savefig(file_path)
    plt.show()

df = pd.read_csv('clean_dataset.csv')

df = df.assign(title=df['title'].astype('category').cat.codes)
df = df.assign(location=df['location'].astype('category').cat.codes)
df = df.fillna(0)

print(df)
scatter_group_by("kmeansTruthGroups.png", df, "applies", "views", "formatted_work_type")

X = df[['applies', 'views']]
Y = df['formatted_work_type']
MinMaxScaler = MinMaxScaler()
X_minmax = MinMaxScaler.fit_transform(X)
data = pd.DataFrame(X_minmax,columns=['applies', 'views'])

X_train, X_test, y_train, y_test = train_test_split(data, Y,test_size=0.2, random_state = 1)


X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

kmeans = KMeans(n_clusters = 6, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)

sns.scatterplot(data = X_train, x = 'applies', y = 'views', hue = kmeans.labels_)
plt.show()
