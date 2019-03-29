import sys
print("версия Python: {}".format(sys.version))
import pandas as pd
print("версия pandas: {}".format(pd.__version__))
import matplotlib
print("версия matplotlib: {}".format(matplotlib.__version__))
import numpy as np
print("версия NumPy: {}".format(np.__version__))
import scipy as sp
print("версия SciPy: {}".format(sp.__version__))
import IPython
print("версия IPython: {}".format(IPython.__version__))
import sklearn
print("версия scikit-learn: {}".format(sklearn.__version__))
import mglearn
print("версия mglearn: {}".format(mglearn.__version__))
import matplotlib.pyplot as plt

import csv
def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    str = []
    for row in reader:
        print(" ".join(row))
        str += row;





if __name__ == "__main__":
    csv_path = "D:\dat.csv"
    with open(csv_path, "r") as f_obj:
        csv_reader(f_obj)


dataset = np.loadtxt(csv_path, delimiter=";")
# separate the data from the target attributes
X = dataset[:, 0]
y = dataset[:, 1]

print("X.shape:", X.shape)
print("y.shape:", y.shape)



import matplotlib.pyplot as plt

# plot dataset
plt.plot(X, y, 'o')
plt.ylim(0, 150)
plt.xlabel("Скорость, км/ч")
plt.ylabel("Дистанция, м")
#plt.show()
plt.close()




from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# строим модель кластеризации
kmeans = KMeans(n_clusters=10)
kmeans.fit(dataset)
points = kmeans.labels_
print("Принадлежность к кластерам:\n{}".format(points))

from collections import Counter
c = Counter(points)
print("Всего точек:\n{}".format(points.size))
print("Принадлежность к кластерам:\n{}".format(c))

mglearn.discrete_scatter(dataset[:, 0], dataset[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter( kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2,3,4,5,6,7,8,9], markers='^', markeredgewidth=5)
plt.legend(["кластер 1", "кластер 2","кластер 3","кластер 4","кластер 5","кластер 6","кластер 7",
            "кластер 8","кластер 9","кластер 10"], loc='best')
plt.xlabel("Скорость, км/ч")
plt.ylabel("Дистанция, м")
plt.show()
print("График")
plt.close()


plt.show()
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=10)
assignment = agg.fit_predict(dataset)
mglearn.discrete_scatter(dataset[:, 0], dataset[:, 1], assignment)
plt.legend(["кластер 1", "кластер 2","кластер 3","кластер 4","кластер 5","кластер 6","кластер 7",
            "кластер 8","кластер 9","кластер 10"], loc='best')
plt.xlabel("Скорость, км/ч")
plt.ylabel("Дистанция, м")
plt.show()
print("График")





# импортируем функцию dendrogram и функцию кластеризации ward из SciPy
from scipy.cluster.hierarchy import dendrogram, ward
# применяем кластеризацию ward к массиву данных dataset
# функция SciPy ward возвращает массив с расстояниями
# вычисленными в ходе выполнения агломеративной кластеризации
linkage_array = ward(dataset)
# теперь строим дендрограмму для массива связей, содержащего расстояния
# между кластерами
dendrogram(linkage_array)
# делаем отметки на дереве, соответствующие двум или трем кластерам
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, ' два кластера', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' три кластера', va='center', fontdict={'size': 15})
plt.xlabel("Индекс наблюдения")
plt.ylabel("Кластерное расстояние")

