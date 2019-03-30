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


if __name__ == "__main__":
    csv_path = "D:\dat.csv"
dataset = np.loadtxt(csv_path, delimiter=";")
# separate the data from the target attributes
from sklearn.datasets import make_blobs

plt.show()
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=10)
assignment = agg.fit_predict(dataset)
mglearn.discrete_scatter(dataset[:, 0], dataset[:, 1], assignment)

print("BuildAgglomerativeClustering")




