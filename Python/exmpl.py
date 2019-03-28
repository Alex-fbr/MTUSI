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

plt.show()
# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:", X.shape)
print("y.shape:", y.shape)
plt.close()

#-----------------------------------------------------
plt.show()
X, y = mglearn.datasets.make_wave(n_samples=100)
plt.plot(X, y, 'o')
plt.ylim(0, 5)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.close()
#-----------------------------------------------------
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Ключи cancer(): \n{}".format(cancer.keys()))

print("Форма массива data для набора cancer: {}".format(cancer.data.shape))
print("Количество примеров для каждого класса:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Имена признаков:\n{}".format(cancer.feature_names))