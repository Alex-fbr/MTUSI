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
from Lib import mglearn


import csv
def csv_reader(file_obj):
    # Read a csv file
    reader = csv.reader(file_obj)
    str = []
    for row in reader:
        print(" ".join(row))
        str += row;

if __name__ == "__main__":
    csv_path = "D:\data.csv"
    with open(csv_path, "r") as f_obj:
        csv_reader(f_obj)


dataset = np.loadtxt(csv_path, delimiter=";")
X = dataset[:, 0]
y = dataset[:, 1]
z = dataset[:, 2]

print("X.shape:", X.shape)
print("y.shape:", y.shape)
print("z.shape:", z.shape)



import matplotlib.pyplot as plt

# plot dataset
plt.plot(X, y, 'o')
plt.ylim(0, 150)
plt.xlabel("Дистанция, м")
plt.ylabel("Скорость, км/ч")
plt.show()
plt.close()


