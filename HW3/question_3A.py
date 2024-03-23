import pandas as pd
import numpy as np
from random import gauss
import matplotlib.pyplot as plt
from itertools import combinations

d = 100
t = 200

vector_list = {}
l = np.array(range(d)) + 1
pair_list = list(combinations(l, 2))
n_pair = len(pair_list)
final_report = pd.DataFrame([], columns=["Pair", "dot Product"])
for t in range(t):
    output = []
    for i in range(d):
        output.append(gauss(0, 1))
    vector_list.update({t + 1: output})
for pair in pair_list:
    prod = np.dot(vector_list.get(pair[0]), vector_list.get(pair[1]))
    final_report = final_report.append(pd.DataFrame([[pair, prod]], columns=["Pair", "dot Product"]))

final_report.hist(bins=n_pair * 2, cumulative=True, density=True)
plt.show()
