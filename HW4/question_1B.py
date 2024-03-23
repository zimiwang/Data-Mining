from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

delimiter = "\s+"
C1 = pd.read_csv("C1.txt", delimiter=delimiter, header=None)

n_clusters = 3
clustering = AgglomerativeClustering(n_clusters=3).fit(C1)


print(clustering.labels_)
