import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import euclidean

delimiter = "\s+"
C2 = pd.read_csv("C2.txt", delimiter=delimiter, header=None)

# print(C2)

data_points = {}
index = 1
for rows in C2.itertuples():
    # append the list to the final list
    data_points.update({index: (rows._2, rows._3)})
    index += 1

# print(data_points)

data_points2 = {1: (0, 0), 2: (0, 7), 3: (3, 1), 4: (10, 0)}
k = 3
centers = {1: data_points2.get(1)}
array_ofj = np.random.uniform(1, 1, size=len(data_points2.keys()))
index = 2
for _ in range(k - 1):
    m = 0
    centers[index] = data_points2.get(1)
    for key, points in data_points2.items():
        current_center = centers.get(array_ofj[key - 1])
        distance = euclidean(points, current_center)
        if distance > m:
            m = distance
            centers[index] = points
    for key, points in data_points2.items():
        current_center = centers.get(array_ofj[key - 1])
        distance_j = euclidean(points, current_center)
        distance_i = euclidean(points, centers[index])
        if distance_j > distance_i:
            array_ofj[key - 1] = index
    index += 1
    pass
print(centers)

headers = ['Cluster_Group', "Center_Point", "X_values", "Y_values"]
gonzalez_report = pd.DataFrame([], columns=headers)
for key, value in data_points2.items():
    center = array_ofj[key - 1]
    value = list(value)
    rep = pd.DataFrame([["Cluster_Group" + str(center), centers.get(center), value[0], value[1]]], columns=headers)
    gonzalez_report = gonzalez_report.append(rep)

sns.scatterplot(data=gonzalez_report, x='X_values', y='Y_values', hue='Cluster_Group')

# print(gonzalez_report["Center_Group", "Center_Point"].groupby(by='Center_Point'))
plt.title("Gonzalez")
plt.show()
