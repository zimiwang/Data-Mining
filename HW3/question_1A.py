import pandas as pd
import numpy as np

t = 200
r_list = np.arange(5, 101, 5).tolist()
report = pd.DataFrame(columns=["b", "r"])
s = np.arange(0.01, 1.01, 0.01).tolist()
for i, r in enumerate(r_list):
    rep = pd.DataFrame([[t / r, r]], columns=["b", "r"])
    report = report.append(rep)
print(report)
