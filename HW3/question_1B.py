import pandas as pd
import math

# Here I choose b=13.3 and r=15
best_b = 13.3
best_r = 15
pairs = {"AB": 0.77, "AC": 0.25, "AD": 0.33, "BC": 0.20, "BD": 0.55, "DC": 0.91}
final_report = pd.DataFrame(columns=["Pair", "Probability", "b", "r", "Similarity"])
for key, p in pairs.items():
    f_s = 1 - math.pow(1 - math.pow(p, best_b), best_r)
    temp = pd.DataFrame([[key, f_s, best_b, best_r, p]], columns=["Pair", "Probability", "b", "r", "Similarity"])
    final_report = final_report.append(temp)

print(final_report)
