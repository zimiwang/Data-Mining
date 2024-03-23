import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import time
#random.seed(seed = 57)

start_time = time.time()
n = 10000

#Problem A
#Generate random numbers in the domain [n] until two have the same value.
#How many random trials did this take? We will use k to represent this value.


def getK(n):
    k = 0
    arr = [False for i in range(n)]
    r1 = random.randint(0,n)
    arr[r1] = True
    while(1):
        k += 1
        r2 = random.randint(0,n)
        if arr[r2] == True:
            break
        else:
            arr[r2] = True
    return k
k = getK(n)

print("Q1A. This took {} random trials.".format(k))


#Problem B
#Repeat the experiment m = 300 times,
#and record for each time how many random trials this took.
#Plot this data as a cumulative density plot
#where the x-axis records the number of trials required k, and
#the y-axis records the fraction of experiments that succeeded (a collision)
#after k trials.
#The plot should show a curve that starts at a y value of 0,
#and increases as k increases, and eventually reaches a y value of 1.

m = 500

count = []
count_trials = []
cumulative = [0 for i in range(m)]
fraction = [0 for i in range(m)]

for i in range(m):
    k = getK(n)
    count_trials.append(k)
    count.append(1)
single_end_time = time.time()

d = {'k': count_trials, 'count': count,
     'cumulative': cumulative, 'Fraction': fraction}
df = pd.DataFrame(data = d)
df2 = df.sort_values(by = ['k'])
#df.groupby(by = ['k'])
np_df = df2.to_numpy()
np_df = np_df.astype(float)

for ind, rows in enumerate(np_df):
    if ind != 0:
        rows[2] = rows[1] + np_df[ind-1][2]
        rows[0] = float(rows[2]) / float(m)
    else:
        rows[2] = 1
        rows[0] = float(rows[2]) / float(m)

x = np_df[:,-1] #k
y = np_df[:, 0] #Cumulative Distribution

plt.plot(x, y)
plt.show()
