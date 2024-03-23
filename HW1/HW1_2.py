import math
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import timeit

m = 500
n = 1000

# The code for question 2A
def question_2A(n):
    k = 0
    status = []
    count = 0
    # Create a random boolean array with all 0
    for i in range(n):
        status.append(0)
    while 1:
        k += 1
        random_num = random.randint(0, n - 1)
        if status[random_num] >= 1:
            status[random_num] += 1
        else:
            count += 1
            status[random_num] = 1
        if count == n:
            break
    return k


print("How many random trials did this take? Answer:", question_2A(n))


# The code for question 2B
start_time = timeit.default_timer()

count = []
trials = []
cumulative_x = []
fraction_y = []

for i in range(m):
    cumulative_x.append(0)
    fraction_y.append(0)

for j in range(m):
    k = question_2A(n)
    trials.append(k)
    count.append(1)


# The code for question 2C
def question_2C(k, m):
    k_sum = sum(k)
    estimate = k_sum / float(m)

    return estimate


print('Empirically estimate the expected number of k random trials in order to have a collision. Answer:',
      question_2C(trials, 500))

end_time = timeit.default_timer()
C_time = start_time - end_time
print("How long did 1.C take for n = 10, 000 and m = 500 trials? Answer:", C_time)


# Question 1D
M = [500, 1000, 2000, 5000]
N = [1000, 5000, 10000, 20000]
total_times = []


def question_2D(N, M):
    for m in M:
        print("Current m=", m)
        times = []
        for n in N:
            print("Current n=", n)
            start_time = time.time()
            for i in range(m):
                k = question_2A(n)
            end_time = time.time()
            current_time = start_time - end_time
            times.append(current_time)
        total_times.append(times)

    plt.plot(N, total_times[0])
    plt.plot(N, total_times[1])
    plt.plot(N, total_times[2])
    plt.plot(N, total_times[3])
    plt.legend(['m = 500', 'm = 1000', 'm = 5000', 'm = 10000'], loc='upper left')
    plt.show()


question_2D(N, M)
