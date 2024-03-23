import math
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import timeit


# The code for question 1A
def question_1A(n):
    k = 0
    status = []
    # Create a random boolean array with all False
    for i in range(n):
        status.append(False)
    # Randomly choose one element as the True
    status[random.randint(0, n)] = True
    while 1:
        k += 1
        # Generate another random number
        random_num = random.randint(0, n - 1)
        if status[random_num]:
            break
        else:
            status[random_num] = True
    return k


print("How many random trials did this take? Answer:", question_1A(10000))

# The code foe question 1B
# def question_1B(m, n):
# question_1B(500, 10000)
start_time = timeit.default_timer()

m = 500
n = 10000
count = []
trials = []
cumulative_x = []
fraction_y = []

for i in range(m):
    cumulative_x.append(0)
    fraction_y.append(0)

for j in range(m):
    k = question_1A(n)
    trials.append(k)
    count.append(1)


# trials = trials.sort(reverse=False)

# Question 1C
def question_1C(k, m):
    k_sum = sum(k)
    estimate = k_sum / float(m)

    return estimate

print('Empirically estimate the expected number of k random trials in order to have a collision. Answer:',
      question_1C(trials, 500))

end_time = timeit.default_timer()
C_time = start_time - end_time
print("How long did 1.C take for n = 10, 000 and m = 500 trials? Answer:", C_time)

# Question 1D
M = [500, 1000, 5000, 10000]
N = [10000, 100000, 500000, 1000000]
total_times = []

# Get K and avoid using list
def get_k(n):
    k = 0
    rand_1 = random.randint(0, n)
    while 1:
        k += 1
        rand_2 = random.randint(0, n)
        if rand_1 == rand_2:
            break
    return k

# The code of Question 1D
def question_1D(N, M):
    for m in M:
        print("Current m=", m)
        # record all different cases
        times = []
        for n in N:
            print("Current n=", n)
            start_time = time.time()
            for i in range(m):
                k = get_k(n)
            end_time = time.time()
            current_time = start_time - end_time
            times.append(current_time)
        total_times.append(times)

    # plot the time graph
    plt.plot(N, total_times[0])
    plt.plot(N, total_times[1])
    plt.plot(N, total_times[2])
    plt.plot(N, total_times[3])
    plt.legend(['m = 500', 'm = 1000', 'm = 5000', 'm = 10000'], loc='upper left')
    plt.show()

question_1D(N, M)
