import numpy as np
from numpy import linalg
# Generate the random unit vector multiple times
d = 10
rand_uniform = np.random.uniform(0, 1, 10)
for x in rand_uniform:
    gaussian_x = (1 / (np.power(2 * np.pi, d / 2))) *\
                 (np.exp(-np.power(linalg.norm(x), 2) / 2))
    print(gaussian_x)
