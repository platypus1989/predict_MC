import numpy as np
import pandas as pd

n = 10000
x = np.random.normal(size = n)

y = np.cos(x) + np.random.normal(scale = 0.1, size = n)

np.mean(y)


y = np.sin(x) + np.random.normal(scale = 0.1, size = n)

np.mean(y)


y = np.power(x, 2) + np.random.normal(scale = 0.1, size = n)

np.mean(y)
