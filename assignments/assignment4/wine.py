
import numpy as np

data = np.load_csv("winequality-white.csv")

X = data[:len(data - 1)]
y = data[len(data - 1)]


