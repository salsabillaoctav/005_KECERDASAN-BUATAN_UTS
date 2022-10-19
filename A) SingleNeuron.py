# Salsabilla Octavianingrum (21091397005)
# Coding Single Neuron serta Input Layer Feature 10 dan Neuron 1

# Inisiasi Numpy
import numpy as np

# Inisiasi input dengan 10 variabel
# Input layer feature yaitu 10
inputs = [3, 6, 9, 1, 4, 7, 10, 2, 8, 5]

# Weights berjumlah 10 sesuai input
# jumlah weights 1 seperti neuron
weights = [0.2, 0.7, 0.9, 0.1, 0.8, 0.3, 0.4, 0.6, 0.5, -0.10]

# Jumlah bias 1 disamakan sesuai dengan jumlah Neuron
bias = 7

# Output
output = np.dot(weights,inputs) + bias
print(output)
