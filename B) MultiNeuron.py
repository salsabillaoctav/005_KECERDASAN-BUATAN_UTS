# Salsabilla Octavianingrum (21091397005)
# Coding Multi Neuron dengan Input Layer Feature 10 dan Neuron 5

# Inisiasi Numpy
import numpy as np

# Inisiasi input dengan 10 variabel
# Input layer feature berjumlah 10
inputs = [1.3, 4.3, 2.4, 3.5, 7.1, 2.8, 8.6, 9.3, 2.3, 1.2]

# Weights berjumlah 10 sesuai input
# jumlah weights 5 seperti neuron
weights = [[0.9, 0.5, 0.2, 0.1, 0.7, 0.8, 0.6, 0.3, 0.4, -0.7],
           [0.16, 0.19, -0.7, 0.2, 0.9, -0.23, -0.3, -0.27, 0.54, -0.28],
           [0.5, -0.3, 0.12, 0.1, 0.15, -0.24, 0.6, 0.54, 0.25, -0.8],
           [0.13, 0.6, -0.15, -0.17, 0.21, 0.27, 0.8, 0.1, -0.7, 0.12],
           [0.7, 0.9, 0.13, -0.6, -0.11, 0.29, 0.3, 0.17, 0.22, 0.41]]

# Jumlah bias 5 disamakan sesuai dengan jumlah Neuron
bias = [5.6, 3.4, 7.3, 9.2, 2.8]

# Output
layer_outputs = np.dot(weights,inputs) + bias
print(layer_outputs)
