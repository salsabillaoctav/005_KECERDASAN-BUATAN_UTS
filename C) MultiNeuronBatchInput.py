# Salsabilla Octavianingrum (21091397005)
# Coding Multi Neuron Batch dengan Input Layer Feature 10 dan Neuron 5
# Per Batch 6 Input

# Inisiasi Numpy
import numpy as np

# Inisiasi variabel
# Input layer feature berjumlah 10
# Per Batch 6 Input
inputs = [[0.4, 2.4, 5.4, 2.7, 3.7, 7.4, 3.9, 5.9, 3.2, 3.0],
          [0.3, 2.8, 9.1, 5.8, 3.2, 1.0, 5.7, 2.5, 4.0, 4.5],
          [1.7, 7.2, 7.6, 6.7, 4.9, 4.4, 3.5, 9.8, 4.6, 3.6],
          [2.6, 3.6, 1.4, 6.5, 4.7, 4.9, 2.0, 3.8, 7.6, 9.0],
          [0.9, 6.3, 3.3, 3.9, 9.9, 0.6, 6.4, 6.5, 7.2, 4.7],
          [3.4, 2.9, 5.8, 2.4, 3.5, 8.3, 8.0, 8.7, 9.1, 4.6]]

# Weights berjumlah 10 sesuai input
# jumlah weights 5 seperti neuron
weights = [[3.5, 1.0, 7.9, 1.2, 7.3, 4.8, 3.8, 7.5, 4.9, 2.6],
           [5.8, 8.9, 6.7, 5.7, 4.9, 6.8, 2.6, 5.9, 7.8, 9.8],
           [3.0, 4.7, 4.9, 5.8, 8.3, 5.5, 3.9, 3.5, 1.4, 9.9],
           [2.8, 2.5, 4.9, 7.9, 7.5, 7.8, 2.6, 5.3, 6.8, 7.3],
           [4.9, 3.5, 9.5, 6.9, 2.3, 3.9, 2.9, 4.1, 8.4, 7.9]] 

bias = [3.1, 2.3, 3.4, 1.2, 3.6]

# Output
layer_outputs = np.dot(inputs,np.array(weights).T) + bias
print(layer_outputs)
