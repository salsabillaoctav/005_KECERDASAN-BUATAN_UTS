# Multi Neuron Batch Input menggunakan Input Layer Feature 10 
# Per batch nya 6 Input
# Hidden Layer 1, 5 Neuron
# Hidden Layer 2, 3 Neuron

# Inisialisasi Numpy
import numpy as np

# Inisialisasi variabel
# Input layer feature 10 dan Per Batch nya 6 Input
inputs = [[9.4, 4.4, 7.2, 8.8, 2.9, 8.3, 3.8, 7.9, 8.2, 2.8],
          [7.5, 6.9, 1.6, 9.2, 1.5, 8.6, 2.8, 3.0, 2.8, 3.9],
          [9.3, 2.6, 8.8, 7.9, 9.6, 7.3, 6.9, 3.8, 2.8, 2.3],
          [2.6, 6.8, 3.9, 1.6, 5.2, 2.2, 9.3, 2.9, 8.9, 2.8],
          [6.8, 2.7, 7.5, 2.8, 3.7, 9.8, 2.6, 7.9, 9.2, 2.5],
          [4.8, 6.2, 2.8, 4.9, 8.0, 2.4, 5.9, 4.7, 9.3, 5.8]]

# Weights 1 sesuai dengan panjang Input yaitu 10
# Jumlah Weights 1 sama dengan jumlah Neuron yaitu 5
weights1 = [[3.6, 2.9, 4.8, 2.6, 4.5, 9.6, 2.7, 5.2, 4.7, 2.5],
           [3.2, 2.9, 2.7, 3.3, 7.5, 7.2, 6.7, 7.8, 6.8, 2.7],
           [5.2, 5.9, 3.9, 1.3, 2.7, 5.4, 2.8, 5.5, 3.2, 2.5],
           [3.7, 1.7, 4.8, 6.8, 5.7, 2.3, 4.7, 2.4, 6.3, 6.4],
           [7.8, 2.3, 2.9, 2.5, 3.7, 7.4, 9.8, 3.3, 6.6, 1.2]]

# Inisialisasi Biases Layer 1 sesuai dengan Neuron yaitu 5
biases1 = [2.5, 8.6, 3.8, 2.8, 5.7]

# Weights 2 sama dengan neuron yang ada di layer 1 yaitu 5
# Jumlah Weights 2 sesuai dengan neuron yang ada di layer 2 yaitu 3 neuron
weights2 = [[0.5, 2.8, 5.3, 6.2, -7.8],
			[5.2, 2.6, 6.7, 6.2, 2.5],
			[3.7, 3.5, 2.6, 2.7, 4.7]]

# Inisialisasi Biases Layer 1 disesuaikan dengan Neuron yaitu 3
biases2 =  [7.3, 7.3, 3.7]

# Output
# Penghitungan Layer 1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# Output
# Penghitungan Layer 2 yaitu dengan hasil perhitungan yang ada di Layer 1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs) 