from neuralnetwork import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(17)
# Glorot Initialization
nn_glorot = NeuralNetwork(hidden_dims=(512, 256), activation_type='sigmoid', initialization_type='Glorot',
                          batch_size=64, epochs=10, step=0.03)
nn_glorot.train()
loss_glorot = nn_glorot.cache['loss']

# Normal Initialization
nn_normal = NeuralNetwork(hidden_dims=(512, 256), activation_type='sigmoid', initialization_type='Normal',
                          batch_size=64, epochs=10, step=0.03)
nn_normal.train()
loss_normal = nn_normal.cache['loss']

# zero Initialization
nn_zero = NeuralNetwork(hidden_dims=(512, 256), activation_type='sigmoid', initialization_type='Zero',
                        batch_size=64, epochs=10, step=0.03)
nn_zero.train()
loss_zero = nn_zero.cache['loss']

assert(len(loss_glorot) == len(loss_normal))
assert(len(loss_zero) == len(loss_normal))
print('Glorot Initialization: ', loss_glorot)
print('Normal Initialization: ', loss_normal)
print('Zero Initialization: ', loss_zero)
plt.plot(range(10), loss_glorot, 'b', loss_normal, 'r', loss_zero, 'k-')
plt.legend(['glorot Initialization', 'Normal Initialization', 'Zero Initialization'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()