# solve xor gate logic to test code functioning
from dense_layer import Dense
from activation_func import Tanh
import losses as ls
# imports
import numpy as np

# create X matrix and Y label matrix
X = np.reshape([[0, 1],[0, 0],[1, 0],[1, 1]],(4, 2, 1))
Y = np.reshape([[1], [0], [1], [0]],(4, 1, 1))
zp = zip(X, Y)
print(f"-- X matrix -- \n {X} ", end="\n\n")
print(f"-- Y matrix -- \n {Y} ", end="\n\n")
for i, j in zp:
    print(f"-- Zip matrix --\n {i}{j}")

# build network
network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# define epochs and learning rate
epochs = 10000
learning_rate = 0.001

# train the model
# start the loop and set error to 0
for e in range(epochs):
    error = 0
    # zip X matrix and Y labels, than loop trough
    for x, y in zip(X, Y):
        output = x
        # forward prop
        for j in network:
            output = j.forwardProp(output.T)
        #calculate erro
        error += ls.mse(Y, output) 

        # backward prop
        grad = ls.mse_prime(Y, output)
        for k in reversed(network):
            grad = k.backwardProp(grad, learning_rate)
        
    print(f"error {error}")