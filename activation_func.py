from activation import Activation
import numpy as np
# implement different activation functions 

class Tanh(Activation):
    def __init__(self, X = None):
        self.tanh = lambda x: np.tanh(x)
        self.tanh_prime = lambda x: 1 - np.tanh_prime(x)

# TODO implement other activation functions





