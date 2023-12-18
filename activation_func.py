from activation_layer import Activation
import numpy as np
# implement different activation functions 
class Tanh(Activation):
    def __init__(self, X = None):
        self.tanh = lambda x: np.tanh(x)
        self.tanh_prime = lambda x: 1 - np.tanh_prime(x)
        super().__init__(self.tanh, self.tanh_prime)

# TODO implement other activation functions





