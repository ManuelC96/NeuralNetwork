from activation_layer import Activation
import numpy as np
#collection of various activation functions 
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)
        super().__init__(tanh, tanh_prime)

