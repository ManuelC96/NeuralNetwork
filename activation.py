from layer import Layer
import numpy as np
# implement forward and backward for activation func

class Activation(Layer):
    def __init__(self, activation_func = None, activation_func_prime = None):
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime
        
    def forwardProp(self, input = None):
        self.input = input
        output_gradient = self.activation_func(self.input)
        return output_gradient
        
    def backwardProp(self, output_gradient=None, alpha=None):
        input_gradient = np.multiply(output_gradient, self.activation_func_prime(self.input))
        return input_gradient












