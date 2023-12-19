from layer_parent import Layer
import numpy as np
# implement forward and backward for activation func

class Activation(Layer):
    # activation_func = non linear function for activation
    # activation_func_prime = derivative of activation function
    def __init__(self, activation_func = None, activation_func_prime = None):
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime

    # element wise application of f(x) to input matrix   
    def forwardProp(self, input = None):
        self.input = input
        output_gradient = self.activation_func(self.input)
        return output_gradient

    # computation of f'(x)   
    def backwardProp(self, output_gradient=None, learning_rate = None):
        input_gradient = np.multiply(output_gradient, self.activation_func_prime(self.input))
        return input_gradient












