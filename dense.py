from layer import Layer
import numpy as np
# implement forward and backward prop, [ - activation func not included - ]

# inherit from base class Layer
class Dense(Layer):
    # matrix_n_size = number of input neurons
    # matrix_p_size = number of output neurons
    def __init__(self, matrix_n_size = None, matrix_p_size = None):
        self.weights = np.random.rand(matrix_n_size, matrix_p_size) 
        self.biases = np.random.rand(matrix_n_size, 1)

    # returns output for a given input
    def forwardProp(self, input_gradient = None):
        self.input_gradient = input_gradient
        output_gradient = np.dot( input_gradient,self.weights ) + self.biases
        return output_gradient
    
    # computes dE/dW, dE/dB for a given output_gradient=dE/dY. Returns input_gradient=dE/dX.
def backwardProp(self, output_gradient = None, alpha = None):
        self.weights = self.weights - np.dot(output_gradient, self.input_gradient.T) * alpha
        self.biases = self.biases - output_gradient * alpha
        input_gradient = np.dot(self.weights.T, output_gradient)
        return input_gradient