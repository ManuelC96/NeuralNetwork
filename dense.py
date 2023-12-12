from layer import Layer
import numpy as np
# implement forward and backward prop, [ - activation func not included - ]

class Dense(Layer):
    def __init__(self, matrix_n_size = None, matrix_p_size = None):
        self.weights = np.random.rand(matrix_n_size, matrix_p_size) 
        self.biases = np.random.rand(matrix_n_size, 1)
    
    def forwardProp(self, input_gradient = None):
        self.input_gradient = input_gradient
        output_gradient = np.dot( input_gradient,self.weights ) + self.biases
        return output_gradient

    def backwardProp(self, output_gradient = None, alpha = None):
        self.weights = self.weights - np.dot(output_gradient, self.input_gradient.T) * alpha
        self.biases = self.biases - output_gradient * alpha
        input_gradient = np.dot( self.weights.T, output_gradient )
        return input_gradient