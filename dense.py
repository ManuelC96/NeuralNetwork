from layer import Layer
import numpy as np
# implement forward and backward prop

class Dense(Layer):
    def __init__(self, matrix_n_size = None, matrix_p_size = None):
        self.weights = np.random.rand(matrix_n_size, matrix_p_size) 
        self.biases = np.random.rand(matrix_n_size, 1)
    
    def forwardProp(self, input_gradient):
        self.input_gradient = input_gradient
        return np.dot(self.weights, input_gradient) + self.biases

    def backwardProp(self, output_gradient = None, alpha = None):
        weights_gradient = np.dot(output_gradient, self.input_gradient.T)
        biases_gradients