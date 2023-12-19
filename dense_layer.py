from layer_parent import Layer
import numpy as np
# implement forward and backward prop, [ - activation func not included - ]

# inherit from base class Layer
class Dense(Layer):
    # input = number of input neurons X(i)
    # output = number of output neurons Y(j)
    def __init__(self, input = None, output = None):
        self.weights = np.random.rand(output, input) 
        self.biases = np.random.rand(output, 1)
        super().__init__()

    
    def forwardProp(self, input_gradient = None):
        self.input_gradient = input_gradient
        output_gradient = np.dot(self.weights, self.input_gradient ) + self.biases
        return output_gradient
    
    
    def backwardProp(self, output_gradient = None, lrn_rate = None):
        self.weights = self.weights - np.dot(output_gradient, self.input_gradient.T) * lrn_rate
        self.biases = self.biases - output_gradient * lrn_rate
        input_gradient = np.dot(self.weights.T, output_gradient)
        Layer.container.clear()
        Layer.container.append([f"weight: {self.weights}, bias: {self.biases}"])
        return input_gradient
    

