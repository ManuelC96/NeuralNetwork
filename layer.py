# parent layer class
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    
    def forwardProp(self, input = None):
        # TODO return output
        pass
    
    def backwardProp(self, output_gradient = None, alpha = None):
        # TODO update parameters and return input_gradient
        pass