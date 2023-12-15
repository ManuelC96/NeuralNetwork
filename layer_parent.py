# parent layer class
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    # compute Y for a given X trough a linear func, f(X) = (WX + B)
    def forwardProp(self, input = None):
        # TODO return output
        raise NotImplementedError
    # compute dE/dX(outpur_error) for a given dE/dY (input_error) and update parameters (W - B) at a specified learning rate
    def backwardProp(self, output_error = str, lrn_rate = None):
        # TODO update parameters and return input_gradient
        raise NotImplementedError
        