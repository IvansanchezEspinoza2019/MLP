import numpy as np
from ActivationFunctions import * # import activation functions

### MLP class definition
class MLP:
    def __init__(self, layers_dim, output_activation=logistic, hidden_activation=tanh):
        
        #### Atributes declaration ####
        # Number of layers in the network
        self.L = len(layers_dim) - 1
        # Synaptic weights
        self.w = [None] * (self.L + 1)
        # Bias
        self.b = [None] * (self.L + 1)
        # Activation functions
        self.f = [None] * (self.L + 1)
        
        #### Initialize weigts, bias and activation functions ####
        for l in range(1, self.L + 1):
            self.w[l] = -1 + 2 * np.random.rand(layers_dim[l], layers_dim[l-1])
            self.b[l] = -1 + 2 * np.random.rand(layers_dim[l], 1)
            
            if l == self.L:  # if is the last layer
                self.f[l] = output_activation
            else:           # is is a hidden layer
                self.f[l] = hidden_activation
        
        
        
        