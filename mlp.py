import numpy as np
from ActivationFunctions import * # import activation functions

### MLP class definition
class MLP:
    def __init__(self, layers_dim, 
                 output_activation=logistic, 
                 hidden_activation=tanh):
        
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
            else:           #if is a hidden layer
                self.f[l] = hidden_activation
        
    def predict(self, X):
        ##### Prediction (propagation)
        a = np.asanyarray(X)
        for l in range(1, self.L + 1):
            z = np.dot(self.w[l], a) + self.b[l]
            a = self.f[l](z)
        return a
        
    def fit(self, X, Y, epochs=500, lr=0.1):
        
        """
            Training algorithm used: Stochastic Gradient Descent (SGD)
        """
        P = X.shape[1]          # Number of patterns 
        for _ in range(epochs): # for every epoch
            for p in range(P):  # for every pattern
                
                ### Initialize
                lg = [None] * (self.L + 1)          # local gradient (lg)
                a = [None] * (self.L + 1)           # output of every layer (a)
                da = [None] * (self.L + 1)          # derivative of a (da)
                
                ### Propagation
                a[0] = X[:, p].reshape(-1, 1)       # first entry 
                for l in range(1, self.L + 1):      # for every layer
                    z = np.dot(self.w[l], a[l-1]) + self.b[l]
                    a[l], da[l] = self.f[l](z, derivative=True)
                         
                ### Backpropagation
                for l in range(self.L, 0, -1):
                    if l == self.L:
                        lg[l] = (Y[:, p].reshape(-1, 1) - a[l]) * da[l]
                    else:
                        lg[l] = np.dot(self.w[l+1].T, lg[l+1]) * da[l]
                
                ### Gradient descent
                for l in range(1, self.L + 1):
                    self.w += lr * np.dot(lg[l], a[l-1].T)
                    self.b += lr * lg[l]                    
        
        