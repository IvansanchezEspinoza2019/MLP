"""
 Activation functions for hidden and output layers
"""

import numpy as np


################### For OUTPUT LAYER #######################
# For regression problems
def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

# For Multi-Labels Classification problems
def logistic(z, derivative=False): # logistic function
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

# For Multi-Class Classification problems
def softmax(z, derivative=False):
    e = np.exp(z - np.max(z, axis=0))
    a = e / np.sum(e, axis=0)
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a


############## For HIDDEN LAYERS ##################
def tanh(z, derivative=False):
    a = np.tanh(z)
    if derivative:
        da = (1 + a) * (1 - a)
        return a, da
    return a

def relu(z, derivative=False):
    a = z * (z >= 0)
    if derivative:
        da = np.array( z >=0, dtype=np.float64)
        return a,da
    return a

# Logistic for hidden layers function
def logistic_hidden(z, derivative=False): 
    a = 1 / (1 + np.exp(-z))
    if derivative:
        da = a * (1 - a)
        return a, da
    return a