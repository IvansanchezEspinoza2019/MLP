
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlp import *


problem = 1

if problem == 1: # XOR
    
    # read csv
    data = pd.read_csv("datasets/XOR.csv")
    # entries
    X = np.asanyarray(data.iloc[:, :2]).T               # transposed to be (2, 4)
    # expected values for every pattern in x
    Y = np.asanyarray(data.iloc[:, 2]).reshape(1, -1)   # redimentioned to be (1, 4)

    # because is a classification problem we should choose logist activation for the output layer
    # and tanh or reLU activations for hidden layers
    net = MLP((2,3,1), hidden_activation=tanh, output_activation=logistic)

    # train the neural network
    net.fit(X,Y, epochs=500, lr=0.1)
    
    # make predictions over X
    ypred = net.predict(X)
        
