
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlp import *


def draw2DNetData(x, y, model):
    plt.figure()
    ## plot every single pattern in the graph
    for i in range(x.shape[1]):
        if y[0, i] == 1:
            plt.plot(x[0, i], x[1, i], '.r', markersize=15)   # class 0
        else:
            plt.plot(x[0, i], x[1, i], '.b', markersize=15)   # class 1
            
    # get the limits of the entries
    xmin, ymin = np.min(x[0, :])-0.5, np.min(x[1, :])-0.5   # minimum value of dimension 1 and 2 (X[0], X[1])
    xmax, ymax = np.max(x[0, :])+0.5, np.max(x[1, :])+0.5   # maximum value of every dimension (X[0], X[1])
    
    # makes a mesh with coordinate pairs between xx(x[0, i]) and yy(x[1, i])
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), # makes 100 * 100 pints
                         np.linspace(ymin, ymax, 100))
    
    # create 2-dimentions matrix
    data = [xx.ravel(), yy.ravel()]
   
    # predict the data 2*10000 shape and return 1*100 shape
    zz = model.predict(data)
    
    # to be 100 * 100 shape
    zz = zz.reshape(xx.shape)
    
    # contourf
    plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdBu)
    
    # axis limits
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.show()    

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
    net = MLP((2,5,1), hidden_activation=tanh, output_activation=logistic)

    # train the neural network
    net.fit(X,Y, epochs=500, lr=0.1)
    
    # make predictions over X
    ypred = net.predict(X)
    
    draw2DNetData(X, Y, net)
        
