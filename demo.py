
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlp import *


def draw2DNetData(x, y, model, title, pointSize=5): 
    """
        Plot the decision surface for 2-class problems
    """
    
    # figure
    plt.figure()
    plt.title(title)
    
    ## plot every single pattern of every class in the graph 
    for i in range(x.shape[1]):
        if y[0, i] == 1:
            plt.plot(x[0, i], x[1, i], '.b', markersize=pointSize)   # class 0
        else:
            plt.plot(x[0, i], x[1, i], '.r', markersize=pointSize)   # class 1
            
    ## get the limits of every dimention (1 and 2 only)
    xmin, ymin = np.min(x[0, :])-0.5, np.min(x[1, :])-0.5   # minimum value of dimension 1 and 2 (X[0], X[1])
    xmax, ymax = np.max(x[0, :])+0.5, np.max(x[1, :])+0.5   # maximum value of every dimension (X[0], X[1])
    
    ## makes a mesh with coordinate pairs between xx(x[0, i]) and yy(x[1, i])
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), # makes 100 * 100 points
                         np.linspace(ymin, ymax, 100))
    
    ## create 2-dimentions matrix to make predictions
    data = [xx.ravel(), yy.ravel()]
   
    ## predict the data with 2*10000 shape and get the predictions of 1 * 100000 shape
    zz = model.predict(data)
    
    ## to be 100 * 100 shape
    zz = zz.reshape(xx.shape)
    
    ## contourf graph
    plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdBu)
    
    ## axis limits
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.show()    




problem = 2

if problem == 1: # XOR
    
    # read csv
    data = pd.read_csv("datasets/XOR.csv")
    # entries
    X = np.asanyarray(data.iloc[:, :2]).T               # transposed to be (2, 4)
    # expected values for every pattern in x
    Y = np.asanyarray(data.iloc[:, 2]).reshape(1, -1)   # redimentioned to be (1, 4)

    # because is a classification problem we should choose logistic activation for the output layer
    # and tanh or reLU activations for hidden layers
    net = MLP((2,5,1), hidden_activation=tanh, output_activation=logistic)

    # draw graph before training
    draw2DNetData(X, Y, net, "XOR Before Training")
    
    # train the neural network
    net.fit(X,Y, epochs=500, lr=0.1)
    
    # draw graph after training
    draw2DNetData(X, Y, net,"XOR After Training")
        
elif problem == 2: # CIRCLES
    
    # read csv
    data = pd.read_csv("datasets/circles.csv")
    
    # entries
    X = np.asanyarray(data.drop(columns=["y"])).T
    # output class
    Y = np.asanyarray(data[['y']]).T
    
    # create our network
    net = MLP((2,5,5,1), output_activation=logistic, hidden_activation=tanh)
    
    # before training
    draw2DNetData(X, Y, net, "Model before training",pointSize=7)
    
    # train the network
    net.fit(X, Y, epochs=250, lr=0.1)
    
    # after training
    draw2DNetData(X, Y, net, "Model after training", pointSize=7)
    
    
    
    
    
    
    
    