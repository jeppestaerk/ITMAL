#!/opt/anaconda3/bin/python

#import sys,os
#from itmal import utils

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from keras.datasets import mnist
from sklearn import datasets

def MOON_GetDataSet(n_samples=100):
    X, y=make_moons(n_samples=n_samples, noise=0.1, random_state=0)
    return X, y

def MOON_Plot(X, y):
    figure = plt.figure(figsize=(12, 9))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k');


def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")

def MNIST_GetDataSet():
    mnist = fetch_mldata('MNIST original')
    # NOTE: or (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print("TODO: insert train/test split and shuffle code")
    #  ...
    #return ...    
    
def MNIST_GetDataSet(fetchmode=True):
    if fetchmode:
        d = fetch_mldata('MNIST original')
        X, y= d["data"], d["target"]  
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X, y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
    
    # NOTE: notice that X and y are defined inside if's, not in outer scope as in C++, strange!
    assert X.shape[0]==70000    
    assert X.shape[0]==y.shape[0]
    assert y.ndim==1
    return X, y

def IRIS_GetDataSet():
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data  # we only take the first two features.
    y = iris.target
    return X, y 

def IRIS_PlotFeatures(X, y, i, j):
    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.show()
         
def TrainTestSplit(X, y, N, shuffle=True, verbose=False):     
    assert X.shape[0]>N
    assert y.shape[0]==X.shape[0]
    assert X.ndim>=1
    assert y.ndim==1
    
    X_train, X_test, y_train, y_test = X[:N,:], X[N:,:], y[:N], y[N:] # or X[:N], X[N:], y[:N], y[N:]

    if shuffle:
        shuffle_index = np.random.permutation(N)
        X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    
    if verbose:
        print("X_train.shape=",X_train.shape)
        print("X_test .shape=",X_test.shape)
        print("y_train.shape=",y_train.shape)
        print("y_test .shape=",y_test.shape)
    
    return X_train, X_test, y_train, y_test

######################################################################################################
# 
# TESTS
#
######################################################################################################

def TestAll():
    print("ALL OK")

if __name__ == '__main__':
    TestAll()
