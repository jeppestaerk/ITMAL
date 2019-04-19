import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_iris
from keras.datasets import mnist


def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    # add plot functionality for a single digit...
    plt.imshow(image, cmap='gray')
    plt.show()


def MNIST_GetDataSet():
    # use mnist = fetch_mldata('MNIST original') or mnist.load_data(),
    # but return as a single X-y tuple
    mnist_data = fetch_mldata('MNIST original')
    X = mnist_data["data"]
    y = mnist_data["target"]
    return X, y


def MOON_GetDataSet(n_samples):
    moons = make_moons(n_samples, noise=0.1)
    # print(moons)
    return moons
    

def MOON_Plot(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.set(title='Moon dataset')
    plt.show()


def MOON_GetDataSet(n_samples):
    moons = make_moons(n_samples, noise=0.1)
    # print(moons)
    return moons
    

def MOON_Plot(X, y, title="Moon dataset", xlabel="", ylabel=""):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    fig = plt.figure(figsize=(6, 6))
    plot_train = plt.subplot(2,1,1)
    plot_test = plt.subplot(2,1,2)
    
    plot_train.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plot_test.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    
    plot_train.set(xlabel=xlabel, ylabel=ylabel, title=title + " Train")
    plot_test.set(xlabel=xlabel, ylabel=ylabel, title=title + " Test")
    
    plt.tight_layout()
    plt.show()


def IRIS_GetDataSet():
    iris_data = load_iris()
    X = iris_data["data"]
    y = iris_data["target"]
    target_names = iris_data["target_names"]
    feature_names = iris_data["feature_names"]
    return X, y, target_names, feature_names


def IRIS_PlotFeatures(X, y, targets, features):
    colors = [None] * len(X)
    for i in range(len(X)):
        if y[i] == 0:
            colors[i] = 'r'
        elif y[i] == 1:
            colors[i] = 'g'
        else:
            colors[i] = 'b'
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10),
                             frameon=True)
    title="Iris Data (Red=setosa,green=versicolor,blue=virginica)"
    fig.suptitle(title, fontsize=16, y=0.92)
    
    for i in range(4):
        for j in range(4):
            ax = axes[i,j]
            if i == j:
                ax.text(0.5, 0.5, features[i], fontsize=12,
                        transform=ax.transAxes,
                        horizontalalignment='center',
                        verticalalignment='center')
                ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_visible(False)
            else:
                ax.scatter(X[:,j], X[:,i], c=colors, s=3)
    
    plt.show()
