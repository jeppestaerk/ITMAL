import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import fetch_mldata
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



