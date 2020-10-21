import numpy as np
import argparse
from utils import *


def get_data(dataset):
    datasets = ['D1', 'D2']
    assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
    X_train = np.loadtxt(f'data/{dataset}/training_data')
    Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
    X_test = np.loadtxt(f'data/{dataset}/test_data')
    Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

    return X_train, Y_train, X_test, Y_test


def one_hot_encode(X, labels):
    '''
    Args:
    X - numpy array of shape (n_samples, 1)
    labels - list of all possible labels for current category

    Returns:
    X in one hot encoded format (numpy array of shape (n_samples, n_labels))
    '''
    X.shape = (X.shape[0], 1)
    newX = np.zeros((X.shape[0], len(labels)))
    label_encoding = {}
    for i, l in enumerate(labels):
        label_encoding[l] = i
    for i in range(X.shape[0]):
        newX[i, label_encoding[X[i, 0]]] = 1
    return newX


class LogisticRegression:
    def __init__(self, C, D):
        """
        C - number of classes
        D - number of features
        """
        self.C = C
        self.D = D
        self.weights = np.random.rand(D, C)

    def softmax(self, X):
        """
        :param X: (N, D) array
        :return: softmax for given X and current weights
        """
        # TODO: Compute Softmax
        Z = np.exp(-np.dot(X, self.weights))
        return Z / np.sum(Z, 1)[:, np.newaxis]
        # END TODO

    def predict(self, X):
        """
        :param X: numpy array of shape (N, D) corresponding to inpus data
        :return: numpy array of shape (N, 1) corresponding to predicted labels
        """
        # TODO: Return a (N, 1) numpy array of predictions.
        return np.argmax(self.softmax(X), 1)
        # END TODO

    def train(self, X, Y, lr=0.1, max_iter=500):
        n = X.shape[0]
        Y = one_hot_encode(Y, np.unique(Y))
        # print(np.linalg.norm(X))
        for i in range(max_iter):
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.
            grad = np.dot(X.T , (self.softmax(X) - Y)) / n
            # END TODO
            self.weights = self.weights + lr * grad  
            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4
            if np.linalg.norm(self.weights) < 0.0001: break
            # End TODO
            # if(i % 100 == 0) :  print(i, np.linalg.norm(grad), np.linalg.norm(self.softmax(X)) , np.linalg.norm(self.weights))
        return

    def eval(self, X, Y):
        """
        X - numpy array of shape (N, D) corresponding to inpus data
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        preds = self.predict(X)
        accuracy = ((preds == Y).sum()) / len(preds)
        return accuracy

if __name__ == '__main__':
    np.random.seed(335)

    # print('preparing data...')
    X_train, Y_train, X_test, Y_test = get_data("D1")

    C = max(np.max(Y_train), np.max(Y_test)) + 1
    D = X_train.shape[1]

    lr = LogisticRegression(C, D)
    
    # print('starting training..')
    lr.train(X_train, Y_train)

    # print('evaluating...')
    acc = lr.eval(X_test, Y_test)

    print(f'Test Accuracy: {acc}')