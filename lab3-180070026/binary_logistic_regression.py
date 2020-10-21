import numpy as np
import argparse
from utils import *
np.random.seed(0)

class BinaryLogisticRegression:
    def __init__(self, D):
        """
        D - number of features
        """
        self.D = D
        self.weights = np.random.rand(D, 1)

    def predict(self, X):
        """
        X - numpy array of shape (N, D)
        """
        # TODO: Return a (N, 1) numpy array of predictions.
        exp = 1 / ( np.exp(-np.dot(X, self.weights)) + 1)
        # print(X.shape)
        pred = np.zeros((X.shape[0],1))
        pred[exp >= 0.5] = 1
        pred[exp < 0.5] = 0
        # print(pred)
        # END TODO
        return pred

    def train(self, X, Y, lr=0.5, max_iter=1000):
        # print('training started')
        n = X.shape[0]
        for _ in range(max_iter):
            # if(i % 100 == 0): print(i)
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.
            prob = 1  / (np.exp(-np.dot(X, self.weights)) + 1)
            # print(prob.shape)
            # print(Y.shape)
            # print(np.sum(X.T, 1).shape)
            grad = np.sum(X * (prob - Y), 0)[:, np.newaxis] / n
            self.weights = self.weights - lr * grad
            # END TODO
            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4
            if(np.linalg.norm(grad) < 0.0001 ): break
            # End TODO
        
        return

    def accuracy(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        accuracy = (np.count_nonzero(preds == Y)) / len(preds)
        return accuracy

    def f1_score(self, preds, Y):
        """
        preds - numpy array of shape (N, 1) corresponding to predicted labels
        Y - numpy array of shape (N, 1) corresponding to true labels
        """
        # TODO: calculate F1 score for predictions preds and true labels Y
        tp = np.count_nonzero( (preds == 1) & (Y == 1) )
        fn = np.count_nonzero( (preds == 0) & (Y == 1) )
        fp = np.count_nonzero( (preds == 1) & (Y == 0) )
        rec = tp / (tp + fn)
        pre = tp / (tp + fp)
        # End TODO
        return 2 * rec * pre / ( rec + pre )


if __name__ == '__main__':
    np.random.seed(335)

    X, Y = load_data('data/songs.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)

    D = X_train.shape[1]

    lr = BinaryLogisticRegression(D)
    lr.train(X_train, Y_train)
    preds = lr.predict(X_test)
    acc = lr.accuracy(preds, Y_test)
    f1 = lr.f1_score(preds, Y_test)
    print(f'Test Accuracy: {acc}')
    print(f'Test F1 Score: {f1}')
