import numpy as np
import matplotlib.pyplot as plt
from utils import load_data2, split_data, preprocess, normalize

np.random.seed(337)


def mse(X, Y, W):
    """
    Compute mean squared error between predictions and true y values

    Args:
    X - numpy array of shape (n_samples, n_features)
    Y - numpy array of shape (n_samples, 1)
    W - numpy array of shape (n_features, 1)
    """
    # TODO
    mse = np.mean( ( Y - (X @ W) ) ** 2 ) / 2
    # END TODO
    return mse

def ista(X_train, Y_train, X_test, Y_test, _lambda=0.1, lr=0.01, max_iter=10000):
    """
    Iterative Soft-thresholding Algorithm for LASSO
    """
    samples, features = X_train.shape
    train_mses = []
    test_mses = []

    # TODO: Initialize W using using random normal
    W_prev = np.random.randn(features, 1)
    # END TODO

    for _ in range(max_iter):
        # TODO: Compute train and test MSE
        train_mse = mse(X_train, Y_train, W_prev)
        test_mse = mse(X_test, Y_test, W_prev)
        # END TODO

        train_mses.append(train_mse)
        test_mses.append(test_mse)
        # TODO: Update w and b using a single step of ISTA. You are not allowed to use loops here.
        W_new = W_prev - lr * ( X_train.T @ ( (X_train @ W_prev) - Y_train ) ) / samples
        c1 = (W_new >=  - lr * _lambda) & (W_new <= lr * _lambda) 
        W_new[c1] = 0
        c1 = W_new > lr * _lambda
        W_new[c1] = W_new[c1] - lr * _lambda 
        c1 = W_new < - lr * _lambda
        W_new[c1] = W_new[c1] + lr * _lambda
        # END TODO
        # TODO: Stop the algorithm if the norm between previous W and current W falls below 1e-4
        if np.linalg.norm( W_new - W_prev ) < 0.0001 : break
        W_prev= W_new
        # End TODO
    # print(np.linalg.norm( W_new - W_prev ))
    W = W_new
    # print(len(train_mses))
    return W, train_mses, test_mses

def grad1(X, Y, W, reg):
	return np.dot((np.dot(X.T, X) / X.shape[0]) + 2 * reg * np.eye(W.shape[0]), W) - (np.dot(X.T, Y.flatten()) / X.shape[0])

def ridge_regression(X_train, Y_train, X_test, Y_test, reg, lr=0.025, max_iter=2000):
	'''
	reg - regularization parameter (lambda in Q2.1 c)
	'''
	train_mses = []
	test_mses = []
	## TODO: Initialize W using using random normal 
	W = np.random.randn(X_train.shape[1],)
	## END TODO

	for _ in range(max_iter):
		## TODO: Compute train and test MSE
		train_mse = mse(X_train, Y_train, W)
		test_mse = mse(X_test, Y_test, W)
		## END TODO
		train_mses.append(train_mse)
		test_mses.append(test_mse)
		## TODO: Update w and b using a single step of gradient descent
		W_new = W - lr * grad1(X_train, Y_train, W, reg)
		if(mse(X_train, Y_train, W_new) >= train_mse): break
		W = W_new
		## END TODO

	return W[:, np.newaxis], train_mses, test_mses

if __name__ == '__main__':
    # Load and split data
    X, Y = load_data2('data2.csv')
    X, Y = preprocess(X, Y)
    X_train, Y_train, X_test, Y_test = split_data(X, Y)


    # TODO: Your code for plots required in Problem 1.2(b) and 1.2(c)
    lam = 0.1
    lams = []
    test_ms = []
    train_ms = []
    for i in range(30):
        W, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test, _lambda= lam, lr = 0.001, max_iter = 10000)
        lams.append(lam)
        test_ms.append(test_mses_ista[-1])
        train_ms.append(train_mses_ista[-1])
        print(lam, test_mses_ista[-1], train_mses_ista[-1])
        lam += (i+1)*0.02
    plt.plot(lams, test_ms)
    plt.plot(lams, train_ms)
    plt.show()
    
    # lam = lams[np.argmin(test_ms)]
    lam = 0.16
    print("lambda for min test_mse_ista: %.6f" %lam )
    plt.figure()
    W_ista, train_mses_ista, test_mses_ista = ista(X_train, Y_train, X_test, Y_test, _lambda = lam, max_iter = 10000)
    W_ridge, train_mses_ridge, test_mses_ridge = ridge_regression(X_train, Y_train, X_test, Y_test, 10, 0.008, 10000)
    plt.scatter(range(W_ista.shape[0]), W_ista.flatten(), c = 'orange', alpha = 0.3)
    plt.show()
    plt.figure()
    plt.scatter(range(W_ridge.shape[0]), W_ridge.flatten(),c = 'orange' ,alpha = 0.3)
    plt.show()
    # plt.legend(labels = ['ista','ridge'])
    print("ista mse: ", test_mses_ista[-1], "ridge mse: ", test_mses_ridge[-1])
    plt.show()
    # End TODO