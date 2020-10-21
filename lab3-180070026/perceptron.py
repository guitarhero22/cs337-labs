import numpy as np
import argparse
import matplotlib.pyplot as plt

def get_data(dataset):
	datasets = ['D1', 'D2']
	assert dataset in datasets, "Dataset {dataset} not supported. Supported datasets {datasets}"
	X_train = np.loadtxt(f'data/{dataset}/training_data')
	Y_train = np.loadtxt(f'data/{dataset}/training_labels', dtype=int)
	X_test = np.loadtxt(f'data/{dataset}/test_data')
	Y_test = np.loadtxt(f'data/{dataset}/test_labels', dtype=int)

	return X_train, Y_train, X_test, Y_test

def get_features(x):
	'''
	Input:
	x - numpy array of shape (2500, )
	Output:
	features - numpy array of shape (D, ) with D <= 5
	'''
	### TODO
	f = []
	x = x.reshape((50, 50))
	k = np.zeros((50,50)) 
	for i in [-1,1]: 
		for j in [0,1]:   
			k[(x == 1) & (np.roll(x, i, j) == 0)] = 1
	A = np.count_nonzero(x) / 2500
	P = np.count_nonzero(k) / 50
	f.append(A)
	f.append(P)
	f.append(A / (np.pi *(P ** 2))) 
	# f.append(1 - A)
	### END TODO
	return np.array(f)

class Perceptron():
    def __init__(self, C, D):
        '''
        C - number of classY_testes
        D - number of features
        '''
        self.C = C
        self.weights = np.zeros((C, D))
        
    def pred(self, x):
        '''
        x - numpy array of shape (D,)
        '''
        ### TODO: Return predicted class for x
        scores = np.dot(self.weights, x)
        return np.argmax(scores)
        ### END TODO

    def train(self, X, Y, max_iter=100):
        for _ in range(max_iter):
            for i in range(X.shape[0]):
                ### TODO: Update weights
                p = self.pred(X[i])
                y = Y[i]
                if(y == p): continue
                self.weights[p] = self.weights[p] - X[i]
                self.weights[y] = self.weights[y] + X[i]
                ### END TODO

    def eval(self, X, Y):
        n_samples = X.shape[0]
        correct = 0
        for i in range(X.shape[0]):
            if self.pred(X[i]) == Y[i]:
                correct += 1
        return correct/n_samples

if __name__ == '__main__':
	X_train, Y_train, X_test, Y_test = get_data('D2')

	X_train = np.array([get_features(x) for x in X_train])
	X_test = np.array([get_features(x) for x in X_test])

	C = max(np.max(Y_train), np.max(Y_test))+1
	D = X_train.shape[1]

	perceptron = Perceptron(C, D)

	perceptron.train(X_train, Y_train)
	acc = perceptron.eval(X_test, Y_test)
	print(f'Test Accuracy: {acc}')