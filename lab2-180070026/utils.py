import numpy as np

def load_data1(file):
	'''
	Given a file, this function returns X, the regression features
	and Y, the output

	Args:
	filename - is a csv file with the format

	feature1,feature2, ... featureN,y
	0.12,0.31,1.33, ... ,5.32

	Returns:
	X - numpy array of shape (number of samples, number of features)
	Y - numpy array of shape (number of samples, 1)
	'''

	data = np.loadtxt(file, delimiter=',', skiprows=1)
	X = data[:, :-1]
	Y = data[:, -1:]

	return X, Y

def load_data2(file):
	'''
	Given a file, this function returns X, the features 
	and Y, the output

	Args:
	filename - is a csv file with the format

	feature1,feature2, ... featureN,y
	0.12,0.31,Yes, ... ,5.32

	Returns:
	X - numpy array of shape (number of samples, number of features)
	Y - numpy array of shape (number of samples, 1)
	'''
	data = np.loadtxt(file, delimiter=',', skiprows=1, dtype='str')
	X = data[:, :-1]
	Y = data[:, -1:].astype(float)
	return X, Y


def split_data(X, Y, train_ratio=0.8):
	'''
	Split data into train and test sets
	The first floor(train_ratio*n_sample) samples form the train set
	and the remaining the test set

	Args:
	X - numpy array of shape (n_samples, n_features)
	Y - numpy array of shape (n_samples, 1)
	train_ratio - fraction of samples to be used as training data

	Returns:
	X_train, Y_train, X_test, Y_test
	'''
	## TODO
	n = int(X.shape[0] * train_ratio)
	X_train, Y_train = X[:n], Y[:n]
	X_test, Y_test = X[n:], Y[n:] 
	## END TODO

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
		newX[i, label_encoding[X[i,0]]] = 1
	return newX

def normalize(X):
	'''
	Returns normalized X

	Args:
	X of shape (n_samples, 1)

	Returns:
	(X - mean(X))/std(X)
	'''
	## TODO
	return (X - np.mean(X)) / np.std(X)
	## END TODO

def is_categorical(x):
	try:
		float(x)
		return False
	except ValueError:
		return True

def preprocess(X, Y):
	'''
	X - feature matrix; numpy array of shape (n_samples, n_features) 
	Y - outputs; numpy array of shape (n_samples, 1)

	Convert data X obtained from load_data2 to a usable format by gradient descent function
	Use one_hot_encode() to convert 

	NOTE 1: X has first column denote index of data point. Ignore that column
			and add constant 1 instead (for bias) 
	NOTE 2: For categorical string data, encode using one_hot_encode() and
			normalize the other features and Y using normalize()
	'''
	## TODO
	samples = X.shape[0]
	features = X.shape[1]
	num_cat = [1,]
	cats = [[]]
	for i in range(1, features):
		cat = []
		if is_categorical(X[0, i]):
			cat = np.unique(X[:, i])
			num_cat.append(len(cat))
		else:
			num_cat.append(1)
		cats.append(cat)
	cols = sum(num_cat)
	X_out = np.ones((samples, cols))
	cur = 0
	for i in range(1, features):
		if is_categorical(X[0,i]):
			X_out[:, cur:num_cat[i]+cur] = one_hot_encode(np.expand_dims(X[:,i], 1), cats[i])
		else:
			# print(X[0,i])
			X_out[:, cur:cur + num_cat[i]] = normalize(np.expand_dims(X[:,i], 1).astype(float))
		cur += num_cat[i]
	return X_out, normalize(Y)
	## END TODO

