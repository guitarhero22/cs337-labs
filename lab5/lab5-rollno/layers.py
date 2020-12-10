'''This file contains the implementations of the layers required by your neural network

For each layer you need to implement the forward and backward pass. You can add helper functions if you need, or have extra variables in the init function

Each layer is of the form - 
class Layer():
    def __init__(args):
        *Initializes stuff*

    def forward(self,X):
        # X is of shape n x (size), where (size) depends on layer
        
        # Do some computations
        # Store activations_current
        return X

    def backward(self, lr, activation_prev, delta):
        """
        # lr - learning rate
        # delta - del_error / del_activations_current
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        """
        # Compute gradients wrt trainable parameters
        # Update parameters
        # Compute gradient wrt input to this layer
        # Return del_error/del_activation_prev
'''
import numpy as np

class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation   # string having values 'relu' or 'softmax', activation function to use
        # Stores the outgoing summation of weights * feautres 
        self.data = None

        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))    
        self.biases = np.random.normal(0,0.1, (1, out_nodes))
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary 

    def forwardpass(self, X):
        '''
                
        Arguments:
            X  -- activation matrix       :[n X self.in_nodes]
        Return:
            activation matrix      :[n X self.out_nodes]
        '''
        # TODO
        self.data = X @ self.weights + self.biases
        if self.activation == 'relu':
            self.data = relu_of_X(self.data)
        elif self.activation == 'softmax':
            self.data = softmax_of_X(self.data)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        return self.data

    def backwardpass(self, lr, activation_prev, delta):
        '''
        # lr - learning rate
        # delta - del_error / del_activations_current  : 
        # activation_prev - input activations to this layer, i.e. activations of previous layer
        '''

        # TODO 
        if self.activation == 'relu':
            del_sum = gradient_relu_of_X(self.data, delta)
        elif self.activation == 'softmax':
            del_sum = gradient_softmax_of_X(self.data, delta)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        del_prev = del_sum @ self.weights.T
        self.weights = self.weights - lr * (activation_prev.T @ del_sum) / activation_prev.shape[0]
        self.biases = self.biases - lr * np.sum(del_sum, axis = 0) / activation_prev.shape[0]
        return del_prev
        # END TODO

class ConvolutionLayer:
    def __init__(self, in_channels, filter_size, numfilters, stride, activation):
        # Method to initialize a Convolution Layer
        # Parameters
        # in_channels - list of 3 elements denoting size of input for convolution layer
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer
        # numfilters  - number of feature maps (denoting output depth)
        # stride      - stride to used during convolution forward pass
        # activation  - can be relu or None
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride
        self.activation = activation
        self.out_depth = numfilters
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)
        as_strided = np.lib.stride_tricks.as_strided
        self.im2col = lambda x : np.reshape(as_strided(x, 
                                                      (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row, self.filter_col),
                                                      x.strides[:2] + (self.stride * x.strides[2], self.stride * x.strides[3], x.strides[2], x.strides[3])),
                                            (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row * self.filter_col))
        self.im2win = lambda x : as_strided(x, 
                                            (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row, self.filter_col),
                                            x.strides[:2] + (self.stride * x.strides[2], self.stride * x.strides[3], x.strides[2], x.strides[3]))
                                  
        # Stores the outgoing summation of weights * feautres 
        self.data = None
        
        # Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
        self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))   
        self.biases = np.random.normal(0,0.1,self.out_depth)
        

    def forwardpass(self, X):
        # INPUT activation matrix       :[n X self.in_depth X self.in_row X self.in_col]
        # OUTPUT activation matrix      :[n X self.out_depth X self.out_row X self.out_col]

        # TODO
        self.data = np.zeros((X.shape[0], self.out_depth, self.out_row, self.out_col))
        strided_view = self.im2col(X)

        for i in range(self.in_depth):
            for j in range(self.out_depth):
                w = self.weights[j, i].flatten()
                self.data[:, j] += strided_view[:, i] @ w
        for j in range(self.out_depth):
            self.data[j] += self.biases[j]
        
        if self.activation == 'relu':
            self.data = relu_of_X(self.data)
            # raise jotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()

        return self.data
        ###############################################
        # END TODO

    def backwardpass(self, lr, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # Update self.weights and self.biases for this layer by backpropagation
        # TODO

        ###############################################

        if self.activation == 'relu':
            # inp_delta = actual_gradient_relu_of_X(self.data, delta)
            temp = self.im2win(activation_prev)
            ret = np.zeros(activation_prev.shape)
            ret_win = self.im2win(ret)
            for i in range(self.in_depth):
                for j in range(self.out_depth):
                    delt = np.expand_dims(delta[:, j], axis = (-2, -1)) 
                    ret_win[:, i] +=  self.weights[j, i] * delt
                    self.weights[j, i] = self.weights[j,i]  - lr * np.sum(temp[:, i] * delt, axis = (0, -3, -4)) / activation_prev.shape[0]
            self.biases = self.biases - lr * np.sum(delta, axis = (0, -2, -1)) / activation_prev.shape[0]
            # raise NotImplementedError
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

        return np.sum(ret, axis = 0) / activation_prev.shape[0]
        # END TODO
    
class AvgPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)
        
        as_strided = np.lib.stride_tricks.as_strided
        self.im2col = lambda x : np.reshape(as_strided(x, 
                                                      (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row, self.filter_col),
                                                      x.strides[:2] + (self.stride * x.strides[2], self.stride * x.strides[3], x.strides[2], x.strides[3])),
                                            (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row * self.filter_col))
        self.im2win = lambda x : as_strided(x, 
                                            (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row, self.filter_col),
                                            x.strides[:2] + (self.stride * x.strides[2], self.stride * x.strides[3], x.strides[2], x.strides[3]))
         

    def forwardpass(self, X):
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer
        
        # TODO
        return np.mean(self.im2col(X), axis = -1)
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        
        # TODO
        pass
        # END TODO
        ###############################################



class MaxPoolingLayer:
    def __init__(self, in_channels, filter_size, stride):
        # Method to initialize a Convolution Layer
        # Parameters
        # filter_size - list of 2 elements denoting size of kernel weights for convolution layer

        # NOTE: Here we assume filter_size = stride
        # And we will ensure self.filter_size[0] = self.filter_size[1]
        self.in_depth, self.in_row, self.in_col = in_channels
        self.filter_row, self.filter_col = filter_size
        self.stride = stride

        self.out_depth = self.in_depth
        self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
        self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

        as_strided = np.lib.stride_tricks.as_strided
        self.im2col = lambda x : np.reshape(as_strided(x, 
                                                      (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row, self.filter_col),
                                                      x.strides[:2] + (self.stride * x.strides[2], self.stride * x.strides[3], x.strides[2], x.strides[3])),
                                            (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row * self.filter_col))
        self.im2win = lambda x : as_strided(x, 
                                            (x.shape[0], self.in_depth, self.out_row, self.out_col, self.filter_row, self.filter_col),
                                            x.strides[:2] + (self.stride * x.strides[2], self.stride * x.strides[3], x.strides[2], x.strides[3]))
          
    def forwardpass(self, X):   
        # print('Forward MP ')
        # Input
        # X : Activations from previous layer/input
        # Output
        # activations : Activations after one forward pass through this layer

        # TODO
        self.data = np.max(self.im2col(X), axis = -1)
        return self.data
        # END TODO
        ###############################################
        
    def backwardpass(self, alpha, activation_prev, delta):
        # Input
        # lr : learning rate of the neural network
        # activation_prev : Activations from previous layer
        # activations_curr : Activations of current layer
        # delta : del_Error/ del_activation_curr
        # Output
        # new_delta : del_Error/ del_activation_prev
        # TODO
        ret = np.zeros(activation_prev.shape)
        ret2 = self.im2win(ret)
        activation_prev2 = self.im2win(activation_prev)
        ret2[activation_prev2 == np.expans_dims(self.data, axis = (-2, -1))] = 1
        ret2 = ret2 * np.expand_dims(delta, axis = axis = (-1, -2))
        return ret
        # END TODO
        ###############################################

# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        # TODO
        # print(X.shape)
        pass
    def backwardpass(self, lr, activation_prev, delta):
        pass
        # END TODO

# Function for the activation and its derivative
def relu_of_X(X):

    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu
    # TODO
    X[X < 0] = 0
    return X
    # raise NotImplementedError
    # END TODO 
    
def gradient_relu_of_X(X, delta):
    # Input
    # Note that these shapes are specified for FullyConnectedLayers, the function also needs to work with ConvolutionalLayer
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation relu amd during backwardpass
    
    # TODO
    X[X > 0] = 1
    return X * delta
    # raise NotImplementedError
    # END TODO

def softmax_of_X(X):
    # Input
    # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
    # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax
    
    # TODO
    temp = np.exp(X)
    return temp / (np.sum(temp, 1)[:, np.newaxis])
    # raise NotImplementedError
    # END TODO 

def gradient_softmax_of_X(X, delta):
    # Input
    # data : Output from next layer/input | shape: batchSize x self.out_nodes
    # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
    # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
    # This will only be called for layers with activation softmax amd during backwardpass
    # Hint: You might need to compute Jacobian first

    # TODO
    temp = X * delta
    return temp - X * np.expand_dims(np.sum(temp, 1), axis = -1)
    # raise NotImplementedError
    # END TODO
