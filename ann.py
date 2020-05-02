
import numpy as np
import pandas as pd

# sigmoid function
def sigmoid(Z):
    A = 1 / (1+np.exp(np.dot(-1, Z)))
    return A

# ReLU function 
def ReLU(Z):
    A = np.maximum(Z,0)
    return A

# the derivative of sigmoid
def Dsigmoid(Z):
    return sigmoid(Z)*(1-sigmoid(Z)) # note that * is the elementwise product

# the derivative of tanh
def Dtanh(Z):
    return 1-(np.tanh(Z))**2

# the derivative of ReLU
def DReLU(Z):
    return np.greater(Z, 0).astype(int)

class NeuralNetwork:
    
    # x is the input data matrix (n, m), n attributes, m samples
    # y is the expected output matrix (1, m), 1 attributes since binary classification, m samples
    # layer_dims is an array of L + 1 elements, represents number of neurons in each layer (including input layer n^[0], n^[1], ..., n^[L]) where n^[0] is the input layer and n^[L] is the output layer
    # learning_rate is floating number
    # epochs is the number iterations for gradient descent 
    def __init__(self, x, y, layer_dims, actFuns, learning_rate, epochs):
        assert(layer_dims[-1] == 1) # binary classification, the output layer has 1 neuron
        self.input = x # input
        self.y = y # expected output
        self.m = x.shape[1] # the number of columns in x
        self.alpha = learning_rate
        self.epochs = epochs
        self.actFuns = actFuns # activation functions for each layer actFuns[l-1] to access layer l's activation functions
        L = len(layer_dims) - 1 # we don't count the input layer in total number of layers
        self.L = L # this is a L-level network 

        self.layer_dims = layer_dims
        params = {} # params contain the weight and bias for each layer, size 2L in total
        for l in range(1, L + 1): # loop through l = 1, 2 ... L
            params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01 # W_l is a weight matrix of (n^[l], n^[l-1]), note n^[0] is the input layer dimension
            params['B'+str(l)] = np.zeros((layer_dims[l], 1)) # B_l is the bias matrix of (n^[l], 1)
        self.params = params

        self.cache = {"A0": x} # need cache for backprob, store Z and A for l-layer
        self.costs = [] # cost array store value from cost functions
    
    def activate(self, Z, l):
        function = self.actFuns[l-1] # the activation function using in the l-layer
        if (function == "sigmoid"):
            return sigmoid(Z)
        elif (function == "tanh"):
            return np.tanh(Z)
        elif (function == "ReLU"):
            return ReLU(Z)

    def forward_prob(self, prediction=False, predictInput=np.array([])):
        if (prediction == False): # check if we are training or make a prediction
            A = self.input
        else:
            A = predictInput

        for l in range(1, self.L + 1): # loop through l = 1, 2 ... L
            A_last = A # A_last = A^[l-1]
            Z = np.dot(self.params['W'+str(l)], A_last) + self.params['B'+str(l)] # linear hypothesis, Z^[l] is (n^[l], m) matrix
            A = self.activate(Z, l)
            self.cache["A"+str(l)] = A # add to cache A^[l]
            self.cache["Z"+str(l)] = Z # add to cache Z^[l]
        return A
    
    # A should A[l] the activation at output layer, this cost function is the one for binary classification
    def cost_function(self, A):
        assert(A.shape == (1, self.m)) # only for binary classification
        assert(self.y.shape == (1, self.m))
        cost = (-1/self.m)*(np.dot(np.log(A), self.y.T) + np.dot(np.log(1-A), 1-self.y.T)) # we should get a (1, 1) matrix 
        return cost 

    # return the derivative of Z at l-layer respect to the activation function using at the l-layer
    def derivative(self, Z, l):
        function = self.actFuns[l-1] # the activation function using in the l-layer
        if (function == "sigmoid"):
            return Dsigmoid(Z)
        elif (function == "tanh"):
            return Dtanh(Z)
        elif (function == "ReLU"):
            return DReLU(Z)

    # AL is the activation from output layer, (1, m) matrix
    def backward_prob(self, AL):
        assert(AL.shape == (1, self.m))
        dAl = -(np.divide(self.y, AL)) + np.divide(1-self.y, 1 - AL) # the derivative of dCost/dA at output layer
        assert(dAl.shape == (1, self.m))
        for l in reversed(range(1, self.L + 1)): # from level L to 1
            dZl = dAl * self.derivative(self.cache["Z"+str(l)], l)
            assert(dZl.shape == (self.layer_dims[l], self.m))

            dWl = (1/self.m) * np.dot(dZl, self.cache["A"+str(l-1)].T)
            assert(dWl.shape == (self.layer_dims[l], self.layer_dims[l-1]))

            dBl = (1/self.m) * np.sum(dZl, axis=1, keepdims=True)
            assert(dBl.shape == (self.layer_dims[l], 1))

            dAl = np.dot(self.params["W"+str(l)].T, dZl) # dA(l-1), the dCost/dA at l-1 layer
            assert(dAl.shape == (self.layer_dims[l - 1], self.m))

            self.params["W" + str(l)] -= self.alpha * dWl # update parameters 
            self.params["B" + str(l)] -= self.alpha * dBl

    def train(self): # train the model

        for _ in range(self.epochs):
            AL = self.forward_prob() # AL is the activation of output layer
            assert(AL.shape == (1, self.m))
            self.costs.append(self.cost_function(AL)) # append the cost for this iteration
            self.backward_prob(AL)
    
    def predict(self, testX): # use current model to predict value Y
        predictions = self.forward_prob(prediction=True, predictInput=testX)
        return np.around(predictions)

