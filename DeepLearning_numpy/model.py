import numpy as np
from .loss_function import *

def ReLU(x):
    return np.maximum(0, x)
    #return np.fmax(0, x)


def dReLU(x):
    # relu가 x >= 0 일때, y=x이면 아래와 같다.
    return np.where(x >= 0, 1, 0)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


class Network():
    def __init__(self, input_size, output_size, hidden_layers):
        # Build a neural network with
        # input size: 'input_size'
        # output size: 'output_size'
        # number of hidden layers as 'hidden_layers' where
        # hidden_layers is a list of number nodes in layer

        ############### Write code ######################

        # Define and initialize Weights and Bias in Dictionary format
        init_coef = 0.1
        self.num_layers = len(hidden_layers) + 1

        # 첫번째 layer wieght bais 초기화
        self.input_size = input_size
        self.size_of_input = self.input_size
        self.output_size = output_size
        self.W = dict()
        self.b = dict()
        # update로 모두 처리
        for i in range(1, self.num_layers):  # 1에서부터 시작
            self.W.update({i: init_coef * np.random.randn(self.size_of_input, hidden_layers[i - 1])})
            self.b.update({i: init_coef * np.random.randn(1, hidden_layers[i - 1])})
            self.size_of_input = hidden_layers[i - 1]

        self.W.update({self.num_layers: init_coef * np.random.randn(self.size_of_input, output_size)})
        self.b.update({self.num_layers: init_coef * np.random.randn(1, output_size)})

    def forward(self, x):  # x shape = batch X input_size
        ############### Write code ######################
        self.Z = dict()  # linear regression dict initialization
        self.A = dict()  # activation function dict initialization

        for i in range(1, self.num_layers + 1):
            x = np.matmul(x, self.W[i]) + self.b[i]
            self.Z.update({i: np.copy(x)})
            if i < self.num_layers:  # Output layer 의 경우 activation function 을 쓰지 않기 때문에
                x = ReLU(x)
                self.A.update({i: np.copy(x)})
        x = softmax(x)
        self.A.update({self.num_layers: x})

        return x

    def backward(self, X, y):
        self.delta = dict()
        self.dw = dict()
        self.db = dict()

        batch_size = X.shape[0]

        # Calculate Gradients

        self.delta = {self.num_layers: self.A[self.num_layers] - one_hot(y, self.output_size)}
        self.dW = {self.num_layers: np.matmul(self.A[self.num_layers - 1].T,
                                              self.delta[self.num_layers]) / batch_size}
        self.db = {self.num_layers: np.matmul(np.ones((1, batch_size)), self.delta[self.num_layers]) / batch_size}

        for i in range(self.num_layers - 1, 1, -1):
            self.delta.update({i: np.matmul(self.delta[i + 1], self.W[i + 1].T) * dReLU(self.Z[i])})

            self.dW.update({i: np.matmul(self.A[i - 1].T, self.delta[i]) / batch_size})
            self.db.update({i: np.matmul(np.ones((1, batch_size)), self.delta[i]) / batch_size})

        self.delta.update({1: np.matmul(self.delta[2], self.W[2].T) * dReLU(self.Z[1])})
        self.dW.update({1: np.matmul(X.T, self.delta[1]) / batch_size})
        self.db.update({1: np.matmul(np.ones((1, batch_size)), self.delta[1]) / batch_size})

    # Batch stochastic gradient descent
    def update(self, alpha):
        for i in range(1, self.num_layers + 1):
            self.W[i] = self.W[i] - alpha * (self.dW[i])
            self.b[i] = self.b[i] - alpha * (self.db[i])