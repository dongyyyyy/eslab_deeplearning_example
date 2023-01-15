import numpy as np
import loss_function as loss 

def ReLU(x):
    ###########################################################


def dReLU(x):
    ###########################################################


def softmax(z):
    ###########################################################


class Network():
    def __init__(self, input_size, output_size, hidden_layers):

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
        ###########################################################
        ###########################################################

    def forward(self, x):  # x shape = batch X input_size
        ############### Write code ######################
        self.Z = dict()  # linear regression dict initialization
        self.A = dict()  # activation function dict initialization

        ###########################################################
        ###########################################################

        return x

    def backward(self, X, y):
        self.delta = dict()
        self.dw = dict()
        self.db = dict()

        batch_size = X.shape[0]

        # Calculate Gradients

        ###########################################################
        ###########################################################

    def update(self, alpha):
        ###########################################################
        
        ###########################################################