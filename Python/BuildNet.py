import numpy as np
import tensorflow as tf
import scipy.io
from random import shuffle





def weight_variable(shape,seed):
    #initial = tf.truncated_normal(shape, stddev=1,dtype=tf.float32,seed=seed) #save


    initial = tf.random.normal(shape,  mean = 0, stddev=1,dtype=tf.float32,seed=seed)
    return tf.Variable(initial)

def bias_variable(shape,seed):

    initial = tf.random.normal(shape=shape,seed=seed) #sav

    #initial = tf.constant(1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)


class Neural_Network2Layers(object):
    def __init__(self,inputSize,hiddenSize1,hiddenSize2,outputSize,doBN=0):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2

        seed = tf.set_random_seed(2)
        self.W1 = weight_variable([self.inputSize, self.hiddenSize1],seed)
        self.b1 = bias_variable([self.hiddenSize1],seed)
        self.W2 = weight_variable([self.hiddenSize1,self.hiddenSize2],seed)
        self.b2 = bias_variable([self.hiddenSize2],seed)
        self.W3 = weight_variable([self.hiddenSize2,self.outputSize],seed)
        self.b3 = bias_variable([self.outputSize],seed)

    def forward(self, X):
        #forward propagation through our network

        hidden_layer1 = tf.add(tf.matmul(X, self.W1), self.b1)
        hidden_layer1 = tf.nn.tanh(hidden_layer1)
        hidden_layer2 = tf.add(tf.matmul(hidden_layer1, self.W2), self.b2)
        hidden_layer2 = tf.nn.tanh(hidden_layer2)
        o = tf.add(tf.matmul(hidden_layer2, self.W3), self.b3)

        return o


class Neural_Network2LayersPos(object):
    def __init__(self,inputSize,hiddenSize1,hiddenSize2,outputSize,doBN=0):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2

        seed = tf.set_random_seed(2)
        self.W1 = weight_variable([self.inputSize, self.hiddenSize1],seed)
        self.b1 = bias_variable([self.hiddenSize1],seed)
        self.W2 = weight_variable([self.hiddenSize1,self.hiddenSize2],seed)
        self.b2 = bias_variable([self.hiddenSize2],seed)
        self.W3 = weight_variable([self.hiddenSize2,self.outputSize],seed)
        self.b3 = bias_variable([self.outputSize],seed)

    def forward(self, X):
        #forward propagation through our network

        hidden_layer1 = tf.add(tf.matmul(X, self.W1), self.b1)
        hidden_layer1 = tf.nn.tanh(hidden_layer1)
        hidden_layer2 = tf.add(tf.matmul(hidden_layer1, self.W2), self.b2)
        hidden_layer2 = tf.nn.tanh(hidden_layer2)
        o =tf.add(tf.matmul(hidden_layer2, self.W3), self.b3)
        #o = tf.nn.relu(o)
        return o




class Neural_Network3Layers(object):
    def __init__(self,inputSize,hiddenSize1,hiddenSize2,hiddenSize3,outputSize,doBN):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2
        self.hiddenSize3 = hiddenSize3
        self.doBN = doBN

        seed = tf.set_random_seed(2)
        self.W1 = weight_variable([self.inputSize, self.hiddenSize1],seed)
        self.b1 = bias_variable([self.hiddenSize1],seed)
        self.W2 = weight_variable([self.hiddenSize1,self.hiddenSize2],seed)
        self.b2 = bias_variable([self.hiddenSize2],seed)
        self.W3 = weight_variable([self.hiddenSize2,self.hiddenSize3],seed)
        self.b3 = bias_variable([self.hiddenSize3],seed)
        self.W4 = weight_variable([self.hiddenSize3, self.outputSize], seed)
        self.b4 = bias_variable([self.outputSize], seed)

    def forward(self, X):
        #forward propagation through our network
        np.random.seed(9001)
        noise1 = np.random.normal(0, 0.5, self.hiddenSize1)
        noise2 = np.random.normal(0, 0.5, self.hiddenSize2)
        noise3 = np.random.normal(0, 0.5, self.hiddenSize3)

        hidden_layer1 = tf.add(tf.matmul(X, self.W1), self.b1)#+noise1
        hidden_layer1 = tf.nn.tanh(hidden_layer1)
        if self.doBN:
            hidden_layer1 = tf.layers.batch_normalization(hidden_layer1)

        hidden_layer2 = tf.add(tf.matmul(hidden_layer1, self.W2), self.b2)#+noise2
        hidden_layer2 = tf.nn.tanh(hidden_layer2)
        if self.doBN:
            hidden_layer2 = tf.layers.batch_normalization(hidden_layer2)


        hidden_layer3 = tf.add(tf.matmul(hidden_layer2, self.W3), self.b3)#+noise3
        hidden_layer3 = tf.nn.tanh(hidden_layer3)
        if self.doBN:
            hidden_layer3 = tf.layers.batch_normalization(hidden_layer3)

        o = tf.add(tf.matmul(hidden_layer3, self.W4), self.b4)

        return o



class Neural_Network1Layer(object):
    def __init__(self,inputSize,hiddenSize1,outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1


        seed = tf.set_random_seed(2)
        self.W1 = weight_variable([self.inputSize, self.hiddenSize1],seed)
        self.b1 = bias_variable([self.hiddenSize1],seed)
        self.W2 = weight_variable([self.hiddenSize1,self.outputSize],seed)
        self.b2 = bias_variable([self.outputSize],seed)


    def forward(self, X):
        #forward propagation through our network

        hidden_layer1 = tf.add(tf.matmul(X, self.W1), self.b1)
        hidden_layer1 = tf.nn.tanh(hidden_layer1)

        o = tf.add(tf.matmul(hidden_layer1, self.W2), self.b2)

        return o


class Neural_Network4Layers(object):
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2
        self.hiddenSize3 = hiddenSize3
        self.hiddenSize4 = hiddenSize4

        seed = tf.compat.v1.set_random_seed(2)
        self.W1 = weight_variable([self.inputSize, self.hiddenSize1], seed)
        self.b1 = bias_variable([self.hiddenSize1], seed)
        self.W2 = weight_variable([self.hiddenSize1, self.hiddenSize2], seed)
        self.b2 = bias_variable([self.hiddenSize2], seed)
        self.W3 = weight_variable([self.hiddenSize2, self.hiddenSize3], seed)
        self.b3 = bias_variable([self.hiddenSize3], seed)
        self.W4 = weight_variable([self.hiddenSize3, self.hiddenSize4], seed)
        self.b4 = bias_variable([self.hiddenSize4], seed)
        self.W5 = weight_variable([self.hiddenSize4, self.outputSize], seed)
        self.b5 = bias_variable([self.outputSize], seed)

    def forward(self, X):
        # forward propagation through our network

        #hidden_layer1 = tf.add(tf.matmul(X, self.W1), self.b1)
        hidden_layer1 = tf.matmul(X, self.W1)+self.b1
        hidden_layer1 = tf.nn.tanh(hidden_layer1)
        #hidden_layer2 = tf.add(tf.matmul(hidden_layer1, self.W2), self.b2)
        hidden_layer2 = tf.matmul(hidden_layer1, self.W2)+self.b2
        hidden_layer2 = tf.nn.tanh(hidden_layer2)
        #hidden_layer3 = tf.add(tf.matmul(hidden_layer2, self.W3), self.b3)
        hidden_layer3 = tf.matmul(hidden_layer2, self.W3)+self.b3
        hidden_layer3 = tf.nn.tanh(hidden_layer3)
        #hidden_layer4 = tf.add(tf.matmul(hidden_layer3, self.W4), self.b4)
        hidden_layer4 = tf.matmul(hidden_layer3, self.W4)+self.b4
        hidden_layer4 = tf.nn.tanh(hidden_layer4)
        o = tf.matmul(hidden_layer4, self.W5)+self.b5

        return o


class Neural_NetworkDict4Layers(object):
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, hiddenSize4, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize1 = hiddenSize1
        self.hiddenSize2 = hiddenSize2
        self.hiddenSize3 = hiddenSize3
        self.hiddenSize4 = hiddenSize4
        self.dict = {}

        seed = tf.set_random_seed(2)
        self.dict['W1'] = weight_variable([self.inputSize, self.hiddenSize1], seed)
        self.dict['b1'] = bias_variable([self.hiddenSize1], seed)
        self.dict['W2'] = weight_variable([self.hiddenSize1, self.hiddenSize2], seed)
        self.dict['b2'] = bias_variable([self.hiddenSize2], seed)
        self.dict['W3'] = weight_variable([self.hiddenSize2, self.hiddenSize3], seed)
        self.dict['b3'] = bias_variable([self.hiddenSize3], seed)
        self.dict['W4'] = weight_variable([self.hiddenSize3, self.hiddenSize4], seed)
        self.dict['b4'] = bias_variable([self.hiddenSize4], seed)
        self.dict['W5'] = weight_variable([self.hiddenSize4, self.outputSize], seed)
        self.dict['b5'] = bias_variable([self.outputSize], seed)

    def forward(self, X):
        # forward propagation through our network


        hidden_layer1 = tf.matmul(X, self.dict['W1'])+self.dict['b1']
        hidden_layer1 = tf.nn.tanh(hidden_layer1)
        hidden_layer2 = tf.matmul(hidden_layer1, self.dict['W2'])+self.dict['b2']
        hidden_layer2 = tf.nn.tanh(hidden_layer2)
        hidden_layer3 = tf.matmul(hidden_layer2, self.dict['W3'])+self.dict['b3']
        hidden_layer3 = tf.nn.tanh(hidden_layer3)
        hidden_layer4 = tf.matmul(hidden_layer3, self.dict['W4'])+self.dict['b4']
        hidden_layer4 = tf.nn.tanh(hidden_layer4)
        o = tf.matmul(hidden_layer4, self.dict['W5'])+self.dict['b5']

        return o