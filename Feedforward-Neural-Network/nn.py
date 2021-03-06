#!/usr/bin/env python

### Please see README for documentation 

from __future__ import division
import sys
import timeit
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import LabelBinarizer
import cPickle
from collections import Counter
import math
import random
import itertools
np.random.seed(3)

class NeuralNet(object):
    def __init__(self, alpha=0.3, iterations=4000, lambd=0.7, keep_prob=0.8, layer_dims=[192, 128, 64, 4]):
        self.set_parameters(alpha, iterations, lambd, keep_prob, layer_dims)

    def set_parameters(self, alpha, iterations, lambd, keep_prob, layer_dims):
        self.alpha = alpha
        self.iterations = iterations
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.layer_dims = layer_dims
        self.initialize_parameters(layer_dims)

    def initialize_parameters(self, layer_dims):
        self.parameters = {}
        for l in range(1, len(layer_dims)):
            self.parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l - 1])
            self.parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def softmax(self, Z):
        Z_exp = np.exp(Z - np.max(Z))
        Z_sum = np.sum(Z_exp, axis=0, keepdims=True)
        return Z_exp / Z_sum

    def relu(self, Z):
        return np.maximum(0, Z)

    def dropout(self, A):
        D = np.random.rand(A.shape[0], A.shape[1])
        D = D < self.keep_prob
        A = A * D
        A = A / self.keep_prob
        return A, D

    def linear_fwd_activation(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b

        if activation == "sigmoid":
            A = self.sigmoid(Z)
            cache = ((A_prev, W, b), Z)

        elif activation == "relu":
            A = self.relu(Z)
            A, D = self.dropout(A)
            cache = ((A_prev, W, b), Z, D)

        elif activation == "softmax":
            A = self.softmax(Z)
            cache = ((A_prev, W, b), Z)

        return A, cache

    def forward_propogation_with_dropout(self, X):
        caches = []
        L = len(self.layer_dims) - 1

        A = X
        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            A, cache = self.linear_fwd_activation(A_prev, W, b, "relu")
            caches.append(cache)

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        AL, cache = self.linear_fwd_activation(A, W, b, "softmax")
        caches.append(cache)

        return AL, caches

    def cross_entropy_softmax(self, AL, ZL, Y):
#        return - np.sum(np.multiply(Y, np.log(AL)))
        return - np.sum(np.multiply(Y, ZL - np.log(np.sum(np.exp(ZL), axis=0, keepdims=True))))

    def compute_cost(self, AL, Y, caches):
        m = Y.shape[1]
        L = len(self.layer_dims) - 1

#       For Sigmoid
#        cross_entropy_cost = (-1/m) * np.sum( np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)) )

        ZL = caches[-1][1]
        cross_entropy_cost = self.cross_entropy_softmax(AL, ZL, Y) / m

        W_sum = 0
        for l in range(1, L+1):
            W_sum += np.sum(np.square(self.parameters["W"+str(l)]))
        regularized_cost = self.lambd * W_sum / (2 * m)

        cost = np.squeeze(cross_entropy_cost) + regularized_cost
        return cost

    def linear_backward(self, dZ, cache, l):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        W = self.parameters["W" + str(l)]

        dW = (1/m) * np.dot(dZ, A_prev.T) + (self.lambd * W) / m
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def dZL(self, AL, Y):
        return AL - Y

    def linear_activation_backward(self, dA, cache, activation, l):
        ((A_prev, W, b), Z, D) = cache

        if activation == "relu":
            dA = dA * D                 # Dropout
            dA = dA / self.keep_prob
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0

        elif activation == "sigmoid":
            t = self.sigmoid(-Z)
            dZ = dA * t * (1-t)

        elif activation == "softmax":
            t = self.softmax(Z)
            dZ = t - Z

        dA_prev, dW, db = self.linear_backward(dZ, (A_prev, W, b), l)

        return dA_prev, dW, db

    def backpropogation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        cache = caches[L-1]

#        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
#        grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = self.linear_activation_backward(dAL, cache, "sigmoid", L)

        grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = self.linear_backward(self.dZL(AL, Y), cache[0], L)
        for l in range(L-1, 0, -1):
            cache = caches[l-1]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA"+str(l+1)], cache, "relu", l)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l)] = dW_temp
            grads["db" + str(l)] = db_temp

        return grads

    def lrate(self, epoch):
        drop = 0.5
        epochs_drop = 1000
        lr = self.alpha * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lr

    def update_parameters(self, grads, i):
        L = len(self.layer_dims) - 1
        for l in range(1, L+1):
            self.parameters["W"+str(l)] -= self.alpha * grads["dW"+str(l)]
            self.parameters["b"+str(l)] -= self.alpha * grads["db"+str(l)]

    def train(self, X_train, y_train):
        X = X_train
        Y = y_train

        costs = [(0,0)]
        for i in xrange(self.iterations + 1):
            AL, caches = self.forward_propogation_with_dropout(X)

            cost = self.compute_cost(AL, Y, caches)

            grads = self.backpropogation(AL, Y, caches)

            self.update_parameters(grads, i)

            if i%100 == 0:
                if i % 1000 == 0:
                    self.alpha /= 1.5
                costs.append((i, cost))
                _, acc = self.test(X, Y)
                print "Iteration", i, "->", "Accuracy", acc, "|| Cost", cost

    def forward_propogation(self, X):
        L = len(self.layer_dims) - 1

        A = X
        for l in range(1, L):
            A_prev = A
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)

        W = self.parameters["W" + str(L)]
        b = self.parameters["b" + str(L)]
        ZL = np.dot(W, A) + b
        AL = self.softmax(ZL)

        return AL

    def test(self, X_test, y_test):
        X_t = X_test
        Y_t = y_test
        AL = self.forward_propogation(X_t)

        original = Y_t.argmax(0)
        predicted = AL.argmax(0)
        m = len(original)
        incorrect = np.count_nonzero(original - predicted)
        return predicted, round((m - incorrect) / m, 4) * 100


def read_file(fname, shuffle_data=True):
    print "Reading data from", fname, "..."
    image = np.loadtxt(fname, usecols=0, dtype=str)  # .reshape(len(X), 1)
    X = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
    y = np.loadtxt(fname, usecols=1, dtype=int)  # .reshape(len(X), 1)

    if shuffle_data:
        shuffle_indices = range(len(y))
        np.random.shuffle(shuffle_indices)
        X  = X[shuffle_indices, ]
        y = y[shuffle_indices, ]
        image = image[shuffle_indices, ]

    return list(image), X/255, y


def transform_Y_for_NN(Y):
    lb = LabelBinarizer()
    lb.fit(Y)
    return lb


def to_file(image, pred):
    f = open('output.txt', 'w')
    for line in xrange(len(image)):
        f.write(image[line] + ' ' + str(pred[line]) + '\n')
    f.close()


if __name__ == "__main__":
    task, fname, model_file, model = sys.argv[1:]

    image, X, y = read_file(fname, shuffle_data=True)

    if task == "train":
        print "Training", model, "model..."
        tic = timeit.default_timer()
        lb = transform_Y_for_NN(y)
        Y_lb = lb.transform(y)

        alpha = 0.3
        iterations = 2000
        lambd = 0.5
        keep_prob = 0.6
        layers = [X.shape[1]] + [193, 64] + [Y_lb.shape[1]]

        nnet = NeuralNet(alpha=alpha, iterations=iterations, lambd=lambd,
                         keep_prob=keep_prob, layer_dims=layers)

        nnet.train(X.T, Y_lb.T)

        models = (lb, nnet)

        cPickle.dump(models, open(model_file, "wb"), protocol=2)
        toc = timeit.default_timer()
        print "Time taken", int(toc - tic), "seconds"

    else:
        print "Testing", model, "model..."
        tic = timeit.default_timer()

        models = cPickle.load(open(model_file, "rb"))


        lb, nnet = models
        Y_lb = lb.transform(y)
        pred, score = nnet.test(X.T, Y_lb.T)
        pred *= 90
        print("Writing to a file...")
        to_file(image, pred)

        print ("Accuracy", score, "%")
        toc = timeit.default_timer()
        print ("Time taken", int(toc - tic), "seconds")


