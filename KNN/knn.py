#!/usr/bin/env python

### Please see README for documentation 

from __future__ import division
import sys
import timeit
import numpy as np
from numpy.linalg import norm
import cPickle
from collections import Counter
from heapq import nsmallest
import math
import random
import itertools
np.random.seed(3)


class KNN(object):
    def __init__(self, k=7):
        self.k = k

    def train(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def test(self, X_test, y_test):
        correct = 0
        pred = []
        for X_ins, y_ins in zip(X_test, y_test):
            predict = self.predict(X_ins)
            pred.append(predict)
            if y_ins == predict:
                correct += 1
        return pred, round(correct/len(X_test), 4) * 100

    def predict(self, p):
        class_count = Counter(self.nearest_neighbours(p))
        return class_count.most_common()[0][0]

    def nearest_neighbours(self, p):
        distances = norm(self.X_train - p, axis=1)
        neighbors = zip(distances, self.y_train)
        k_nearest = nsmallest(self.k, neighbors, key=lambda x: x[0])
        return map(lambda x: x[1], k_nearest)


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
        
        knn = KNN(k=9)
        knn.train(X, y)
        models = (knn)

        cPickle.dump(models, open(model_file, "wb"), protocol=2)
        toc = timeit.default_timer()
        print "Time taken", int(toc - tic), "seconds"

    else:
        print "Testing", model, "model..."
        tic = timeit.default_timer()

        models = cPickle.load(open(model_file, "rb"))

        knn = models
        pred, score = knn.test(X, y)
        print("Writing to a file...")
        to_file(image, pred)

        print ("Accuracy", score, "%")
        toc = timeit.default_timer()
        print ("Time taken", int(toc - tic), "seconds")
