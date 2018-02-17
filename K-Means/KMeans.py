#!/usr/bin/env python3
"""
@author: Pulkit Maloo
"""
import os
import random
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

SEED = 5


def load_data(fname="test.csv"):
	""" returns numpy array of data and label col """
	fpath = os.path.join(os.getcwd(), "data", fname)
	df = pd.read_csv(fpath, header=None)
	df.columns = range(len(df.columns))
	df.iloc[:, -1] = df.iloc[:, -1].astype("category").cat.codes
	return df.iloc[:,:-1].values, df.iloc[:,-1].values


def distance(p1, p2, metric):
    if metric == "euclidean":
        return np.sqrt(np.sum(np.power((p1-p2), 2)))
    elif metric == "cityblock":
        return np.sum(np.abs(p1-p2))
    elif metric == "distance3":
        return distance3(p1, p2)
    elif metric == "distance4":
    	return distance4(p1, p2)
    elif metric == "cosine":
        xy = np.sqrt(np.sum(np.power(p1, 2)))*np.sqrt(np.sum(np.power(p2, 2)))
        return 1 - (np.dot(p1, p2)/xy)
    else:
        raise ValueError


class KMeans(object):
	def __init__(self, k, algo="lloyd", metric="euclidean", init="random", max_iter=1000, random_state=SEED):
		self.k = k
		self.init = init
		self.algo = algo
		self.metric = metric
		self.random_state = random_state
		self.max_iter = max_iter

	def get_init_centroids(self, X, k, init="random"):
		if init == "random":
			return X[np.random.choice(X.shape[0], self.k, replace=False), :]
		else:
			raise ValueError("Invalid method for Initialization")

	def stopping_cond(self, iterations, old_centroids, new_centroids):
		if self.max_iter < iterations:
			return True		
		if np.allclose(old_centroids, new_centroids):
			return True
		return False

	def recompute_centroids(self, X, clusters):
		clusters_points = [X[clusters==i] for i in range(self.k)]
		if self.metric == "euclidean":
			return np.array([np.mean(clusters_points[i], axis=0) for i in range(self.k)])
		elif self.metric == "cityblock":
			return np.array([np.median(clusters_points[i], axis=0) for i in range(self.k)])
		else:
			raise ValueError("centroid not defined for", self.metric)		

	def SSE(self, X, centroids, clusters):
		SSE_clusters = [np.sum(cdist(X[clusters==i], clusters[i])**2) for i in range(self.k)]
		return np.sum(SSE_clusters)

	def fit_lloyd(self, X):
		# Get Initial Centroids and other initializations	
		centroids = np.array(self.get_init_centroids(X, self.k, self.init))
		old_centroids = np.zeros(centroids.shape)
		clusters = np.zeros((X.shape[0], 1))
		iterations = 0

		print(centroids)

		while not self.stopping_cond(iterations, old_centroids, centroids):		
			# Compute Distance to all centroids
			distances = cdist(X, centroids, self.metric)
			# Assign points to closest centroids
			clusters = np.argmin(distances, axis=1)
			# Recompute centroids
			old_centroids = centroids
			centroids = self.recompute_centroids(X, clusters)					
						
			#print("centroids\n",centroids)
			#print(clusters)

			iterations += 1 
		
		self.iterations = iterations
		return centroids
		
	def fit_elkan(self, X):
		# Get Initial Centroids and other initializations	
		centroids = np.array(self.get_init_centroids(X, self.k, self.init))
		old_centroids = np.zeros(centroids.shape)
		clusters = np.zeros((X.shape[0], 1))
		iterations = 0

		##### Initialization ##########
		# Set lower bound to 0 for each pt x and center c
		lower = np.zeros((X.shape[0], self.k))
		upper = np.full(clusters.shape, np.inf)
		# Assign x to closest centroid
		#c = np.argmin(cdist(X, centroids, self.metric), axis=1)
		centroid_dist = cdist(centroids, centroids)
		for i in range(X.shape[0]):
			# Assign to the first centroid
			clusters[i] = 0
			d = distance(X[i], centroids[0], self.metric)
			lower[i, 0] = d
			# Then loop over all other centroids and exploit traingle ineq
			for c in range(1, self.k):
				if centroid_dist[clusters[i], c] < (2 * d):	# lemma 1
					d = distance(X[i], centroids[0], self.metric)
					lower[i, c] = d
					clusters[i] = c
			upper[i] = d
		print(lower)
		print(upper)





		##### Main Loop ##########
		while not self.stopping_cond(iterations, old_centroids, centroids):
			# Step 1
			# Step 2
			# Step 3
			# Step 4
			# Step 5
			# Step 6
			# Step 7
			break	# <-~~~~~~{<
			iterations +=1

		self.iterations = iterations
		return centroids

	def set_attributes(self, X, centroids):
		self.labels_ = np.argmin(cdist(X, centroids, self.metric), axis=1)	
		self.cluster_centers_ = centroids		
		#self.inertia_ = self.SSE(X, centroids, self.labels_)

	def fit(self, X):
		if self.algo is None or self.algo == "lloyd":
			centroids = self.fit_lloyd(X)
		elif self.algo == "elkan":
			centroids = self.fit_elkan(X)
		else:
			raise ValueError("Algorithm not found\nValid options: lloyd, elkan")
		
		self.set_attributes(X, centroids)

		print(self.cluster_centers_)
		print(self.labels_)
		print(self.iterations)
		#print("SSE:",self.inertia_)


if __name__ == '__main__':
	X, labels = load_data("test.csv")
	model = KMeans(k=2, algo="elkan")
	model.fit(X)
	print(labels)
