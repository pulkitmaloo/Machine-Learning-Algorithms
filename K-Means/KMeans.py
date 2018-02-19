#!/usr/bin/env python3
"""
@author: Pulkit Maloo
"""

import os
import time
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from kmeans_utils import *

SEED = np.random.randint(1024)


def load_data(fname="test.csv", labels=True):
    """ returns numpy array of data and label col """
    fpath = os.path.join(os.getcwd(), "data", fname)
    df = pd.read_csv(fpath, header=None)
    df.columns = range(len(df.columns))
    if labels:
        df.iloc[:, -1] = df.iloc[:, -1].astype("category").cat.codes
        return df.iloc[:, :-1].values, df.iloc[:, -1].values
    else:
        return np.array(df[[0]].applymap(lambda x: list(map(int, x.split()))
                        )[[0]].values.tolist()).reshape(100000, 2)


def distance3(p1, p2, p=2):
    diff = p1-p2
    f_plus = np.vectorize(lambda x: max(x, 0))
    f_minus = np.vectorize(lambda x: max(-x, 0))
    return np.power(np.sum(f_plus(diff))**p + np.sum(f_minus(diff))**p, 1/p)


def distance4(p1, p2, p=2):
    num = distance3(p1, p2)
    den = np.sum(np.maximum(np.abs(p1), np.abs(p2), np.abs(p1-p2)))
    return num/den


def distance(p1, p2, metric, axis=None):
    if metric == "euclidean":
        return np.sqrt(np.sum(np.power((p1-p2), 2), axis=axis))
    elif metric == "cityblock":
        return np.sum(np.abs(p1-p2), axis=axis)
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
    def __init__(self, k, algo="elkan", metric="euclidean", init="kmeans++",
                 max_iter=50, random_state=SEED):
        self.k = k
        self.init = init
        self.algo = algo
        self.metric = metric
        self.max_iter = max_iter
        self.dist_calc = 0
        self.random_state = random_state
        np.random.seed(SEED)

    def __repr__(self):
        res = "KMeans("
        res += "K=" + str(self.k)
        res += ", algo=" + self.algo
        res += ", metric=" + self.metric
        res += ", max_iter=" + str(self.max_iter)
        res += ", init=" + self.init
        res += ", random_state=" + str(self.random_state)
        res += ")"
        return res

    def kpp(self, X, k):
        try:
            return ScalableKMeansPlusPlus(X, k, 1+k//10)
        except:
            pass
        mu = self.get_init_centroids(X, 1, "random")
        while len(mu) < k:
            D2 = np.min(cdist(X, mu)**2, axis=1)
            probs = D2/D2.sum()
            idx_possible = np.where(probs >= np.random.random())[0]
            if not (idx_possible.size > 0):
                continue
            idx_possible = idx_possible.reshape(idx_possible.shape[0], -1)
            idx = self.get_init_centroids(idx_possible, 1, "random")[0][0]
            mu = np.append(mu, np.array([X[idx]]), axis=0)
        return mu

    def get_init_centroids(self, X, k, init="kmeans++"):
        if init == "random":
            idx = np.random.randint(X.shape[0], size=k)
            return X[idx, :]
        elif init == "kmeans++":
            return self.kpp(X, k)
        else:
            raise ValueError("Invalid method for Initialization")

    def stopping_cond(self, iterations, old_centroids, new_centroids):
        if self.max_iter <= iterations:
            return True
        if np.allclose(old_centroids, new_centroids):
            return True
        return False

    def recompute_centroids(self, X, clusters):
        clusters_points = [X[(clusters == i)[:, 0]] for i in range(self.k)]

        if self.metric == "cityblock":
            return np.array([np.median(clusters_points[i], axis=0)
                             for i in range(self.k)])
        else:   # euclidean
            return np.array([np.mean(clusters_points[i], axis=0)
                             for i in range(self.k)])

    def SSE(self, X, centroids, clusters):
        SSE_clusters = [np.sum(distance(X[(clusters == i)], centroids[i],
                                        metric=self.metric, axis=1)**2)
                        for i in range(self.k)]
        return np.sum(SSE_clusters)

    def fit_lloyd(self, X):
        # Get Initial Centroids and other initializations
        centroids = self.init_centroids
        old_centroids = np.zeros(centroids.shape)
        clusters = np.zeros((X.shape[0], 1), dtype=int)
        iterations = 0

#        print(self.algo,"initial_centroids\n", centroids, "\n")

        while not self.stopping_cond(iterations, old_centroids, centroids):
            # Compute Distance to all centroids
            distances = cdist(X, centroids, self.metric)
            self.dist_calc += distances.shape[0] * distances.shape[1]
            # Assign points to closest centroids
            clusters = np.argmin(distances, axis=1).reshape(X.shape[0], 1)
            # Recompute centroids
            old_centroids = centroids
            centroids = self.recompute_centroids(X, clusters)

            iterations += 1

# ==========  Debug Statements ================================================
#             print(self.algo,"iterations", iterations)
#             print(self.algo,"centroids\n", centroids)
#             print(self.algo,"clusters\n", clusters[:,0])
#             print()
# =============================================================================

        self.iterations = iterations
        return centroids

    def fit_elkan(self, X):
        # Get Initial Centroids and other initializations
        centroids = self.init_centroids
        old_centroids = np.zeros(centroids.shape)
        clusters = np.zeros((X.shape[0], 1), dtype=int)
        iterations = 0

#        print(self.algo,"initial_centroids\n", centroids, "\n")

        # #### Initialization ##########
        # Set lower bound to 0 for each pt x and center c
        lower = np.zeros((X.shape[0], self.k))
        upper = np.full(clusters.shape, np.inf)
        # Assign x to closest centroid
        centroid_dist = cdist(centroids, centroids)
        self.dist_calc += centroids.shape[0] ** 2

        for i in range(X.shape[0]):
            # Assign to the random centroid
            r_c = np.random.randint(0, self.k)
            clusters[i] = r_c
            d = distance(X[i], centroids[r_c], self.metric)
            self.dist_calc += 1
            lower[i, r_c] = d

            # loop over all other centroids and exploit traingle ineq
            for c in range(0, self.k):
                if c == r_c:
                    continue
                if not centroid_dist[clusters[i], c] >= (2 * d):    # lemma 1
                    # Assign point to c if closer
                    d2 = distance(X[i], centroids[c], self.metric)
                    self.dist_calc += 1
                    lower[i, c] = d2
                    if d2 < d:
                        clusters[i] = c
                        d = d2

            upper[i] = d

# ========== Debug Statements  ================================================
#         print(self.algo,"iterations", iterations)
#         print(self.algo,"centroids\n", centroids)
#         print(self.algo,"clusters\n", clusters[:,0])
#         print("matches with lloyd", np.array_equal(clusters[:,0],
#                     np.argmin(cdist(X, centroids, self.metric), axis=1)))
#         print()
# =============================================================================

        # #### Main Loop ##########
        while not self.stopping_cond(iterations, old_centroids, centroids):

            # Step 1
            centroid_dist = cdist(centroids, centroids, self.metric)
            self.dist_calc += centroids.shape[0] ** 2
#            centroid_dist_nan = np.copy(centroid_dist)
#            np.fill_diagonal(centroid_dist_nan, np.nan)
#            s_c = np.nanmin(centroid_dist_nan, axis=0) / 2

            # Step 2
#            step2_idx = np.nonzero(np.array([upper[i] <= s_c[clusters[i]]
#                                             for i in range(X.shape[0])]))[0]

            # Step 3
            for c in range(self.k):
                # To-Do -> filter rows on each step
                # convert upper and lower to masked arrays
                cond1 = clusters != c
                cond2 = upper > lower[:, c].reshape(lower.shape[0], 1)
                cond3 = upper > centroid_dist[clusters, c]/2
                cond_and = np.logical_and(np.logical_and(cond1, cond2), cond3)
                step3_idx_1 = np.where(cond_and)[0]
                if not step3_idx_1.size > 0:
                    continue

                upper[step3_idx_1] = distance(X[step3_idx_1, :],
                         centroids[clusters[step3_idx_1].reshape(-1)],
                         self.metric, axis=1).reshape(step3_idx_1.shape[0], 1)
                self.dist_calc += step3_idx_1.shape[0]

                cond3b_1 = np.greater(upper, lower[:, c].reshape(X.shape[0], 1),
                                      where=cond_and)
                cond3b_2 = np.greater(upper, centroid_dist[clusters, c]/2,
                                      where=cond_and)

                cond_or = np.logical_or(cond3b_1, cond3b_2)
                cond_and_or = np.logical_and(cond_and, cond_or)
                step3_idx_2 = np.where(cond_and_or)[0]
                if not step3_idx_2.size > 0:
                    continue

                lower[step3_idx_2, c] = distance(X[step3_idx_2], centroids[c],
                                                 self.metric, axis=1)
                self.dist_calc += step3_idx_2.shape[0]

                cond_f = np.greater(upper, lower[:, c].reshape(X.shape[0], 1),
                                    where=cond_and_or)
                step3_idx_3 = np.where(np.logical_and(cond_and_or, cond_f))[0]
                if not step3_idx_3.size > 0:
                    continue

                clusters[step3_idx_3] = c
                upper[step3_idx_3] = lower[step3_idx_3, c].reshape(
                                                    step3_idx_3.shape[0], 1)

            # Step 4
            new_centroids = self.recompute_centroids(X, clusters)
            newcentroid_dist = distance(new_centroids, centroids, self.metric,
                                        axis=1)
            self.dist_calc += self.k

            # Step 5
            for c in range(self.k):
                lower[:, c] = np.maximum(lower[:, c] - newcentroid_dist[c], 0)
#                for i in range(X.shape[0]):
#                    l = lower[i, c] - newcentroid_dist[c, c]
#                    lower[i, c] = np.maximum(0, l)

            # Step 6
            upper += newcentroid_dist[clusters]
#            for i in range(X.shape[0]):
#                upper[i] += newcentroid_dist[clusters[i], clusters[i]]

            # Step 7
            old_centroids = centroids
            centroids = new_centroids

            iterations += 1

# ========== Debug Statements  ================================================
#             print(self.algo,"iterations", iterations)
#             print(self.algo,"centroids\n", centroids)
#             print(self.algo,"clusters\n", clusters[:,0])
#             print("matches with lloyd", np.array_equal(clusters[:,0],
#                         np.argmin(cdist(X, centroids, self.metric), axis=1)))
#             print(np.argmin(cdist(X, centroids, self.metric), axis=1))
#             print()
# =============================================================================

        self.iterations = iterations
        return centroids

    def set_attributes(self, X, centroids, time_taken):
        self.labels_ = np.argmin(cdist(X, centroids, self.metric), axis=1)
        self.cluster_centers_ = centroids
        self.time = time_taken
        self.inertia_ = self.SSE(X, centroids, self.labels_)

    def print_model(self):
        print("Iterations", self.iterations)
        print("Total distance calculations", self.dist_calc)
        print("Total sum of squared errors", round(self.inertia_, 2))
        print("Time taken", round(self.time, 3), "seconds\n")

    def fit(self, X):
        print(self)

        self.init_centroids = self.get_init_centroids(X, self.k, self.init)
        print("Initial centroids chosen")

        print("Finding clusters...", end=" ")
        tic = time.time()
        if self.algo is None or self.algo == "lloyd":
            centroids = self.fit_lloyd(X)
        elif self.algo == "elkan":
            centroids = self.fit_elkan(X)
        else:
            raise ValueError("Algo not found\nValid options: lloyd, elkan")
        toc = time.time()
        print(centroids.shape[0], "clusters found")

        self.set_attributes(X, centroids, toc-tic)

        self.print_model()

    def score(self, labels):
        clf_df = pd.DataFrame({"Class": labels, "Predict": self.labels_})
        clf_cnt = clf_df.groupby(list(clf_df.columns)[::-1]).size()
        clf_cnt = clf_cnt.reset_index(name='count')
#        print(clf_cnt)
        correct_predict = sum(clf_cnt.groupby(["Predict"],
                                              sort=False)["count"].max())

        print("Accuracy =", round(100*correct_predict/X.shape[0], 2), "%")


def check_models(model1, model2):
    if not (np.allclose(model1.cluster_centers_, model2.cluster_centers_) or
       np.allclose(model1.labels_, model2.labels_) or
       model1.iterations == model2.iterations):
        raise ValueError("Find the bug")


if __name__ == '__main__':
    X = load_data("birch.txt", False)
    n_clusters = 20
    init = "kmeans++"
    lloyd_kmeans = KMeans(k=n_clusters, algo="lloyd", init=init)
    lloyd_kmeans.fit(X)

    elkan_kmeans = KMeans(k=n_clusters, algo="elkan", init=init)
    elkan_kmeans.fit(X)

#    print("Predicted\n", elkan_kmeans.labels_)
#    print("Class\n", labels)
#    elkan_kmeans.score(labels)
    speedup = lloyd_kmeans.dist_calc/elkan_kmeans.dist_calc

    print("\nk =", n_clusters)
    print("iterations\t", elkan_kmeans.iterations)
    print("standard  \t", lloyd_kmeans.dist_calc)
    print("fast      \t", elkan_kmeans.dist_calc)
    print("speedup   \t", round(speedup, 2))
    check_models(elkan_kmeans, lloyd_kmeans)  # remove this
