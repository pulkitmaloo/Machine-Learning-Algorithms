import numpy as np


def get_weight(dist, centroids):
    min_dist = np.zeros(dist.shape)
    min_dist[range(dist.shape[0]), np.argmin(dist, axis=1)] = 1
    count = np.array([np.count_nonzero(min_dist[:, i])
                      for i in range(centroids.shape[0])])
    return count/np.sum(count)


def distance_(data, centroids):
    dist = np.sum((data[:, np.newaxis, :] - centroids)**2, axis=2)
    return dist


def ScalableKMeansPlusPlus(data, k, l, iter_=5):

    """ Apply the KMeans|| clustering algorithm

    Parameters:
      data     ndarrays data
      k        number of cluster
      l        number of point sampled in each iteration

    Returns:   the final centroids finded by KMeans||

    """
    centroids = data[np.random.choice(range(data.shape[0]), 1), :]

    for i in range(iter_):
        # Get the distance between data and centroids
        dist = distance_(data, centroids)

        # Calculate the cost of data with respect to the centroids
        norm_const = np.sum(np.min(dist, axis=1))

        # Calculate the distribution for sampling l new centers
        p = np.min(dist, axis=1)/norm_const

        # Sample the l new centers and append them to the original ones
        sample_new = data[np.random.choice(range(len(p)), l, p=p), :]
        centroids = np.r_[centroids, sample_new]

    # reduce k*l to k using KMeans++
    dist = distance_(data, centroids)
    weights = get_weight(dist, centroids)

    return centroids[np.random.choice(len(weights), k, replace=False,
                                      p=weights), :]
