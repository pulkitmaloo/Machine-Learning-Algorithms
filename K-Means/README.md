### K-Means

The K-Means algorithm source code can be found in the file `KMeans.py`

I have implemented a Class `KMeans` which has the following parameters:

* **K**: Number of Clusters (*required*)
* **algo**: Algorithm to use, *options*: lloyd, elkan, *default*: lloyd
* **metric**: Distance metric to use, *options*: any SciPy supported distance function passed as string, distance3, distance4,
	*default*: euclidean <dt> Note: distance3, distance4 are functions</dt>
* **max_iter**: Maximum number of iterations to run, *default*: 50
* **init**: Initialization to use for initial centroids, *options*: random, kmeans++, *default*: kmeans++
* **random_state**: Random seed, *default*: random integer

___

To run KMeans on a dataset use the function: **KMeans.fit(X)** where **X** should be a numpy ndarray

To get accuracy of the clusters formed against true class labels use the function: **KMeans.score(labels)**

To print the summary of the run of the algorithm use the function: **KMeans.print_model()**
___

After running the algorithm to form clusters, the following parameters can also be accessed:

* **KMeans.labels_**: To get the cluster assignments for each data point
* **KMeans.iterations**: To get the iterations taken to convergence
* **KMeans.dist_calc**: To get the total number of distance calculations
* **KMeans.cluster_centers_**: To get the final cluster centroids
* **KMeans.intertia_**: To get the Sum of Squared Errors
* **KMeans.time**: To get the time taken to run the algorithm
___
```
Sample run of the algorithm on the Iris dataset
KMeans(K=3, algo=lloyd, metric=euclidean, max_iter=50, init=kmeans++, random_state=614)
Initial centroids chosen
Finding clusters... 3 clusters found
Iterations 5
Total distance calculations 2250
Total sum of squared errors 78.85
Time taken 0.002 seconds

   Predict  Class  count
0        0      1     48
1        0      2     14
2        1      1      2
3        1      2     36
4        2      0     50
Accuracy = 89.33 %
```
___
Comparison of Lloyd KMeans vs Elkan's Accelerated K-Means
```
KMeans(K=3, algo=lloyd, metric=euclidean, max_iter=50, init=kmeans++, random_state=639)
Initial centroids chosen
Finding clusters... 3 clusters found
Iterations 19
Total distance calculations 5700000
Total sum of squared errors 3759631482485495.5
Time elapsed 0.198 seconds

KMeans(K=3, algo=elkan, metric=euclidean, max_iter=50, init=kmeans++, random_state=639)
Initial centroids chosen
Finding clusters... 3 clusters found
Iterations 19
Total distance calculations 570254
Total sum of squared errors 3759631482485495.5
Time elapsed 3.731 seconds

k = 3
iterations       19
standard         5700000
fast             570254
speedup          10.0
```
