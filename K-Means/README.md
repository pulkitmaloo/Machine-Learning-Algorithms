### K-Means


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
