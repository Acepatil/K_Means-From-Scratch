
# K-Means Clustering from Scratch

## Introduction

This project implements the K-Means clustering algorithm from scratch in Python, inspired by the paper "Automated Variable Weighting in k-means Type Clustering" by Huang et al. The implementation includes functions for centroid initialization using the k-means++ method, distance calculation, cluster assignment, centroid updates, and visualization of the clustering process.

## Files

- `k_means.py`: Contains the implementation of the K-Means algorithm and supporting functions.
- `README.md`: Documentation of the project and explanation of the implementation.
- `LICENSE.md`: The MIT License for the project.

## Installation

To run this code, you'll need Python 3 and the following libraries:

- NumPy
- Pandas
- Matplotlib

You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib
```

## Usage

The main function to use is `K_Means`, which performs the K-Means clustering on a given dataset.

### Example

```python
import numpy as np
from k_means import K_Means

# Create a sample dataset
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

n, m = X.shape
k = 2
max_iter = 100

# Run K-Means
centroids, U, iterations = K_Means(n, k, m, X, max_iter)

print("Centroids:\n", centroids)
print("Iterations:", iterations)
```

## Functions

##### `init_centroids_kmeanspp(X, k)`

Initializes centroids using the k-means++ method to ensure better clustering results.

##### `dist(X, Z, i, j, l)`

Calculates the squared distance between a data point and a centroid.

##### `dist_l(X, Z, i, l, m)`

Calculates the total distance between a data point and a centroid across all dimensions.

##### `modify_D(n, k, m, X, Z, D)`

Updates the distance matrix `D` with the distances between all data points and centroids.

##### `find_min(D, i, k)`

Finds the index of the centroid with the minimum distance to a data point.

##### `modify_U(n, k, U, D)`

Updates the cluster assignment matrix `U` based on the current distances.

##### `modify_Z(n, k, m, X, Z, U)`

Updates the centroids `Z` based on the current cluster assignments.

##### `show_clusters(X, cluster, cg)`

Visualizes the current state of the clusters and centroids.

##### `K_Means(n, k, m, X, max_iter)`

Main function to perform K-Means clustering. It initializes the centroids, iteratively updates the assignments and centroids, and checks for convergence.

## Reference

The implementation of the K-Means algorithm in this project is inspired by the following paper:

Huang JZ, Ng MK, Rong H, Li Z. Automated variable weighting in k-means type clustering. IEEE Trans Pattern Anal Mach Intell. 2005 May;27(5):657-68. doi: 10.1109/TPAMI.2005.95. PMID: 15875789.

## License

This project is licensed under the MIT License.

