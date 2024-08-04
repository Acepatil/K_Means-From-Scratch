import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

def init_centroids_kmeanspp(X, k):
    n = X.shape[0]
    centroids = np.zeros((k, X.shape[1]))
    centroids[0] = X[np.random.randint(0, n)]
    for i in range(1, k):
        distances = np.min([np.linalg.norm(X - centroids[j], axis=1)**2 for j in range(i)], axis=0)
        prob = distances / distances.sum()
        centroids[i] = X[np.random.choice(n, p=prob)]
    
    return centroids

def dist(X, Z, i, j, l):
    return (X[i, j] - Z[l, j])**2

def dist_l(X, Z, i, l, m):
    dist_m = 0
    for j in range(m):
        dist_m += dist(X, Z, i, j, l)
    return dist_m

def modify_D(n, k, m, X, Z, D):
    for i in range(n):
        for l in range(k):
            D[i, l] = dist_l(X, Z, i, l, m)

def find_min(D, i, k):
    min_index = 0
    min_value = D[i, 0]
    for l in range(1, k):
        if min_value > D[i, l]:
            min_value = D[i, l]
            min_index = l
    return min_index

def modify_U(n, k, U, D):
    for i in range(n):
        min_dist = find_min(D, i, k)
        for l in range(k):
            U[i, l] = 1 if l == min_dist else 0

def modify_Z(n, k, m, X, Z, U):
    for l in range(k):
        for j in range(m):
            num = 0
            den = 0
            for i in range(n):
                num += U[i, l] * X[i, j]
                den += U[i, l]
            if den != 0:
                Z[l, j] = num / den

def show_clusters(X, cluster, cg):
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=cluster))
    colors = {0:'blue', 1:'orange', 2:'green', 3:'yellow', 4:'black', 5:'pink', 6:'purple'}
    fig, ax = plt.subplots(figsize=(8, 8))
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    ax.scatter(cg[:, 0], cg[:, 1], marker='*', s=150, c='#ff2222')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()

def K_Means(n, k, m, X, max_iter):
    # Random Initialization of centroids Z
    Z = init_centroids_kmeanspp(X, k)
    # Initialize U and D
    U = np.zeros((n, k))
    D = np.zeros((n, k))
    # Initial distance and assignment update
    modify_D(n, k, m, X, Z, D)
    modify_U(n, k, U, D)
    i = 0
    while i < max_iter:
        # Save the current U for convergence check
        previous_U = U.copy()
        # Update centroids based on current assignments
        modify_Z(n, k, m, X, Z, U)
        # Compute new distances and update assignments
        labels = np.argmax(U, axis=1)
        show_clusters(X,labels,Z)
        modify_D(n, k, m, X, Z, D)
        modify_U(n, k, U, D)
        # Check for convergence
        if np.all(previous_U == U):
            break
        i += 1
    return Z, U, i

