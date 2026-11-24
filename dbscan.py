#!/usr/bin/env python
"""From-scratch implementation of DBSCAN (Ester et al., 1996)."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# Dataset 
# To run on a different dataset, replace this section so it defines:
#   X       : ndarray of shape (n, d), one observation per row
#   eps     : float, neighborhood radius
#   min_pts : int, minimum neighborhood size for a core point

X, y = make_moons(n_samples = 300, noise = 0.08, random_state = 42)
print(f"{X.shape=}")

eps = 0.25
min_pts = 4


#  DBSCAN Algorithm 

UNCLASSIFIED = -2 #for points that have not been assigned a cluster
NOISE = -1


def region_query(X, point, eps):
    """Return indices of all points in X within Euclidean distance eps of X[point] (inclusive)."""
    d = np.sqrt(np.sum((X - X[point]) ** 2, axis=1))
    return np.where(d <= eps)[0]


def expand_cluster(X, c, point, cluster_id, eps, min_pts):
    """Expand a new cluster from X[point] in place; return True if X[point] is a core point."""
    seeds = list(region_query(X, point, eps)) #list of given point's neighbors

    if len(seeds) < min_pts:
        c[point] = NOISE  # May later be reassigned as a border point of another cluster
        return False

    c[seeds] = cluster_id
    seeds.remove(point)

    # Breadth-first expansion over all density-reachable points.
    while len(seeds) > 0:
        curr = seeds.pop(0)
        neighbors = region_query(X, curr, eps)
        if len(neighbors) >= min_pts:
            for q in neighbors:
                if c[q] == UNCLASSIFIED or c[q] == NOISE:
                    # NOISE points already failed the core test and become border points
                    if c[q] == UNCLASSIFIED:
                        seeds.append(q)
                    c[q] = cluster_id

    return True


def dbscan(X, eps, min_pts):
    """Cluster rows of X with DBSCAN; returns labels in {0, 1, ...} or -1 for noise."""
    n = X.shape[0]
    c = np.full(n, UNCLASSIFIED, dtype=int)
    cluster_id = 0

    for i in range(n):
        if c[i] == UNCLASSIFIED:
            if expand_cluster(X, c, i, cluster_id, eps, min_pts):
                cluster_id += 1

    return c


# Execution & Visualization 

c = dbscan(X, eps=eps, min_pts=min_pts)

n_clusters = c.max() + 1
n_noise = np.sum(c == NOISE)
print(f"clusters found: {n_clusters}, noise points: {n_noise}")

colors = np.array(["tab:blue", "tab:orange", "tab:green", "tab:red",
                   "tab:purple", "tab:brown", "tab:pink", "tab:olive"])

fig, ax = plt.subplots(figsize=(4.8, 4.2))
ax.scatter(X[c == NOISE, 0], X[c == NOISE, 1], s=8, c="lightgray", label="noise")
for j in range(n_clusters):
    ax.scatter(X[c == j, 0], X[c == j, 1], s=8,
               c=colors[j % len(colors)], label=f"cluster {j}")

ax.set_title(f"DBSCAN (eps={eps}, min_pts={min_pts})", fontsize=10)
ax.set_aspect("equal")
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08),
          ncol=n_clusters + 1, frameon=False)
fig.tight_layout()

plt.savefig("dbscan_result.pdf")
