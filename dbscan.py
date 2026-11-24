#!/usr/bin/env python
"""From-scratch implementation of DBSCAN (Ester et al., 1996)."""

import numpy as np
import matplotlib.pyplot as plt


# Dataset 
# To run on a different dataset, replace this section so it defines:
#   X       : ndarray of shape (n, d), one observation per row
#   eps     : float, neighborhood radius
#   min_pts : int, minimum neighborhood size for a core point

def two_moons(n, n_outliers, seed):
    """Generate two interleaved half-moons with uniform background outliers."""
    np.random.seed(seed)
    t = np.pi * np.random.rand(n // 2)
    X = np.r_[np.c_[np.cos(t), np.sin(t)],
              np.c_[1 - np.cos(t), 0.5 - np.sin(t)]]
    X = X + 0.1 * np.random.randn(*X.shape)

    # Sample uniform outliers within a padded bounding box around the moons.
    lo = X.min(axis=0) - 0.5
    hi = X.max(axis=0) + 0.5
    outliers = lo + (hi - lo) * np.random.rand(n_outliers, 2)
    return np.r_[X, outliers]


X = two_moons(n=300, n_outliers=20, seed=0)
X = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardize features so eps is scale-consistent
print(f"{X.shape=}")

eps = 0.25
min_pts = 4


#  DBSCAN Algorithm 

UNCLASSIFIED = -2
NOISE = -1


def region_query(X, i, eps):
    """Return indices of all points in X within Euclidean distance eps of X[i] (inclusive)."""
    d = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
    return np.where(d <= eps)[0]


def expand_cluster(X, c, i, cluster_id, eps, min_pts):
    """Expand a new cluster from X[i] in place; return True if X[i] is a core point."""
    seeds = list(region_query(X, i, eps))
    if len(seeds) < min_pts:
        c[i] = NOISE  # May later be reassigned as a border point of another cluster
        return False

    c[seeds] = cluster_id
    seeds.remove(i)

    # Breadth-first expansion over all density-reachable points.
    while len(seeds) > 0:
        curr = seeds.pop(0)
        neighbors = region_query(X, curr, eps)
        if len(neighbors) >= min_pts:
            for q in neighbors:
                if c[q] == UNCLASSIFIED or c[q] == NOISE:
                    # Only unclassified points can be core points; prior NOISE points are border points.
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
