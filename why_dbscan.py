"""
why_dbscan.py
=============
A TINY visual demo of WHERE K-Means & Hierarchical clustering FAIL,
and why DBSCAN is the fix -- all in one picture.

The idea:
    * K-Means and Ward-Hierarchical assume clusters are ROUND, similar-sized blobs.
      They draw straight-ish boundaries, so they cut awkward shapes in half.
    * DBSCAN groups by DENSITY, so it follows any shape and also spots outliers.

We test on two classic "awkward" shapes that are NOT round blobs:
    1. Two moons   (two interlocking crescents)
    2. Two circles (one ring inside another)

Run it:
    python why_dbscan.py
It saves one comparison image: outputs/03_why_dbscan.png
"""

# ------------------------------------------------------------------
# 0) IMPORTS
# ------------------------------------------------------------------
import matplotlib                        # the plotting library
matplotlib.use("Agg")                    # save to a file (no pop-up window needed)
import matplotlib.pyplot as plt          # the drawing part
from pathlib import Path                 # safe file paths

from sklearn.datasets import make_moons, make_circles          # ready-made shapes
from sklearn.preprocessing import StandardScaler               # put x and y on same scale
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

RANDOM_STATE = 42                        # fixed seed -> the same picture every run
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1) MAKE THE AWKWARD SHAPES  (each has 2 true groups)
# ------------------------------------------------------------------
# make_moons / make_circles return (points, true_labels). We ignore the labels
# (clustering is unsupervised) and only use them to know the "right" answer is 2 groups.
moons_X,   _ = make_moons(n_samples=300, noise=0.06, random_state=RANDOM_STATE)
circles_X, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=RANDOM_STATE)

# Always scale before distance-based models (fair footing for x and y).
moons_X   = StandardScaler().fit_transform(moons_X)
circles_X = StandardScaler().fit_transform(circles_X)

# ------------------------------------------------------------------
# 2) THE THREE MODELS WE COMPARE
#    KMeans / Agglomerative are told there are 2 clusters (their best case!).
#    DBSCAN is NOT told the count -- it discovers it from density.
#    (eps was picked from these scaled shapes; min_samples=5 is a common default.)
# ------------------------------------------------------------------
def models_for(eps):
    return [
        ("K-Means",      KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)),
        ("Hierarchical", AgglomerativeClustering(n_clusters=2, linkage="ward")),
        ("DBSCAN",       DBSCAN(eps=eps, min_samples=5)),
    ]

# Each row = one shape, with the eps that suits it (found by trying a few values).
rows = [("Two moons", moons_X, 0.30),
        ("Two circles", circles_X, 0.35)]

# ------------------------------------------------------------------
# 3) DRAW A GRID:  rows = shapes,  columns = the 3 models
# ------------------------------------------------------------------
fig, axes = plt.subplots(len(rows), 3, figsize=(11, 7))

for r, (shape_name, X, eps) in enumerate(rows):
    for c, (model_name, model) in enumerate(models_for(eps)):
        labels = model.fit_predict(X)            # cluster labels for every point
        ax = axes[r][c]
        # Colour points by their predicted cluster; DBSCAN noise (-1) shows as grey.
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=12)
        n_noise = (labels == -1).sum()           # DBSCAN may flag outliers as -1
        title = f"{model_name}"
        if model_name == "DBSCAN" and n_noise:
            title += f"  ({n_noise} noise)"
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])     # coordinates are not important here
        if c == 0:
            ax.set_ylabel(shape_name, fontsize=12, fontweight="bold")

fig.suptitle("K-Means & Hierarchical cut the shapes in half -- DBSCAN follows them",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "03_why_dbscan.png", dpi=120)
plt.close(fig)
print("Saved comparison -> outputs/03_why_dbscan.png")
print("Look: K-Means/Hierarchical split each shape with a straight cut;")
print("DBSCAN recovers the true moons/rings because it groups by density.")
