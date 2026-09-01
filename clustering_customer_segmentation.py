"""
clustering_customer_segmentation.py
===================================
A beginner-friendly, end-to-end clustering walkthrough for absolute freshers.

REAL-WORLD USE CASE: BANK CREDIT-CARD CUSTOMER SEGMENTATION
    A bank has 6 months of usage data for ~8,950 active credit-card holders, but
    NO ready-made customer "types" (no labels). Marketing wants to split holders
    into a few clear groups to design targeted strategy -- e.g. reward big
    spenders, watch cash-advance-heavy (higher-risk) customers, and re-engage
    dormant cards. That is a textbook CLUSTERING problem, and this script solves it
    on the well-known public "CC GENERAL" credit-card dataset (saved as
    bank_customers.csv). We cluster on three easy-to-read behaviours:
        BALANCE       -- average balance the customer carries
        PURCHASES     -- total amount spent on purchases
        CASH_ADVANCE  -- cash withdrawn against the card (a risk signal)

WHAT THIS SCRIPT DOES (all on ONE real dataset so the comparison is fair):
    1. Loads the REAL bank_customers.csv (no synthetic data generation).
    2. K-Means      -> draws an ELBOW curve to help pick the number of clusters (k).
    3. Hierarchical -> draws a DENDROGRAM to help pick the number of clusters.
    4. Shows the DRAWBACKS of K-Means and Hierarchical (round clusters, forced grouping).
    5. DBSCAN       -> fixes those drawbacks (any shape + labels outliers as "noise").
    6. GRID SEARCH  -> tries many hyperparameters automatically to tune DBSCAN.
    7. EVALUATION   -> compares every model with 3 metrics to see which is best.
    8. Saves the winning model safely, ready for the Streamlit web app (app.py).

HOW TO RUN:
    pip install -r requirements.txt
    python clustering_customer_segmentation.py

All charts are saved as .png files inside the "outputs" folder.
"""

# =====================================================================
# 0) IMPORTS  --  load the tools we need (one clear comment per line)
# =====================================================================
import numpy as np                       # fast maths on arrays of numbers
import pandas as pd                      # tables (called DataFrames), like Excel in Python
import matplotlib                        # the plotting library
matplotlib.use("Agg")                    # "Agg" = save charts to files (no pop-up window)
import matplotlib.pyplot as plt          # the part of matplotlib we draw with

from sklearn.preprocessing import StandardScaler         # puts every column on the same scale
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN  # the 3 clustering models
from sklearn.metrics import (
    silhouette_score,                    # score: -1..+1, HIGHER is better (needs NO labels)
    davies_bouldin_score,                # score:  0..inf, LOWER  is better
    calinski_harabasz_score,             # score:  0..inf, HIGHER is better
)
from scipy.cluster.hierarchy import dendrogram, linkage  # tools that build the dendrogram picture
import joblib                            # saves/loads a trained model to a file
from pathlib import Path                 # safe, OS-independent way to handle file paths

RANDOM_STATE = 42                        # a fixed "seed" so every run gives the SAME result
OUT = Path("outputs")                    # one tidy folder to hold the CSV, charts + model
OUT.mkdir(exist_ok=True)                 # create the folder if it does not exist yet

DATA_FILE = OUT / "bank_customers.csv"   # the REAL dataset we load (~8,950 card holders)
# We cluster on three easy-to-read spending behaviours. Other columns (credit limit,
# payments, tenure, ...) stay in the file and can be used later to PROFILE groups.
FEATURES = ["BALANCE", "PURCHASES", "CASH_ADVANCE"]   # columns we cluster on (in ONE place)


# =====================================================================
# 1) BUSINESS + DATA (CRISP-ML(Q) Phase 1: Business & Data Understanding)
#    Load the REAL bank_customers.csv. No synthetic generation.
#    The file has 18 columns; we keep the 3 behaviour columns we cluster on.
# =====================================================================
def load_customers():
    """Load the real bank credit-card CSV and return a clean, numeric table."""
    df = pd.read_csv(DATA_FILE)                     # read the CSV file into a DataFrame
    df = df[FEATURES]                               # keep ONLY the columns we cluster on
    df = df.dropna()                                # drop rows with missing values, if any
    return df


# =====================================================================
# 1b) WHY ONLY 3 FEATURES?  (feature selection, shown with evidence)
#     The file has 17 usable columns, but MORE is not better for clustering:
#       * many columns are REDUNDANT (they measure almost the same thing), and
#       * clusters built on 17 mixed columns are very hard to explain/name.
#     We keep three LOW-OVERLAP, business-meaningful behaviours. This function
#     prints a correlation table so you can SEE the redundancy for yourself.
# =====================================================================
def explain_feature_choice():
    """Print evidence for why we cluster on just BALANCE, PURCHASES, CASH_ADVANCE."""
    full = pd.read_csv(DATA_FILE)                   # the whole file (all columns)
    print(f"Columns available: {full.shape[1] - 1} (ignoring CUST_ID)")
    # Show how strongly some columns overlap with PURCHASES (1.0 = identical info).
    overlap = ["PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
               "PURCHASES_TRX", "PAYMENTS"]
    print("How much do these repeat PURCHASES? (correlation, >0.8 = redundant)")
    print(full[overlap].corr()["PURCHASES"].round(2).to_string())
    # Show that OUR three chosen features are NOT redundant with each other.
    print("\nOur 3 chosen features barely overlap (each adds new information):")
    print(full[FEATURES].corr().round(2).to_string())
    print("=> keep BALANCE (money carried), PURCHASES (spending), "
          "CASH_ADVANCE (cash pulled = risk).\n")


# =====================================================================
# 2) DATA PREP (Phase 2): LOG-TRANSFORM, THEN SCALE
#    Money columns are very "skewed": a few customers spend HUGE amounts, most
#    spend little. log1p(x) = log(1 + x) squashes that long tail so clusters are
#    about behaviour, not just a handful of big spenders. Then StandardScaler puts
#    every column on the same footing (mean=0, std=1) so distance is fair.
# =====================================================================
def scale_features(df):
    """Return a scaler (to reuse later) and the log-then-scaled data array."""
    X_log = np.log1p(df[FEATURES].to_numpy())       # log1p handles zeros safely (log of 1+x)
    scaler = StandardScaler()                       # create the scaler
    X = scaler.fit_transform(X_log)                 # LEARN the scale AND apply it in one step
    return scaler, X


# =====================================================================
# 3) K-MEANS + THE ELBOW CURVE
#    K-Means needs YOU to choose k (the number of clusters) up front.
#    The elbow method: plot "inertia" for many k values and look for the
#    bend ("elbow") where adding more clusters stops helping much.
#
#    K-MEANS HYPERPARAMETERS (the knobs you can turn):
#      n_clusters   -> how many clusters to make (the main choice).
#      n_init       -> how many random restarts; keeps the best (avoids bad luck).
#      init         -> "k-means++" picks smart starting points (default, recommended).
#      max_iter     -> max refinement steps per run (300 is plenty here).
#      random_state -> fixes randomness so results are reproducible.
# =====================================================================
def kmeans_elbow(X, k_range=range(2, 9)):
    """Try several k values; save an elbow + silhouette chart; return the best k."""
    inertias, silhouettes = [], []                  # empty lists to collect our scores
    for k in k_range:                               # test each candidate number of clusters
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)                  # train the model AND get cluster labels
        inertias.append(km.inertia_)                # inertia = tightness (lower = tighter)
        silhouettes.append(silhouette_score(X, labels))  # silhouette (higher = better)

    ks = list(k_range)
    # --- Draw two mini-charts side by side: the elbow, and the silhouette ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ks, inertias, "o-", color="#0080a8")   # the ELBOW curve
    ax1.set_title("Elbow method (look for the bend)")
    ax1.set_xlabel("k (number of clusters)"); ax1.set_ylabel("Inertia (lower = tighter)")
    ax2.plot(ks, silhouettes, "o-", color="#f38020")  # the SILHOUETTE curve
    ax2.set_title("Silhouette (higher = better)")
    ax2.set_xlabel("k (number of clusters)"); ax2.set_ylabel("Silhouette score")
    fig.tight_layout(); fig.savefig(OUT / "01_elbow_silhouette.png", dpi=120); plt.close(fig)

    best_k = ks[int(np.argmax(silhouettes))]        # the k with the highest silhouette score
    print(f"[K-Means] silhouette by k: {dict(zip(ks, np.round(silhouettes, 3)))}")
    print(f"[K-Means] chosen k = {best_k}")
    return best_k


# =====================================================================
# 4) HIERARCHICAL CLUSTERING + DENDROGRAM
#    Hierarchical clustering merges the closest points step by step into a
#    tree. A DENDROGRAM shows that tree. Cut the tree where the vertical
#    lines are longest to read off a sensible number of clusters.
#
#    HYPERPARAMETERS:
#      linkage="ward" -> merge groups so clusters stay tight (a safe default).
#      n_clusters     -> where to "cut" the tree (how many groups to keep).
# =====================================================================
def plot_dendrogram(X, sample_size=800):
    """Save a dendrogram picture so we can visually choose the number of clusters."""
    # A dendrogram on ~9,000 points is slow to build and unreadable, so we draw it
    # on a random SAMPLE. The sample only shapes the picture, not the final model.
    rng = np.random.default_rng(RANDOM_STATE)       # seeded -> same sample every run
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    Z = linkage(X[idx], method="ward")              # build the merge tree ("ward" = tight groups)
    fig, ax = plt.subplots(figsize=(11, 4))
    dendrogram(Z, truncate_mode="lastp", p=12, ax=ax)  # show only the top of the tree (readable)
    ax.set_title("Dendrogram on a sample (cut across the tallest gap to choose #clusters)")
    ax.set_xlabel("Customers (grouped)"); ax.set_ylabel("Distance when merged")
    fig.tight_layout(); fig.savefig(OUT / "02_dendrogram.png", dpi=120); plt.close(fig)
    print("[Hierarchical] dendrogram saved -> outputs/02_dendrogram.png")


# =====================================================================
# 5) DBSCAN  --  how it FIXES the drawbacks of K-Means / Hierarchical
#    Drawbacks it fixes:
#      * You do NOT tell it how many clusters to make (it discovers them).
#      * It finds clusters of ANY shape (not just round blobs).
#      * It labels weird points as NOISE (label = -1) instead of forcing them in.
#
#    HYPERPARAMETERS (the two that matter most):
#      eps          -> the neighbourhood radius. Points within eps are "neighbours".
#                      Too small -> everything is noise; too big -> everything is one cluster.
#      min_samples  -> how many neighbours a point needs to be a dense "core" point.
# =====================================================================
def grid_search_dbscan(X, max_noise_frac=0.25):
    """Try many (eps, min_samples) combinations and keep the best silhouette."""
    best = {"score": -1}                            # start with an impossible-to-beat-low score
    eps_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]     # candidate radii to test
    min_samples_values = [5, 10, 15, 20]            # candidate density thresholds to test

    for eps in eps_values:                          # loop over every radius...
        for min_samples in min_samples_values:      # ...and every density threshold
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)              # -1 in labels means "noise/outlier"
            mask = labels != -1                     # ignore noise points when scoring
            n_clusters = len(set(labels[mask]))     # how many real clusters were found
            if n_clusters < 2:                      # need at least 2 clusters to score fairly
                continue                            # skip useless settings
            # GUARD: silhouette rises when we dump points as noise, so reject settings
            # that call more than max_noise_frac of the data "noise" (a degenerate win).
            if (labels == -1).mean() > max_noise_frac:
                continue
            score = silhouette_score(X[mask], labels[mask])  # grade this setting
            if score > best["score"]:               # did we beat the best so far?
                best = {"score": score, "eps": eps, "min_samples": min_samples,
                        "labels": labels, "n_clusters": n_clusters,
                        "n_noise": int((labels == -1).sum())}
    print(f"[DBSCAN] best params: eps={best['eps']}, min_samples={best['min_samples']} "
          f"-> {best['n_clusters']} clusters, {best['n_noise']} noise points")
    return best


# =====================================================================
# 6) EVALUATION  --  compare every model with 3 metrics to see which is best
#    We use INTERNAL metrics (no true labels needed), which is realistic
#    because in real clustering you usually do NOT have the answers.
# =====================================================================
def evaluate(name, X, labels):
    """Return a small results row for one model. Skips 'noise' (-1) points fairly."""
    mask = labels != -1                             # keep only points that got a real cluster
    n_clusters = len(set(labels[mask]))             # count the real clusters
    if n_clusters < 2:                              # metrics need at least 2 clusters
        return {"model": name, "clusters": n_clusters, "note": "too few clusters to score"}
    return {
        "model": name,
        "clusters": n_clusters,
        "noise_pts": int((labels == -1).sum()),
        "silhouette":     round(silhouette_score(X[mask], labels[mask]), 3),   # higher = better
        "davies_bouldin": round(davies_bouldin_score(X[mask], labels[mask]), 3),  # lower = better
        "calinski_h":     round(calinski_harabasz_score(X[mask], labels[mask]), 1),  # higher = better
    }


# =====================================================================
# 6b) NAMING THE CLUSTERS  --  how we turn 0,1,2,3 into business names
#     K-Means only gives numbers. WE give meaning, in three clear steps:
#       1. average each cluster's behaviour (in real money units),
#       2. tag each number HIGH / MID / LOW versus the overall average,
#       3. read the HIGH/LOW pattern and give the group a plain-English name.
# =====================================================================
def level(value, avg):
    """Step 2: is this value HIGH, LOW, or about average (MID)?"""
    if value > 1.3 * avg:                           # clearly above the typical customer
        return "HIGH"
    if value < 0.7 * avg:                           # clearly below the typical customer
        return "LOW"
    return "MID"

def name_from_pattern(bal, pur, cash):
    """Step 3: explicit, explainable rules that map a pattern to a name."""
    if bal == "LOW" and pur == "LOW" and cash == "LOW":
        return "Low-Activity / Dormant"             # barely uses the card
    if cash == "HIGH" and pur == "LOW":
        return "Cash-Advance Users (higher risk)"   # pulls cash, does not shop
    if pur == "HIGH" and cash != "HIGH":
        return "Big Spenders"                       # shops a lot, avoids cash advances
    if bal == "HIGH" and cash == "HIGH":
        return "Heavy Users"                        # high on everything
    return "Mixed / Review"                         # does not fit a clean rule

def name_segments(df, labels):
    """Run the 3 steps and return a {cluster_id: name} dictionary."""
    tmp = df.copy(); tmp["segment"] = labels
    overall = tmp[FEATURES].mean()                  # the 'typical' customer
    profile = tmp.groupby("segment")[FEATURES].mean()

    print("\n=== Step 1: average behaviour per segment (real money units) ===")
    print(profile.round(0).astype(int).to_string())
    print("overall average:", overall.round(0).astype(int).to_dict())

    print("\n=== Step 2 + 3: HIGH/MID/LOW vs overall  ->  a name ===")
    names = {}
    for seg, row in profile.iterrows():
        bal, pur, cash = (level(row[f], overall[f]) for f in FEATURES)
        names[int(seg)] = name_from_pattern(bal, pur, cash)
        print(f"  segment {seg}: BALANCE={bal:<4} PURCHASES={pur:<4} "
              f"CASH_ADVANCE={cash:<4} -> {names[int(seg)]}")
    return names


# =====================================================================
# 7) MAIN  --  run the whole story in order
# =====================================================================
def main():
    # ---- Phase 1: load the REAL data + justify the feature choice ----
    df = load_customers()
    print("Data shape:", df.shape)                  # (8949, 3)
    print(df.describe().round(0), "\n")             # quick sanity check before modelling
    explain_feature_choice()                        # WHY only these 3 columns (with evidence)

    # ---- Phase 2: scaling ----
    scaler, X = scale_features(df)

    # ---- K-Means: pick k with the elbow, then fit ----
    best_k = kmeans_elbow(X)
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    km_labels = kmeans.fit_predict(X)

    # ---- Hierarchical: draw the dendrogram, then fit with the same k ----
    plot_dendrogram(X)
    hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_labels = hc.fit_predict(X)

    # ---- DBSCAN: grid-search the best (eps, min_samples) ----
    db_best = grid_search_dbscan(X)
    db_labels = db_best["labels"]

    # ---- Compare all three models on the same data ----
    results = pd.DataFrame([
        evaluate("K-Means",      X, km_labels),
        evaluate("Hierarchical", X, hc_labels),
        evaluate("DBSCAN",       X, db_labels),
    ])
    print("\n=== Model comparison (internal metrics) ===")
    print(results.to_string(index=False))

    # Pick the winner by silhouette (higher is better).
    winner = results.sort_values("silhouette", ascending=False).iloc[0]["model"]
    print(f"\nBest by silhouette: {winner}")

    # ---- Phase 4b: NAME the K-Means segments by analysing their behaviour ----
    names = name_segments(df, km_labels)            # analysis -> business names
    sizes = pd.Series(km_labels).value_counts().sort_index().to_dict()
    print("segment sizes:", sizes)

    # ---- Phase 5: Save the K-Means bundle for the web app ----
    # We deploy K-Means because it can .predict() brand-new customers directly.
    # IMPORTANT: the app must apply the SAME log1p step before this scaler.
    # We also save the NAMES we just derived, so the app shows them (not just ids).
    joblib.dump({"scaler": scaler, "model": kmeans, "features": FEATURES,
                 "log_transform": True, "names": names},
                OUT / "segmenter.joblib")
    print("\nSaved trained pipeline (with segment names) -> outputs/segmenter.joblib")


# Only run main() when this file is executed directly (not when imported).
if __name__ == "__main__":
    main()
