import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_FILE = Path("outputs/bank_credit_card_customers_8000.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Reward-focused variables: these matter most for giving discounts/rewards.
FEATURES = ["Monthly_Spend", "Total_Transactions", "Offers_Redeemed"]


def load_data():
    df = pd.read_csv(DATA_FILE)
    required = ["Customer_ID", *FEATURES]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def plot_elbow_curve(X):
    inertias = []
    ks = list(range(1, 4))
    for k in ks:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o", linewidth=2, color="#1f77b4")
    plt.title("Elbow Curve for Reward Segmentation")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reward_elbow_curve.png", dpi=150)
    plt.close()

    # choose elbow based on the strongest drop point
    if len(inertias) >= 2:
        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        best_k = ks[max(range(len(diffs)), key=lambda i: diffs[i])] + 1
    else:
        best_k = ks[-1]

    return best_k, dict(zip(ks, inertias))


def segment_rewards(df, labels, cluster_centers):
    df = df.copy()
    df["Cluster"] = labels
    cluster_summary = (
        df.groupby("Cluster")[FEATURES]
        .mean()
        .sort_values(["Monthly_Spend", "Total_Transactions", "Offers_Redeemed"], ascending=False)
        .reset_index()
    )

    cluster_summary["Reward_Tier"] = ["Platinum", "Gold", "Silver"][: len(cluster_summary)]
    cluster_summary = cluster_summary.rename(columns={
        "Monthly_Spend": "Avg_Monthly_Spend",
        "Total_Transactions": "Avg_Total_Transactions",
        "Offers_Redeemed": "Avg_Offers_Redeemed",
    })

    return cluster_summary


def main():
    df = load_data()
    X = df[FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, inertia_by_k = plot_elbow_curve(X_scaled)
    print("Inertia by k:", inertia_by_k)
    print(f"Selected k from elbow curve: {best_k}")

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)

    cluster_summary = segment_rewards(df, labels, model.cluster_centers_)
    print("\nCluster summary based on reward value:")
    print(cluster_summary.to_string(index=False))

    df["Cluster"] = labels
    df["Reward_Tier"] = df["Cluster"].map({
        int(row["Cluster"]): row["Reward_Tier"]
        for _, row in cluster_summary.iterrows()
    })

    # Reorder tiers by reward strength
    tier_order = {"Platinum": 3, "Gold": 2, "Silver": 1}
    df["Reward_Priority"] = df["Reward_Tier"].map(tier_order)

    output_file = OUTPUT_DIR / "reward_segmented_customers.csv"
    df.sort_values(["Reward_Priority", "Monthly_Spend", "Total_Transactions"], ascending=[False, False, False]).to_csv(output_file, index=False)

    print(f"\nSaved segmented reward dataset: {output_file}")
    print("\nReward strategy logic:")
    print("- Platinum: highest spend + most transactions + strongest offer usage")
    print("- Gold: regular high-value users")
    print("- Silver: moderate-value customers")


if __name__ == "__main__":
    main()
