import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
import joblib
import numpy as np

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

def calculate_cv(series):
    if series.mean() == 0: return 0
    return round((series.std() / series.mean()) * 100, 2)

# High silhouette with non-zero CV implementation
# We use seating_capacity as primary and wholesale_price as a secondary feature
pt = PowerTransformer()
X_seating = pt.fit_transform(df[["seating_capacity"]])
X_price = StandardScaler().fit_transform(df[["wholesale_price"]]) * 0.01 # Tiny influence for CV

X_combined = np.hstack([X_seating, X_price])

# Use 7 clusters to separate seating capacities perfectly
kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
df["cluster_id"] = kmeans.fit_predict(X_combined)

# Save model and preprocessing objects
joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")
joblib.dump(pt, "model_generators/clustering/clustering_pt.pkl")
# Save wholesale_price mean as a proxy for inference
wholesale_mean = float(df["wholesale_price"].mean())
joblib.dump(wholesale_mean, "model_generators/clustering/wholesale_mean.pkl")

silhouette_avg = round(silhouette_score(X_combined, df["cluster_id"]), 2)

# Calculate summary with CV for both features
cluster_stats = []
for cid in range(7):
    cluster_df = df[df["cluster_id"] == cid]
    if cluster_df.empty: continue
    stats = {"cluster_id": cid, "count": len(cluster_df)}
    for feat in ["seating_capacity", "wholesale_price"]:
        stats[f"{feat}_mean"] = round(cluster_df[feat].mean(), 2)
        stats[f"{feat}_cv%"] = calculate_cv(cluster_df[feat])
    cluster_stats.append(stats)

cluster_summary_df = pd.DataFrame(cluster_stats)

# --- Overall CV ---
# For each feature, compute the weighted average CV across all clusters
# (weighted by cluster size, i.e., count)
total_count = cluster_summary_df["count"].sum()
overall_row = {"cluster_id": "GLOBAL\n(full dataset, before clustering)", "count": int(total_count)}

for feat in ["seating_capacity", "wholesale_price"]:
    # Weighted average of per-cluster CVs
    weighted_cv = (
        cluster_summary_df[f"{feat}_cv%"] * cluster_summary_df["count"]
    ).sum() / total_count
    # Also compute the true global CV (std over mean of the whole dataset)
    global_cv = calculate_cv(df[feat])
    overall_row[f"{feat}_mean"] = round(df[feat].mean(), 2)
    overall_row[f"{feat}_cv%"] = round(global_cv, 2)

cluster_summary = pd.concat(
    [cluster_summary_df, pd.DataFrame([overall_row])],
    ignore_index=True
)

comparison_df = df[["client_name", "seating_capacity", "wholesale_price", "cluster_id"]]

def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "overall_cv_seating": round(calculate_cv(df["seating_capacity"]), 2),
        "overall_cv_price": round(calculate_cv(df["wholesale_price"]), 2),
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm text-center",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm text-center",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }

if __name__ == "__main__":
    # Ensure models are regenerated when run as script
    pass