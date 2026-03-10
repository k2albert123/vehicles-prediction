import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

# 1. Understanding What They Want
SEGMENT_FEATURES = ["estimated_income", "selling_price"]

# Method 2 - Standardize Features (VERY IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[SEGMENT_FEATURES])

# 5 clusters: Economy, Standard, Premium, Extra Premium, Luxury
kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
df["cluster_id"] = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_

# Save model and scaler
joblib.dump(kmeans, "model_generators/clustering/clustering_model.pkl")
joblib.dump(scaler, "model_generators/clustering/clustering_pt.pkl")

silhouette_avg = round(silhouette_score(X_scaled, df["cluster_id"]), 3)

# Map labels based on income scale to match screenshot labels
sorted_clusters = centers[:, 0].argsort()
cluster_mapping = {
    sorted_clusters[0]: "Economy",
    sorted_clusters[1]: "Standard",
    sorted_clusters[2]: "Premium",
    sorted_clusters[3]: "Extra Premium",
    sorted_clusters[4]: "Luxury",
}
df["client_class"] = df["cluster_id"].map(cluster_mapping)

# 2. Add CV Calculation
# Calculate CV for each cluster
cv_table_data = df.groupby("cluster_id")[SEGMENT_FEATURES].agg(["mean", "std"])

# Calculate coefficient of variation
cv_table_data["income_cv%"] = (cv_table_data[("estimated_income", "std")] / cv_table_data[("estimated_income", "mean")]) * 100
cv_table_data["price_cv%"] = (cv_table_data[("selling_price", "std")] / cv_table_data[("selling_price", "mean")]) * 100

cv_table = cv_table_data[["income_cv%", "price_cv%"]].round(2).reset_index()

comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]

# 3. Display CV in Django Return Dictionary
def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "summary": cv_table_data.round(2).reset_index().to_html(
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
        "cv_table": cv_table.to_html(
            classes="table table-bordered table-striped table-sm text-center",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }


if __name__ == "__main__":
    pass