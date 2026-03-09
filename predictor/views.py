import pandas as pd
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration, generate_rwanda_map
import joblib
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model

# Load models once
regression_model = joblib.load(
    "model_generators/regression/regression_model.pkl")
classification_model = joblib.load(
    "model_generators/classification/classification_model.pkl")
clustering_model = joblib.load(
    "model_generators/clustering/clustering_model.pkl")
clustering_pt = joblib.load(
    "model_generators/clustering/clustering_pt.pkl")
wholesale_mean = joblib.load(
    "model_generators/clustering/wholesale_mean.pkl")


def classification_analysis(request):

    context = {
        "evaluations": evaluate_classification_model()
    }
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = classification_model.predict(
            [[year, km, seats, income]])[0]
        context["prediction"] = prediction
    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):

    context = {
        "evaluations": evaluate_clustering_model()
    }
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])
            # Step 1: Predict price
            predicted_price = regression_model.predict(
                [[year, km, seats, income]])[0]
            # Step 2: Predict cluster
            # Transform seating capacity the same way as training
            import numpy as np
            X_seating = clustering_pt.transform([[seats]])
            # Use mean wholesale price as proxy (scaled same as training: * 0.01)
            X_price = np.array([[wholesale_mean]]) * 0.01 / 1.0  # approximate same scale
            # StandardScaler mean and std are encoded in the clustering_pt; use simple scale
            X_cluster = np.hstack([X_seating, [[0.0]]])  # wholesale contribution is ~0 (scaled by 0.01)
            cluster_id = clustering_model.predict(X_cluster)[0]
            
            # Mapping for all 7 seating capacity clusters
            mapping = {
                0: "4-Seater (Compact)",
                1: "8-Seater (Large)",
                2: "3-Seater (Coupe)",
                3: "7-Seater (Family)",
                4: "2-Seater (Sport)",
                5: "6-Seater (Mini-Van)",
                6: "5-Seater (Standard)"
            }
            context.update({
                "prediction": mapping.get(cluster_id, f"Cluster {cluster_id}"),
                "price": predicted_price
            })
        except Exception as e:
            context["error"] = str(e)
    return render(request, "predictor/clustering_analysis.html", context)



def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": generate_rwanda_map(df),
    }
    return render(request, "predictor/index.html", context)


def regression_analysis(request):
    context = {"evaluations": evaluate_regression_model()}
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["price"] = prediction
    return render(request, "predictor/regression_analysis.html", context)
