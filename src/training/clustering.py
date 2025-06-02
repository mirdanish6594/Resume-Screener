# src/training/cluster.py
from sklearn.cluster import KMeans
import joblib

def cluster_resumes(vectors, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(vectors)
    return model, labels

def assign_clusters(resumes, labels):
    for i, r in enumerate(resumes):
        r["cluster"] = int(labels[i])
    return resumes
