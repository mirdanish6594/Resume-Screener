# cluster_pipeline.py

import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from clustering import cluster_resumes, assign_clusters

# Load cleaned resume data
PROCESSED_JSON_PATH = "data/processed/cleaned_resumes.json"
CLUSTERED_OUTPUT_PATH = "outputs/clustered_resumes.json"

def load_cleaned_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_clustered_output(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    resumes = load_cleaned_data(PROCESSED_JSON_PATH)

    # Vectorize using TF-IDF on cleaned text
    texts = [r["cleaned_text"] for r in resumes]
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors = vectorizer.fit_transform(texts)

    # Cluster resumes
    model, labels = cluster_resumes(vectors)
    clustered_resumes = assign_clusters(resumes, labels)

    # Save output
    save_clustered_output(clustered_resumes, CLUSTERED_OUTPUT_PATH)
    print(f"✅ Clustered resumes saved to {CLUSTERED_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
