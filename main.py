from src.ingestion.loader import load_resumes
from src.preprocessing.spacy_cleaner import spacy_clean_text
from src.preprocessing.cleaner import extract_skills
from src.utils.text_extraction import extract_text
from src.training.clustering import cluster_resumes, assign_clusters
from src.preprocessing.vectorizer import build_vectorizer, save_vectorizer

import os
import json

RAW_RESUME_DIR = "data/raw/resumes"
PROCESSED_JSON_PATH = "data/processed/cleaned_resumes.json"

def main():
    resumes = []

    # Load resumes by extracting raw text from files
    for filename in os.listdir(RAW_RESUME_DIR):
        file_path = os.path.join(RAW_RESUME_DIR, filename)
        try:
            content = extract_text(file_path)
            resumes.append({"filename": filename, "content": content})
            print(f"Extracted text from {filename}")
        except Exception as e:
            print(f"Failed to extract text from {filename}: {e}")

    structured = []

    # Clean and extract skills
    for r in resumes:
        cleaned = spacy_clean_text(r["content"])
        skills = extract_skills(cleaned)

        structured.append({
            "filename": r["filename"],
            "original_length": len(r["content"]),
            "cleaned_length": len(cleaned),
            "skills": skills,
            "cleaned_text": cleaned
        })

    # Save processed resumes as JSON
    os.makedirs(os.path.dirname(PROCESSED_JSON_PATH), exist_ok=True)
    with open(PROCESSED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print(f"✅ Cleaned resumes and extracted skills saved to {PROCESSED_JSON_PATH}")

    # --- VECTORIZE cleaned texts ---
    texts = [r["cleaned_text"] for r in structured]
    vectorizer, vectors = build_vectorizer(texts)

    # Save vectorizer for later use
    save_vectorizer(vectorizer)

    # --- CLUSTER resumes ---
    model, labels = cluster_resumes(vectors)
    structured = assign_clusters(structured, labels)

    print("✅ Clustering done and clusters assigned.")

    # Optionally, save clustered resumes with labels
    clustered_path = "outputs/clustered_resumes.json"
    os.makedirs(os.path.dirname(clustered_path), exist_ok=True)
    with open(clustered_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"✅ Clustered resumes saved to {clustered_path}")

if __name__ == "__main__":
    main()
