from src.preprocessing.vectorizer import build_vectorizer, save_vectorizer
import json

with open("data/processed/cleaned_resumes.json", "r") as f:
    resumes = json.load(f)

texts = [r["cleaned_text"] for r in resumes]
vectorizer, vectors = build_vectorizer(texts)

save_vectorizer(vectorizer)
