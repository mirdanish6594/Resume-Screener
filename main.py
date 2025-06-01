from src.ingestion.loader import load_resumes
from src.preprocessing.cleaner import clean_text, extract_skills
import json

resumes = load_resumes("data/raw/resumes")

structured = []

for r in resumes:
    cleaned = clean_text(r["content"])
    skills = extract_skills(cleaned)

    structured.append({
        "filename": r["filename"],
        "original_length": len(r["content"]),
        "cleaned_length": len(cleaned),
        "skills": skills,
        "cleaned_text": cleaned
    })

with open("data/processed/cleaned_resumes.json", "w", encoding="utf-8") as f:
    json.dump(structured, f, indent=2, ensure_ascii=False)

print("✅ Cleaned resumes and extracted skills saved to data/processed/cleaned_resumes.json")
