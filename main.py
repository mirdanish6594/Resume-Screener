# main.py

import os
import pickle
import re
import shutil
import fitz  # PyMuPDF for PDF parsing
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Load model and vectorizer ---
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# --- Initialize FastAPI ---
app = FastAPI()

# --- Preprocessing ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "Welcome to Resume Screener API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF temporarily
        temp_path = "temp_resume.pdf"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text and clean it
        raw_text = extract_text_from_pdf(temp_path)
        clean_resume = clean_text(raw_text)

        # Vectorize and predict
        vectorized = vectorizer.transform([clean_resume])
        prediction = model.predict(vectorized)[0]

        # Remove temp file
        os.remove(temp_path)

        return {"predicted_job_role": prediction}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
