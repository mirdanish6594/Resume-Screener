# =========================================================================
# CRITICAL: These two lines MUST be the very first lines in this file.
import eventlet
eventlet.monkey_patch()
# =========================================================================

import os
import pickle
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from celery import Celery

# Import all ML and file processing libraries
import fitz
import docx
import nltk
from nltk.corpus import stopwords
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- 1. Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. Celery Configuration ---
# The SSL configuration will now be read directly from the URL in .env
redis_url = os.getenv("REDIS_URL")
if not redis_url:
    raise RuntimeError("REDIS_URL not found in .env file. Worker cannot start.")

celery = Celery(__name__, broker=redis_url, backend=redis_url)
celery.conf.task_track_started = True

# --- 3. Load All Models on Worker Startup (Unchanged) ---
try:
    logger.info("Worker: Downloading NLTK stopwords...")
    # ... (The rest of your model loading code remains exactly the same) ...
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    MODEL_DIR = Path("models")
    with open(MODEL_DIR / "model.pkl", "rb") as f:
        classification_model = pickle.load(f)
    with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Worker: Classification model and vectorizer loaded.")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Worker: SentenceTransformer model loaded.")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=google_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    logger.info("Worker: Google Gemini model configured and ready.")
except Exception as e:
    logger.error(f"Worker: FATAL - Could not load models. {str(e)}")
    raise e

# --- 4. Helper Functions (Unchanged) ---
def clean_text(text: str) -> str:
    # ... (Your clean_text function remains exactly the same) ...
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# --- 5. Define Celery Tasks (Unchanged) ---
@celery.task(name="predict_role_from_text")
def predict_role_from_text(raw_text: str):
    # ... (Your predict_role_from_text task remains exactly the same) ...
    logger.info(f"Task: Received job for role prediction.")
    try:
        clean_resume = clean_text(raw_text)
        if len(clean_resume.split()) < 10:
            return {"error": "Insufficient text content for a reliable prediction."}
        vectorized = vectorizer.transform([clean_resume])
        prediction = classification_model.predict(vectorized)[0]
        confidence = classification_model.predict_proba(vectorized).max()
        logger.info(f"Task: Role prediction successful.")
        return {"predicted_role": str(prediction), "confidence": float(confidence)}
    except Exception as e:
        logger.error(f"Task: Role prediction failed: {e}")
        return {"error": f"An error occurred during prediction: {e}"}

@celery.task(name="match_job_from_text")
def match_job_from_text(resume_text: str, job_description_text: str):
    # ... (Your match_job_from_text task remains exactly the same) ...
    logger.info(f"Task: Received job for RAG analysis.")
    try:
        resume_chunks = [chunk for chunk in resume_text.split('\n\n') if chunk.strip()]
        if not resume_chunks:
            return {"error": "Resume text is empty or invalid."}
        resume_embeddings = embedding_model.encode(resume_chunks)
        dimension = resume_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(resume_embeddings, dtype=np.float32))
        job_embedding = embedding_model.encode([job_description_text])
        k = min(len(resume_chunks), 3)
        relevant_chunks = [resume_chunks[i] for i in index.search(np.array(job_embedding, dtype=np.float32), k)[1][0]]
        retrieved_context = "\n---\n".join(relevant_chunks)
        prompt = f"""
        You are an expert recruitment analyst. Perform a detailed analysis...
        JOB DESCRIPTION: {job_description_text}
        RELEVANT RESUME CONTEXT: {retrieved_context}
        """
        response = gemini_model.generate_content(prompt)
        analysis = response.text
        logger.info(f"Task: RAG analysis successful.")
        return {"analysis": analysis, "retrieved_resume_parts": relevant_chunks}
    except Exception as e:
        logger.error(f"Task: RAG analysis failed: {e}")
        return {"error": f"Failed to generate analysis from the AI model: {e}"}