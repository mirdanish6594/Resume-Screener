import os
import pickle
import re
import shutil
import fitz  # PyMuPDF for PDF parsing
import docx  # For .docx files
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import logging

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Model for Text Input ---
class ResumeText(BaseModel):
    """Defines the request model for text-based prediction."""
    text: str

# --- Initialize FastAPI ---
app = FastAPI(
    title="Resume Screener API",
    description="API for predicting job roles from resumes (supports PDF, DOCX, and text input)",
    version="2.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Be more specific in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True) # Create temp directory if it doesn't exist

# --- Load Model and Vectorizer ---
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logger.error(f"Fatal: Could not load model or vectorizer. {str(e)}")
    # This is a critical error, the app shouldn't start.
    raise RuntimeError("Failed to initialize model resources") from e

# --- Text Processing and Extraction Functions ---

def clean_text(text: str) -> str:
    """Normalizes and cleans input text by removing URLs, non-alphabetic characters, and stopwords."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def extract_text_from_pdf(file_path: Path) -> Optional[str]:
    """Safely extracts text from a PDF file."""
    try:
        with fitz.open(file_path) as doc:
            return " ".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {str(e)}")
        return None

def extract_text_from_docx(file_path: Path) -> Optional[str]:
    """Safely extracts text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"DOCX extraction failed for {file_path}: {str(e)}")
        return None

# --- API Routes ---

@app.get("/", summary="Health Check")
async def health_check():
    """Provides a basic health check of the API."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "service": "Resume Screener API v2"
    }

@app.post("/predict-upload", summary="Predict from Resume File")
async def predict_upload(
    file: UploadFile = File(..., description="A resume file (PDF or DOCX)")
):
    """Predicts the job role from an uploaded resume file (PDF or DOCX)."""
    raw_text = ""

    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(400, detail="Invalid file type. Please upload a .pdf or .docx file.")
    
    temp_path = TEMP_DIR / f"temp_{os.urandom(8).hex()}{file_ext}"
    try:
        # Save file securely
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if temp_path.stat().st_size == 0:
            raise HTTPException(400, detail="Empty file uploaded.")
        
        # Extract text based on file type
        if file_ext == '.pdf':
            raw_text = extract_text_from_pdf(temp_path)
        elif file_ext in ['.docx', '.doc']:
            raw_text = extract_text_from_docx(temp_path)
    
    finally:
        # Always clean up the temporary file
        if temp_path.exists():
            temp_path.unlink()

    # Prediction
    if not raw_text or not raw_text.strip():
        raise HTTPException(400, detail="Could not extract any text from the uploaded file.")

    clean_resume = clean_text(raw_text)
    if len(clean_resume.split()) < 15:
        raise HTTPException(400, detail=f"Insufficient text content for a reliable prediction (found {len(clean_resume.split())} words).")

    try:
        vectorized = vectorizer.transform([clean_resume])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()
        
        return {
            "predicted_role": str(prediction),
            "confidence": float(confidence),
            "processed_text_length": len(clean_resume),
            "source": "file-upload"
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, detail="An error occurred during the prediction process.")

@app.post("/predict-text", summary="Predict from Pasted Resume Text")
async def predict_text(
    resume_data: ResumeText = Body(..., description="A JSON object with resume text")
):
    """Predicts the job role from pasted resume text."""
    raw_text = resume_data.text

    if not raw_text or not raw_text.strip():
        raise HTTPException(400, detail="No text provided for prediction.")

    clean_resume = clean_text(raw_text)
    if len(clean_resume.split()) < 15:
        raise HTTPException(400, detail=f"Insufficient text content for a reliable prediction (found {len(clean_resume.split())} words).")

    try:
        vectorized = vectorizer.transform([clean_resume])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()
        
        return {
            "predicted_role": str(prediction),
            "confidence": float(confidence),
            "processed_text_length": len(clean_resume),
            "source": "text-input"
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, detail="An error occurred during the prediction process.")
