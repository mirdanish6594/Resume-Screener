import os
import pickle
import re
import shutil
import fitz  # PyMuPDF for PDF parsing
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
import logging

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI with better defaults ---
app = FastAPI(
    title="Resume Screener API",
    description="API for predicting job roles from resumes",
    version="1.0.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# --- Load model and vectorizer with validation ---
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise RuntimeError("Failed to initialize model resources") from e

# --- Enhanced preprocessing ---
def clean_text(text: str) -> str:
    """Normalize and clean input text"""
    if not text:
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    # More efficient stopword removal
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def extract_text_from_pdf(file_path: Path) -> Optional[str]:
    """Safely extract text from PDF"""
    try:
        text = ""
        with fitz.open(file_path) as doc:
            text = " ".join(page.get_text() for page in doc)
        return text if text.strip() else None
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return None

# --- API Routes ---
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Resume Screener API"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict job role from resume PDF"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, detail="Only PDF files are accepted")
    
    temp_path = TEMP_DIR / f"temp_{os.urandom(8).hex()}.pdf"
    
    try:
        # Secure file handling
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if temp_path.stat().st_size == 0:
            raise HTTPException(400, detail="Empty file uploaded")
        
        # Process content
        raw_text = extract_text_from_pdf(temp_path)
        if not raw_text:
            raise HTTPException(400, detail="No extractable text in PDF")
        
        clean_resume = clean_text(raw_text)
        if len(clean_resume.split()) < 10:  # Minimum word check
            raise HTTPException(400, detail="Insufficient text content")
        
        # Prediction
        try:
            vectorized = vectorizer.transform([clean_resume])
            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized).max()
            
            return {
                "predicted_role": prediction,
                "confidence": float(confidence),
                "processed_text_length": len(clean_resume)
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(500, detail="Prediction service error")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, detail="Internal server error")
    finally:
        # Always clean up
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Temp file deletion failed: {str(e)}")