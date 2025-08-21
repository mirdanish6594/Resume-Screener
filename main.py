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
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
# This will load variables from your .env file (e.g., OPENAI_API_KEY)
load_dotenv()

# --- Pydantic Models ---
class ResumeText(BaseModel):
    """Defines the request model for text-based prediction."""
    text: str

class JobMatchRequest(BaseModel):
    """Defines the request model for the RAG job matching feature."""
    resume_text: str
    job_description_text: str

# --- Initialize FastAPI ---
app = FastAPI(
    title="Resume Screener API",
    description="API for predicting job roles and matching resumes to job descriptions.",
    version="3.0.0"
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
TEMP_DIR.mkdir(exist_ok=True)

# --- Load All Models on Startup ---
try:
    # Download NLTK stopwords
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

    # Load original classification model and vectorizer
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Original classification model and vectorizer loaded successfully.")

    # Load SentenceTransformer model for RAG embeddings
    logger.info("Loading SentenceTransformer model for RAG...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("SentenceTransformer model loaded.")

except Exception as e:
    logger.error(f"Fatal: Could not load one or more models. {str(e)}")
    raise RuntimeError("Failed to initialize model resources") from e


# --- Helper Functions ---
def clean_text(text: str) -> str:
    """Normalizes and cleans input text."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
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
        "classification_model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "embedding_model_loaded": embedding_model is not None,
        "service": "Resume Screener API v3"
    }

@app.post("/predict-upload", summary="Predict Role from Resume File")
async def predict_upload(file: UploadFile = File(..., description="A resume file (PDF or DOCX)")):
    """Predicts the job role from an uploaded resume file."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(400, detail="Invalid file type. Please upload a .pdf or .docx file.")

    temp_path = TEMP_DIR / f"temp_{os.urandom(8).hex()}{file_ext}"
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if temp_path.stat().st_size == 0:
            raise HTTPException(400, detail="Empty file uploaded.")

        if file_ext == '.pdf':
            raw_text = extract_text_from_pdf(temp_path)
        else: # .docx or .doc
            raw_text = extract_text_from_docx(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if not raw_text or not raw_text.strip():
        raise HTTPException(400, detail="Could not extract any text from the uploaded file.")

    clean_resume = clean_text(raw_text)

    logger.info(f"Cleaned word count: {len(clean_resume.split())}")
    if len(clean_resume.split()) < 10:
        raise HTTPException(400, detail=f"Insufficient text content for a reliable prediction.")

    try:
        vectorized = vectorizer.transform([clean_resume])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()
        return {
            "predicted_role": str(prediction),
            "confidence": float(confidence),
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, detail="An error occurred during the prediction process.")

@app.post("/predict-text", summary="Predict Role from Pasted Text")
async def predict_text(resume_data: ResumeText):
    """Predicts the job role from pasted resume text."""
    raw_text = resume_data.text
    if not raw_text or not raw_text.strip():
        raise HTTPException(400, detail="No text provided for prediction.")

    clean_resume = clean_text(raw_text)
    if len(clean_resume.split()) < 10:
        raise HTTPException(400, detail=f"Insufficient text content for a reliable prediction.")

    try:
        vectorized = vectorizer.transform([clean_resume])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized).max()
        return {
            "predicted_role": str(prediction),
            "confidence": float(confidence),
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, detail="An error occurred during the prediction process.")
    

# Replace the ENTIRE /match-job endpoint with this FINAL version

@app.post("/match-job", summary="Match Resume to a Job Description using RAG")
async def match_job_description(request: JobMatchRequest):
    logger.info("Received request for /match-job endpoint.")

    # --- Steps 1 (Ingest) and 2 (Retrieve) are unchanged ---
    resume_chunks = [chunk for chunk in request.resume_text.split('\n\n') if chunk.strip()]
    if not resume_chunks:
        raise HTTPException(status_code=400, detail="Resume text is empty or invalid.")

    resume_embeddings = embedding_model.encode(resume_chunks)

    dimension = resume_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(resume_embeddings, dtype=np.float32))

    job_embedding = embedding_model.encode([request.job_description_text])
    k = min(len(resume_chunks), 3)
    relevant_chunks = [resume_chunks[i] for i in index.search(np.array(job_embedding, dtype=np.float32), k)[1][0]]
    retrieved_context = "\n---\n".join(relevant_chunks)
    logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")

    # --- Step 3: Generate Analysis with Google Gemini API ---
    logger.info("Step 3: Generating analysis with Google Gemini API.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Google API key not found in .env file.")

    try:
        # Configure the Gemini client with the API key
        genai.configure(api_key=api_key)

        # Create the prompt for the model
        prompt = f"""
        You are an expert recruitment analyst. Perform a detailed analysis of the candidate's resume against the provided job description.
        Please structure your analysis into three distinct sections: Matching Skills, Experience Alignment, and Potential Gaps.

        JOB DESCRIPTION:
        {request.job_description_text}

        RELEVANT RESUME CONTEXT:
        {retrieved_context}
        """

        # Initialize the Gemini Pro model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Generate the analysis
        response = model.generate_content(prompt)
        analysis = response.text
        logger.info("Successfully generated analysis from Google Gemini API.")

    except Exception as e:
        logger.error(f"Google Gemini API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate analysis from the AI model: {e}")

    return {
        "analysis": analysis,
        "retrieved_resume_parts": relevant_chunks
    }
