import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from celery.result import AsyncResult

# Import the tasks we defined in worker.py
from worker import predict_role_from_text, match_job_from_text

# --- 1. Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# --- 2. Pydantic Models ---
class ResumeText(BaseModel):
    text: str

class JobMatchRequest(BaseModel):
    resume_text: str
    job_description_text: str

class TaskResponse(BaseModel):
    task_id: str

# --- 3. Initialize FastAPI ---
app = FastAPI(
    title="Resume Screener API - Web Service",
    description="A lightweight web server that delegates ML tasks to a background worker.",
    version="4.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Helper functions for file handling ---
def extract_text_from_file(file_path: Path) -> Optional[str]:
    import fitz # PyMuPDF
    import docx
    file_ext = file_path.suffix.lower()
    try:
        if file_ext == '.pdf':
            with fitz.open(file_path) as doc:
                return " ".join(page.get_text() for page in doc)
        elif file_ext in ['.docx', '.doc']:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        return None
    except Exception as e:
        logger.error(f"File extraction failed for {file_path}: {str(e)}")
        return None

# --- 5. API Routes ---
@app.get("/", summary="Health Check")
async def health_check():
    return {"status": "Web service is healthy"}

@app.post("/predict-text", summary="Start Role Prediction from Text", response_model=TaskResponse)
async def post_predict_text(request: ResumeText):
    """Sends a raw text prediction job to the background worker."""
    logger.info("Web: Received predict-text job, sending to worker.")
    task = predict_role_from_text.delay(request.text)
    return {"task_id": task.id}

@app.post("/predict-upload", summary="Start Role Prediction from File", response_model=TaskResponse)
async def post_predict_upload(file: UploadFile = File(...)):
    """Uploads a file, extracts text, and sends a prediction job to the worker."""
    logger.info(f"Web: Received file for upload: {file.filename}")
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ['.pdf', '.docx', '.doc']:
        raise HTTPException(400, "Invalid file type. Please upload .pdf or .docx")

    temp_path = TEMP_DIR / f"temp_{os.urandom(8).hex()}{file_ext}"
    raw_text = ""
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if temp_path.stat().st_size == 0:
            raise HTTPException(400, "Empty file uploaded.")

        raw_text = extract_text_from_file(temp_path)
        if not raw_text:
            raise HTTPException(422, "Could not extract text from file.")
    finally:
        if temp_path.exists():
            temp_path.unlink()
    
    logger.info("Web: File text extracted, sending predict-text job to worker.")
    task = predict_role_from_text.delay(raw_text)
    return {"task_id": task.id}

@app.post("/match-job", summary="Start a Job Match Analysis Task", response_model=TaskResponse)
async def post_match_job(request: JobMatchRequest):
    """Sends a RAG job to the background worker."""
    logger.info("Web: Received match-job job, sending to worker.")
    task = match_job_from_text.delay(request.resume_text, request.job_description_text)
    return {"task_id": task.id}

@app.get("/results/{task_id}", summary="Get Task Results")
async def get_task_results(task_id: str):
    """Frontend polls this endpoint with a task_id to get the result."""
    task_result = AsyncResult(task_id)
    if task_result.ready():
        if task_result.successful():
            result = task_result.get()
            return {"status": "SUCCESS", "result": result}
        else:
            # For security, don't return the raw exception. Log it instead.
            logger.error(f"Task {task_id} failed with error: {task_result.info}")
            return {"status": "FAILURE", "error": "An error occurred in the background worker."}
    else:
        return {"status": "PENDING"}