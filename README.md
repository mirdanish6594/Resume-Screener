# Resume Screener ML Project 🚀

An end-to-end NLP-based resume classification pipeline with MLOps and deployment.

## Features
- Extracts and processes resumes (PDF, DOCX)
- Classifies resumes into job roles (Data Scientist, Web Dev, etc.)
- FastAPI endpoint for real-time predictions
- Dockerized, CI/CD-enabled, cloud-deployed
- MLOps: Auto-retraining, monitoring, explainability

## Tech Stack
- **ML**: scikit-learn, spaCy, Optuna
- **API**: FastAPI, Docker, Render
- **MLOps**: GitHub Actions, MLflow, DVC, Prefect

## Folder Structure
resume-screener-mlops/
├── data/              # Raw + processed data
├── notebooks/         # Jupyter exploration
├── models/            # Saved models
├── outputs/           # Metrics, logs, results
├── src/
│   ├── preprocessing/ # Cleaning & feature extraction
│   ├── training/      # Training pipeline
│   ├── api/           # FastAPI app
│   └── utils/         # Reusable functions
├── .gitignore
├── requirements.txt
├── README.md
└── main.py