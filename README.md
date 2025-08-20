# AI-Powered Resume Screener

An intelligent application designed to automate the initial stages of recruitment. This tool predicts a candidate's most likely job role from their resume and provides detailed, AI-driven feedback to help them improve it.

The project features a decoupled architecture with a Python/FastAPI backend for core model inference and a React frontend for user interaction and advanced analysis via the Hugging Face API.

---

## Table of Contents
- [Live Demo](#live-demo)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Live Demo
***[https://resumescreenerai.netlify.app/]***

---

## Features
-   **Automated Role Prediction**: Upload a resume (PDF) or paste its text to get an instant prediction of the most suitable job category (e.g., Data Scientist, Software Engineer).
-   **Client-Side PDF Parsing**: Efficiently extracts text from PDF files directly in the browser using `pdf.js` for quick processing.
-   **Advanced AI Analysis**: Leverages multiple Hugging Face models to provide a comprehensive resume review, including:
    -   A contextual score out of 100.
    -   Key strengths and achievements.
    -   Actionable, prioritized suggestions for improvement.
    -   Identification of missing key elements (e.g., GitHub link, quantifiable metrics).
    -   Role-specific advice tailored to the predicted job category.
-   **RESTful Backend**: A robust FastAPI backend serves the primary classification model, containerized with Docker for scalability and portability.
-   **Decoupled Frontend**: A modern and responsive React frontend deployed on Netlify for a smooth user experience.

---

## Architecture
The system is built with a decoupled frontend and backend, interacting with external AI services for enhanced analysis.

1.  **Frontend (React on Netlify)**: The user interacts with the React application. They can either upload a PDF or paste resume text. The frontend's `pdf.js` library extracts the text client-side.
2.  **Backend API (FastAPI on Render)**: The extracted text is sent to our FastAPI backend. The backend cleans the text, vectorizes it using a saved TF-IDF vectorizer, and predicts the job role using a pre-trained Scikit-learn classification model. The predicted role is returned to the frontend.
3.  **Hugging Face API**: The frontend then takes the original resume text and the predicted role and makes direct calls to the Hugging Face Inference API. It queries multiple models to perform skill extraction, experience level classification, and generate strengths and improvement suggestions.
4.  **Display Results**: All results—the predicted role from the backend and the detailed analysis from Hugging Face—are compiled and displayed to the user.
```
+--------------------------------+      +---------------------------------+
|      Frontend (React)          |      |    Hugging Face Inference API   |
|      (Deployed on Netlify)     |      |  (For detailed analysis)        |
+--------------------------------+      +---------------------------------+
| 1. User uploads PDF/pastes text|      | 4. Frontend sends text & role   |
| 2. Client-side text extraction |----->|    for detailed AI feedback     |
| 3. Sends text to Backend API...|      | 5. Receives score, strengths,   |
+-----------------|--------------+      |    and improvement suggestions  |
|                     +-----------------^---------------+
v                                       |
+--------------------------------+                        |
|       Backend API (FastAPI)    |                        |
|   (Dockerized on Render)       |                        |
+--------------------------------+                        |
| - Receives text                |                        |
| - Cleans & vectorizes          |                        |
| - Predicts role with SKLearn   |                        |
| - Returns predicted role       |------------------------+
+--------------------------------+
```

## Technologies Used
-   **Backend**:
    -   Python 3.8+
    -   FastAPI
    -   Scikit-learn
    -   PyMuPDF (Fitz), python-docx
    -   NLTK
-   **Frontend**:
    -   React (with Vite)
    -   TypeScript
    -   pdf.js
-   **AI & ML**:
    -   Scikit-learn (for role classification)
    -   Hugging Face Inference API (for generative analysis)
-   **Deployment & DevOps**:
    -   Docker
    -   Render (for Backend Hosting)
    -   Netlify (for Frontend Hosting)
    -   Git & GitHub (for Version Control)

---

## Installation and Setup

### Prerequisites
-   Python 3.8+ and Pip
-   Node.js and npm/yarn
-   Git
-   Docker (Recommended)

### Setup Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mirdanish6594/Resume-Screener.git](https://github.com/mirdanish6594/Resume-Screener.git)
    cd Resume-Screener
    ```

2.  **Setup Backend:**
    -   Navigate to the backend directory (if separate).
    -   Create and activate a Python virtual environment:
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
    -   Install backend dependencies:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Setup Frontend:**
    -   Navigate to the frontend directory.
    -   Create a `.env` file in the root of the frontend folder.
    -   Add your Hugging Face API key to the `.env` file:
        ```
        REACT_APP_HF_API_KEY="your_hugging_face_api_key_here"
        ```
    -   Install frontend dependencies:
        ```bash
        npm install
        ```

---

## Usage

### Running Locally
1.  **Start the Backend Server:**
    From the backend directory, run:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.

2.  **Start the Frontend Application:**
    From the frontend directory, run:
    ```bash
    npm run dev
    ```
    The application will be accessible at `http://localhost:5173` (or another port if specified).

### Using Docker
You can build and run the backend service using Docker.

# Build the Docker image
```docker build -t resume-screener-api ```

# Run the container
```docker run -d -p 8000:8000 resume-screener-api```
API Endpoints
The backend provides the following endpoints:

```GET /:``` Health check to confirm the API is running and models are loaded.

```POST /predict-text:``` Accepts a JSON payload with raw text and returns a predicted job role.

```Body: { "text": "your resume text here..." }```

```POST /predict-upload:``` Accepts a file upload (.pdf, .docx) and returns a predicted job role.

# Future Work
- This project has a solid foundation, and future enhancements could include:

- CI/CD Integration: Automate testing and deployment using GitHub Actions to both Render and Netlify.

- Model Monitoring & Retraining: Implement tools like MLflow or DVC to track model performance and set up a pipeline for automated retraining on new data.

- RAG Implementation: Integrate a Retrieval-Augmented Generation (RAG) system to match resumes against specific job descriptions for more granular analysis.

- Database Integration: Store prediction results and user feedback in a database to track model accuracy and gather data for retraining.

- HR Dashboard: Create a secure dashboard for recruiters to manage multiple resumes, track candidates, and view analytics.

# Contributing
Contributions are welcome! If you'd like to help improve the project, please follow these steps:

- Fork the repository.

- Create a new feature branch (git checkout -b feature/AmazingFeature).

- Commit your changes (git commit -m 'Add some AmazingFeature').

- Push to the branch (git push origin feature/AmazingFeature).

- Open a Pull Request.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Contact
Created by Danish Mir - feel free to reach out with any questions or collaboration ideas!
