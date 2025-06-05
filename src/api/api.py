from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporary for debugging!
    allow_methods=["*"],
    allow_headers=["*"],
)