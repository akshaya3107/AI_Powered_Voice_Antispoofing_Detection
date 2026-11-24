"""FastAPI service exposing the voice deepfake detector.

Endpoints
---------
GET  /           Health check.
POST /deepfake   Upload an audio file (multipart field ``audio_file``) and get
                 ``{"status": 1, "Message": "REAL" | "FAKE"}`` back. On failure
                 ``status`` is 0 and ``Message`` holds the error.
POST /depfake    Legacy alias of /deepfake (kept for existing clients).
"""
import os
import shutil
import tempfile
import warnings

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.src.deepfake import infa_deepfake

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")


app = FastAPI(title="DeepFake", description="Voice deepfake detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {"Fast API": "API is working"}


@app.post("/deepfake")
@app.post("/depfake", include_in_schema=False)
def deepfake(audio_file: UploadFile = File(...)):
    """Save the upload to a temp file, run inference, and clean up."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Use only the base name to avoid writing outside the temp dir
        filename = os.path.basename(audio_file.filename or "") or "audio.wav"
        temp_file_path = os.path.join(temp_dir, filename)

        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(audio_file.file, temp_file)

        status, message = infa_deepfake(temp_file_path)
        return {"status": status, "Message": str(message)}

    except Exception as e:
        return {"status": 0, "Message": str(e)}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
