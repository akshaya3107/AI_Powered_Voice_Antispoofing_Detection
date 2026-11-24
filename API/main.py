"""Entry point for the FastAPI service.

Run from the ``API`` directory:

    python main.py

The API is then available at http://localhost:8000 (docs at /docs).
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)
