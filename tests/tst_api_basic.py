import pytest
from fastapi.testclient import TestClient
from app.main import app # Import your FastAPI app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to VTIERP API (Custom Notebook Adaptation). Use /docs for API documentation."}

# Add more basic tests here, e.g., for /docs endpoint
def test_read_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    # Check for some content if it's HTML
    assert "text/html" in response.headers["content-type"]

# Placeholder for testing an endpoint that requires API key (would need mocking or live key in CI)
# def test_upload_pdf_no_file(monkeypatch):
#     # This would require mocking GOOGLE_API_KEY for LLM initialization if app pre-initializes
#     # For now, just an example of how you might start
#     # monkeypatch.setenv("GOOGLE_API_KEY", "fake_key_for_test_startup")
#     response = client.post("/upload-pdf/") # Should fail as no file is provided
#     assert response.status_code == 422 # Unprocessable Entity for missing file