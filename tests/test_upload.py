import pytest
from fastapi.testclient import TestClient
from ..main import app
from endpoints.upload_files import generate_unique_file_id, generate_presigned_url
from config import MINIO_BUCKET_NAME, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
from minio import Minio

client = TestClient(app)

def test_upload_single_file():
    # Test uploading a single file
    response = client.post("/upload", files={"file": ("test.pdf", open("tests/test_files/test.pdf", "rb"), "application/pdf")})
    assert response.status_code == 200
    assert response.json()["file_urls"] != {}

def test_upload_multiple_files():
    # Test uploading multiple files
    response = client.post("/upload", files={"file1": ("test1.pdf", open("tests/test_files/test1.pdf", "rb"), "application/pdf"), "file2": ("test2.pdf", open("tests/test_files/test2.pdf", "rb"), "application/pdf")})
    assert response.status_code == 200
    assert response.json()["file_urls"] != {}

def test_upload_invalid_file_type():
    # Test uploading a file with an invalid file type
    response = client.post("/upload", files={"file": ("test.txt", open("tests/test_files/test.txt", "rb"), "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid file type. Only application/pdf, image/tiff, image/png, image/jpeg are allowed."

def test_upload_file_size_exceeds_limit():
    # Test uploading a file that exceeds the size limit
    # Note: This test will fail if the file size limit is not set in the code
    response = client.post("/upload", files={"file": ("test.pdf", open("tests/test_files/test.pdf", "rb"), "application/pdf")})
    assert response.status_code == 500
    assert response.json()["detail"] == "Error uploading file: File size exceeds the limit."

def test_generate_unique_file_id():
    # Test generating a unique file ID
    unique_file_id = generate_unique_file_id()
    assert len(unique_file_id) == 32

def test_generate_presigned_url():
    # Test generating a presigned URL
    minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)
    presigned_url = generate_presigned_url(minio_client, MINIO_BUCKET_NAME, "test.pdf")
    assert presigned_url.startswith("https://")
