from fastapi import FastAPI, File, UploadFile, HTTPException
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel
from typing import List
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

#start the fastapi app
app = FastAPI()

# Minio Configuration
MINIO_ENDPOINT = "play.min.io:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = "my-bucket2"

print(MINIO_ENDPOINT)
# Initialize Minio client
minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)

# Create the bucket if it doesn't exist
try:
    if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
        minio_client.make_bucket(MINIO_BUCKET_NAME)
except S3Error as err:
    raise HTTPException(status_code=500, detail=str(err))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data validation model
class FileUploadResponse(BaseModel):
    file_ids: List[str]

# File upload endpoint
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    allowed_types = ["application/pdf", "image/tiff", "image/png", "image/jpeg"]
    file_ids = []
    for file in files:
        if file.content_type not in allowed_types:
            logger.error(f"Invalid file type for file: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Invalid file type. Only {', '.join(allowed_types)} are allowed.")

        try:
            # Save file to Minio
            minio_client.put_object(MINIO_BUCKET_NAME, file.filename, file.file, -1, part_size=10*1024*1024)

            # Get signed URL for the uploaded file
            presigned_url = minio_client.presigned_get_object(MINIO_BUCKET_NAME, file.filename)

            file_ids.append(presigned_url)
            logger.info(f"File {file.filename} uploaded successfully")
        except S3Error as err:
            logger.error(f"Error uploading file: {err}")
            raise HTTPException(status_code=500, detail=f"Error uploading file: {err}")

    return FileUploadResponse(file_ids=file_ids)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)