# Import necessary modules
from fastapi import APIRouter, File, UploadFile, HTTPException
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel
from typing import Dict, List
import logging
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME
import uuid

# Create a router instance
router = APIRouter()

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

# Define response model
class FileUploadResponse(BaseModel):
    # file_ids: List[str]
    file_urls: Dict[str, str]

# Define endpoint for file upload
@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    # Define allowed file types
    allowed_types = ["application/pdf", "image/tiff", "image/png", "image/jpeg"]
    file_urls = {} #empty dictionary to store the file id and urls
    for file in files:
        if file.content_type not in allowed_types:
            logger.error(f"Invalid file type for file: {file.filename}")
            raise HTTPException(status_code=400, detail=f"Invalid file type. Only {', '.join(allowed_types)} are allowed.")

        try:
            # Generate unique file ID using uuid4
            unique_file_id = generate_unique_file_id()

            # Save file to Minio
            minio_client.put_object(MINIO_BUCKET_NAME, file.filename, file.file, -1, part_size=10*1024*1024, content_type=file.content_type)

            try:
                # Get signed URL for the uploaded file
                presigned_url = generate_presigned_url(minio_client, MINIO_BUCKET_NAME, file.filename)
            except S3Error as err:
                logger.error(f"Error occurred while generating presigned URL for file: {err}")
                raise HTTPException(status_code=500, detail=f"Error occurred generating presigned URL for file: {err}")
            
            # # Get signed URL for the uploaded file with default expiry (i.e. 7 days)
            # presigned_url = minio_client.presigned_get_object(MINIO_BUCKET_NAME, file.filename)

            file_urls[unique_file_id] = presigned_url
            logger.info(f"File {file.filename} uploaded successfully")
        except S3Error as err:
            logger.error(f"Error uploading file: {err}")
            raise HTTPException(status_code=500, detail=f"Error uploading file: {err}")

    return FileUploadResponse(file_urls=file_urls)

def generate_unique_file_id():
    """Function to generate unique file id

    Returns:
        _type_: returns the hex code of the unique file identifier
    """
    return uuid.uuid4().hex


def generate_presigned_url(minio_client, bucket_name, object_name):
    """Seperate function to generate the pre-signed URL

    Args:
        minio_client: Minio client object
        bucket_name: name of the bucket for the file upload
        object_name: name of the object/file to upload

    Returns:
        URL: Returns the URL of the file upload with default expiry
    """
    # Get signed URL for the uploaded file with default expiry (7 days)
    return minio_client.presigned_get_object(bucket_name, object_name)