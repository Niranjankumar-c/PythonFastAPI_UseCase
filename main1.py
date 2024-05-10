from fastapi import FastAPI, File, UploadFile, HTTPException
from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Minio Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = "my-bucket1"

minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)

# Create the bucket if it doesn't exist
try:
    if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
        minio_client.make_bucket(MINIO_BUCKET_NAME)
except S3Error as err:
    raise HTTPException(status_code=500, detail=str(err))


@app.post("/upload")
async def upload_file(files: list[UploadFile] = File(...)):
    """
    Uploads one or more files to Minio object storage.

    Args:
        files (list[UploadFile]): List of files to be uploaded.

    Returns:
        dict: Dictionary containing the unique file identifiers or signed URLs for the uploaded files.
    """
    try:
        file_ids = []
        for file in files:
            # Check if the file type is allowed
            if file.content_type not in ["application/pdf", "image/tiff", "image/png", "image/jpeg"]:
                raise HTTPException(status_code=400, detail=f"File type {file.content_type} is not allowed.")

            # Upload the file to Minio
            file_name = file.filename
            print(file_name)
            minio_client.put_object(MINIO_BUCKET_NAME, file_name, file.file, -1, part_size=10*1024*1024)

            # Generate a signed URL for the uploaded file
            signed_url = minio_client.presigned_get_object(MINIO_BUCKET_NAME, file_name)
            file_ids.append(signed_url)

        return {"file_ids": file_ids}
    except S3Error as err:
        raise HTTPException(status_code=500, detail=str(err))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)