"""
ocr.py: The file used to implement the OCR endpoint

"""
#import required packages
import os
import logging
import pytesseract
import openai
import pinecone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import io
from PIL import Image
import PyPDF2
import json
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, OUTPUT_DIR
from minio import Minio
#define the router
router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/ocr.log', format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Configure OpenAI api
# openai.api_key = OPENAI_API_KEY

# configure pinecone api
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# index = pinecone.Index("ocr-embeddings")

# Define request model
class OCRRequest(BaseModel):
    bucket_name: str = "my-bucket2"
    object_name: str

# Define response model
class OCRResponse(BaseModel):
    message: str

# Initialize Minio client
minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)

@router.post("/ocr", response_model=OCRResponse)
async def ocr_process(data: OCRRequest):
    """Function to perform OCR operation on the input data

    Args:
        data (OCRRequest): signed file string

    Raises:
        HTTPException: HTTP request exception

    Returns:
        JSON: Extracted OCR Information in JSON format
    """
    try:
        allowed_formats = [".pdf", ".tiff", ".png", ".jpeg"]

        try:
            # Fetch the file from MinIO using the bucket_name and object_name
            # response = minio_client.get_object(data.bucket_name, data.object_name)
            input_filepath = os.path.join(OUTPUT_DIR, data.object_name)
            print(input_filepath)
            minio_client.fget_object(data.bucket_name, data.object_name, input_filepath)
        except:
            logger.error("Failed to fetch the file from the given bucket, check the file details again")
            raise HTTPException(status_code=400, detail="Failed to fetch the file from the given bucket, check the file details again")
        
        file_extension = os.path.splitext(data.object_name)[1].lower()
        if file_extension not in allowed_formats:
            error_msg = f"Unsupported file format: {file_extension}. Only PDF, TIFF, PNG, JPEG formats are allowed."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Read the file for OCR processing
        with open(input_filepath, "rb") as file:
            file_bytes = file.read()

            # Determine the file type and process accordingly
            if file_extension == ".pdf":
                ocr_text = extract_text_from_pdf(file_bytes)
            else:
                image = Image.open(io.BytesIO(file_bytes))
                ocr_text = pytesseract.image_to_string(image)
        
        filename = os.path.splitext(data.object_name)[0] + ".json"
        # Save the extracted text in a JSON file                
        save_ocr_result(filename, ocr_text, "succeeded")

        logger.info(f"OCR processing completed for object: {data.object_name}")
        return OCRResponse(message="OCR processing successful")

    except Exception as e:
        logger.error(f"Error processing OCR: {e}")
        filename = os.path.splitext(data.object_name)[0] + ".json"   
        save_ocr_result(filename, "", "failed")
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {e}")
    

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    ocr_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        ocr_text += page.extract_text()
    return ocr_text

def save_ocr_result(filename, ocr_text, status):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, filename)
    print("ocr result")
    print(OUTPUT_DIR, filename, file_path)
    with open(file_path, "w") as f:
        json.dump({"status": status, "analyzeResult": {"content": ocr_text}}, f)