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
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, OUTPUT_DIR
import requests
import io
from PIL import Image
import PyPDF2
import json
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_NAME
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
    pre_signed_url: str

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
        # Fetch the file using the signed URL
        response = requests.get(data.pre_signed_url)
        print(response.status_code)
        print(response.headers['Content-Type'])
        print(data.pre_signed_url.endswith(".pdf"))
        
        # Download the file
        x = minio_client.fget_object(MINIO_BUCKET_NAME, "σ╗║τ»ëσƒ║µ║ûµ│òµû╜ΦíîΣ╗ñ.pdf", "σ╗║τ»ëσƒ║µ║ûµ│òµû╜ΦíîΣ╗ñ.txt")    
        print(x.last_modified)

        # Determine the file type and process accordingly
        if response.headers['Content-Type'] == 'application/pdf':
            ocr_text = extract_text_from_pdf(response.content)
        else:
            image = Image.open(io.BytesIO(response.content))
            ocr_text = pytesseract.image_to_string(image)

        # Save the extracted text in a JSON file
        filename = os.path.splitext(os.path.basename(data.pre_signed_url))[0] + ".json"
        save_ocr_result(filename, ocr_text, "succeeded")

        logger.info(f"OCR processing completed for signed URL: {data.pre_signed_url}")
        return OCRResponse(message="OCR processing successful")
    except Exception as e:
        logger.error(f"Error processing OCR: {e}")
        filename = os.path.splitext(os.path.basename(data.pre_signed_url))[0] + ".json"
        save_ocr_result(filename, "", "failed")
        raise HTTPException(status_code=500, detail="Error processing OCR")
    

def extract_text_from_pdf(pdf_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    ocr_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        ocr_text += page.extract_text()
    return ocr_text

def save_ocr_result(filename, ocr_text, status):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, filename)
    with open(file_path, "w") as f:
        json.dump({"filename": filename, "status": status, "analyzeResult": {"content": ocr_text}}, f)