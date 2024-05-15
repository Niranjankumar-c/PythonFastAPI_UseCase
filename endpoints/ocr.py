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
import tifffile
import cv2


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
        logger.info(f"Analyzing OCR process request for object: {data.object_name}")
        
        file_extension = os.path.splitext(data.object_name)[1].lower()
        if file_extension not in allowed_formats:
            logger.info("Checking if the input file format is valid")
            error_msg = f"Unsupported file format: {file_extension}. Only PDF, TIFF, PNG, JPEG formats are allowed."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        try:
            # Fetch the file from MinIO using the bucket_name and object_name
            logger.info("Fetching the file from MinIO using bucket name and object name")
            input_filepath = os.path.join(OUTPUT_DIR, data.object_name)
            minio_client.fget_object(data.bucket_name, data.object_name, input_filepath)
        except:
            logger.error("Failed to fetch the file from the given bucket, check the file details again")
            raise HTTPException(status_code=400, detail="Failed to fetch the file from the given bucket, check the file details again")
        
        #implement the file related OCR processing
        logger.info("Extracting text information from the file using OCR")
        if file_extension == ".pdf":
            #read the pdf file and extracting the data
            ocr_text = extract_text_from_pdf(input_filepath)
        elif file_extension == ".tiff":
            ocr_text = extract_text_from_tiff(input_filepath)
        elif file_extension in [".png", ".jpeg"]:
            ocr_text = extract_text_from_image(input_filepath)
        
        filename = os.path.splitext(data.object_name)[0] + ".json"

        # Check if the OCR text is empty or None
        # Save the extracted text in a JSON file  
        logger.info("Saving the OCR extracted information to a JSON file")
        if ocr_text:
            save_ocr_result(filename, ocr_text, "succeeded")
        else:
            save_ocr_result(filename, "", "failed")

        logger.info(f"OCR processing completed for object: {data.object_name}")
        return OCRResponse(message="OCR processing successful")

    except Exception as e:
        logger.error(f"Error processing OCR: {e}")
        filename = os.path.splitext(data.object_name)[0] + ".json"   
        save_ocr_result(filename, "", "failed")
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {e}")
    

def extract_text_from_pdf(pdf_file):
    """Function to extract information from pdf using PyPDF

    Args:
        pdf_file (string): location of the file

    Returns:
        text: extracted description of the pdf information
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        ocr_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            ocr_text += page.extract_text()
            print(ocr_text)
        return ocr_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_tiff(tiff_filepath):
    """Function to extract information from tiff file format using pytesseract

    Args:
        tiff_filepath (string): location of the file

    Returns:
        text: extracted information from the tiff object
    """
    try:
        image = tifffile.imread(tiff_filepath)
        ocr_text = pytesseract.image_to_string(image)
        return ocr_text if ocr_text else None
    except Exception as e:
        logger.error(f"Error extracting text from TIFF: {e}")
        return None

def extract_text_from_image(image_filepath):
    """Function to extract information from images png and jpeg

    Args:
        image_filepath (string): location of the file

    Returns:
        text: extracted information from the png and jpeg object
    """
    try:
        image = cv2.imread(image_filepath)
        ocr_text = pytesseract.image_to_string(image)
        return ocr_text if ocr_text else None
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return None

def save_ocr_result(filename, ocr_text, status):
    """saves the extracted OCR information into a JSON File

    Args:
        filename (str): name of the output json file, similar to the input file name
        ocr_text (str): extracted text from ocr
        status (str): status of the ocr process ("succeeded" or "failed")
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, filename)
    logger.info(f"Saving OCR result to json file: {file_path}")
    data = {"status": status, "analyzeResult": {"content": ocr_text}}
    json_string = json.dumps(data, ensure_ascii=False, indent=2)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_string)
    logger.info("OCR result saved to file")