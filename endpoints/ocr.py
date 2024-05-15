"""
ocr.py: The file used to implement the OCR endpoint

"""
#import required packages
import os, time
import logging
import pytesseract, openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
import PyPDF2, tifffile
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, OUTPUT_DIR, EMBEDDING_MODEL_ID, OPENAI_API_KEY, PINECONE_API_KEY
from minio import Minio
import random, string, cv2, json

# Configure OpenAI api
openai.api_key = OPENAI_API_KEY

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_ID)

#define the router
router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/ocr.log', format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# configure pinecone api
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define request model
class OCRRequest(BaseModel):
    bucket_name: str = "my-bucket2"
    object_name: str

# Define response model
class OCRResponse(BaseModel):
    message: str

# Initialize Minio client
minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)

@router.post("/ocr", response_model=OCRResponse, include_in_schema=False)
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
    
@router.post("/embeddings", response_model=OCRResponse, include_in_schema=True)
def create_embeddings(data: OCRRequest):

    try:
        logger.info(f"Creating embeddings for object: {data.object_name}")

        #read the json from the ocr function 
        input_filepath = os.path.join(OUTPUT_DIR, data.object_name.split('.')[0] + ".json")

        logger.info(f"Reading extracted text from the json file: {input_filepath}")
        with open(input_filepath, "r", encoding="utf8") as file:
            ocr_data = json.load(file)
        
        if ocr_data["status"] == "succeeded":
            ocr_text = ocr_data["analyzeResult"]["content"]
            
            #split the text into multiple chunks for creating embeddings
            logger.info(f"Splitting the extracted text into multiple chunks")
            chunks = split_text_into_chunks(ocr_text)
            
            if chunks:
                logger.info("Creating embeddings for the OCR text using OpenAI")
                embedding = get_embedding(chunks)

                if embedding:
                    logger.info("Uploading embeddings to Pinecone")
                    namespace = f"{data.object_name.split('.')[0]}_{generate_unique_number()}"
                    upload_embeddings_to_pinecone(embedding, namespace)
                    
                    logger.info(f"Embeddings processing completed for object: {data.object_name}")
                    return OCRResponse(message=f"Embeddings processing successful. Namespace: {namespace}")
            else:
                error_msg = f"Error splitting text into chunks for object: {data.object_name}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)
        else:
            error_msg = f"OCR extraction status for the file : {data.object_name} has failed, try embeddings creation with valid file"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {e}")
    
# OCR-Embeddings function
@router.post("/ocr-embeddings", response_model=OCRResponse, include_in_schema=False)
def ocr_embeddings_process(data: OCRRequest):
    try:
        ocr_response = ocr_process(data)
        logger.info(f"OCR processing completed for object: {data.object_name}")
    except Exception as e:
        logger.error(f"Error processing OCR: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {e}")

    try:
        # embeddings_response = embeddings_process(data)
        logger.info(f"Embedding processing completed for object: {data.object_name}")
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {e}")

    return OCRResponse(message="OCR and Embeddings processing successful")
    
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

def get_embedding(text_to_embed):
    """Create embeddings from the chunks of the text

    Args:
        text_to_embed (list): list of chunks

    Returns:
        list: list of embeddings for the data
    """

    try:

        max_retries = 5
        base_delay = 2  # Initial delay in seconds

        for attempt in range(max_retries):

            try:
                # Create a list to store the embeddings
                all_embeddings = []

                # Process each chunk
                for chunk in tqdm(text_to_embed):
                    # Embed the chunk using OpenAIEmbeddings
                    chunk_embeddings = embeddings.embed_documents([chunk])
                    # add all the embeddings to the list
                    all_embeddings.extend(chunk_embeddings)
                    #add a small time delay
                    time.sleep(2)

                return all_embeddings
            except OpenAIEmbeddings.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise e
                delay = base_delay * (2 ** attempt)
                print(f"Rate limit reached. Retrying in {delay} seconds...")
                time.sleep(delay)

    except Exception as e:
        logger.error(f"Error while creating embeddings: {e}")
        return None


def split_text_into_chunks(text):
    """Split text into chunks using RecursiveCharacterTextSplitter from langchain

    Args:
        text (str): text for splitting into chunks

    Returns:
        list: list of chunks
    """
    try:
        # Initialize RecursiveCharacterTextSplitter with Japanese-specific separators

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "、", "「", "」", "【", "】", "（", "）", " ", ""],
            chunk_size=2000,
            chunk_overlap=300,
            length_function = len)

        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error while splitting text into multiple chunks: {e}")
        return None
    
def upload_embeddings_to_pinecone(embeddings, namespace):
    """upload embeddings to the pinecone vector database

    Args:
        embeddings (list): list of embeddings to be uploaded
        namespace (str): namespace of the index

    Raises:
        HTTPException: Upload error when failed to upload
    """
    try:
        index_name = "FASTAPI_usecase" #name of the index - pinecone

        #check if the index is present in the pinecone db else will create the index
        try:
            index = pc.Index(name=index_name, namespace=namespace)
        except pc.errors.IndexNotFound:
            logger.info(f"Index '{index_name}' not found. Creating a new index...")
            index = pc.Index.create(name=index_name, namespace=namespace, dimension=len(embeddings[0]))
            logger.info(f"Index '{index_name}' created successfully.")
        
        #
        index.upsert(vectors=[{"id": str(i), "values": embedding} for i, embedding in enumerate(embeddings)])
    except Exception as e:
        logger.error(f"Error uploading embeddings to Pinecone: {e}")
        raise HTTPException(status_code=500, detail="Error uploading embeddings to Pinecone")
    
def generate_unique_number():
    return ''.join(random.choices(string.digits, k=5))
