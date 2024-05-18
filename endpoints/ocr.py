"""
ocr.py: The file used to implement the OCR endpoint

"""
#import required packages
import os, time
import logging
import pytesseract, openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
import PyPDF2, tifffile
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, OUTPUT_DIR, EMBEDDING_MODEL_ID, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX
from minio import Minio
import cv2, json, uuid

# Configure OpenAI api
openai.api_key = OPENAI_API_KEY

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_ID)

# configure pinecone api
pc = Pinecone(api_key=PINECONE_API_KEY)

#define the router
router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class OCRRequest(BaseModel):
    bucket_name: str = "my-bucket2"
    object_name: str

# Define response model
class OCRResponse(BaseModel):
    message: str

# Initialize Minio client
minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=True)

@router.post("/ocr-extraction", include_in_schema=False)
def ocr_process(data: OCRRequest):
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
        logger.info(f"Initializing OCR process request for object: {data.object_name}")
        
        file_extension = os.path.splitext(data.object_name)[1].lower()
        logger.info("Checking if the input file format is valid")
        if file_extension not in allowed_formats:
            error_msg = f"Unsupported file format: {file_extension}. Only PDF, TIFF, PNG, JPEG formats are allowed."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Fetching the file from MinIO using bucket name and object name")
        input_filepath = download_file_from_minio(data)
        
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
        if ocr_text:
            filename = os.path.splitext(data.object_name)[0] + ".json"
            save_ocr_result(filename, ocr_text, "succeeded")
            logger.info(f"OCR processing completed for object: {data.object_name}")
            return OCRResponse(message="OCR processing successful")
        else:
            filename = os.path.splitext(data.object_name)[0] + ".json"
            save_ocr_result(filename, "", "failed")
            logger.error(f"OCR processing failed for object: {data.object_name}")
            return OCRResponse(message="OCR processing failed")

    except Exception as e:
        filename = os.path.splitext(data.object_name)[0] + ".json"   
        save_ocr_result(filename, "", "failed")
        logger.error(f"Error processing OCR: {e}")
        raise Exception(f"Error processing OCR: {e}")
    
@router.post("/embeddings", include_in_schema=False)
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
                    
                    #upload embeddings to the pinecone vectordatabase
                    embeddings_result = upload_embeddings_to_pinecone(embedding, data.object_name)
                    
                    logger.info(f"Embeddings processing completed for object: {data.object_name} and stored in namespace: {embeddings_result['namespace']}" )
                    return OCRResponse(message=f"Embeddings processing successful for object: {data.object_name} and stored in namespace: {embeddings_result['namespace']}")
            else:
                logger.warning(f"No chunks generated for object: {data.object_name}. Skipping embeddings creation.")
                return OCRResponse(message=f"No chunks generated for object: {data.object_name}. Skipping embeddings creation.")
        else:
            logger.warning(f"OCR extraction status for the file: {data.object_name} is {ocr_data['status']}. Skipping embeddings creation.")
            return OCRResponse(message=f"OCR extraction status for the file: {data.object_name} is '{ocr_data['status']}'. Skipping embeddings creation.")

    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        raise Exception(f"Error processing embeddings: {e}")
    
# OCR-Embeddings function
@router.post("/ocr", response_model=OCRResponse, include_in_schema=True)
def ocr_embeddings_process(data: OCRRequest):

    try:
        ocr_response = ocr_process(data)
        print(ocr_response)
    except Exception as e:
        logger.error(f"Error processing OCR: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing OCR: {e}")

    try:
        embeddings_response = create_embeddings(data)
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {e}")
    
    return OCRResponse(message=embeddings_response.message)
    
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
    if not ocr_text:  # Check if OCR text is empty
        logger.warning("OCR text is empty. Skipping saving OCR result.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, filename)
    logger.info(f"Saving OCR result to JSON file: {file_path}")
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
            except Exception as rate_error:
                if isinstance(rate_error, openai.RateLimitError):
                    if attempt == max_retries - 1:
                        logger.error(f"Rate limit exhausted after {max_retries} retries: {rate_error}")
                        raise rate_error
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
    
def upload_embeddings_to_pinecone(embeddings, data_filename):
    """upload embeddings to the pinecone vector database

    Args:
        embeddings (list): list of embeddings to be uploaded
        namespace (str): namespace of the index

    Raises:
        HTTPException: Upload error when failed to upload
    """
    try:
        index_name = PINECONE_INDEX #name of the index - pinecone
        dimension_length = len(embeddings[0])  #get the dimensions to create the index
        namespace = generate_unique_file_id()
        
        try:
            #check if the index is present in the pinecone db else will create the index
            logger.info(f"Fetching the Index '{index_name}' in the pinecone db.")
            if index_name not in pc.list_indexes().names():
                logger.info(f"Index '{index_name}' not found. Creating a new index...")
                pc.create_index(
                name=index_name,
                dimension=dimension_length,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            )
                logger.info(f"Index '{index_name}' created successfully.") 
        except Exception as index_error:
            logger.error(f"Error creating {index_name} in pinecone with dimensions: {dimension_length}: {index_error}")
            raise index_error
        
        
        # Get the index
        index = pc.Index(index_name)

        # Upload embeddings to the index
        index.upsert(vectors=[{"id": str(i), "values": embedding} for i, embedding in enumerate(embeddings)], namespace=namespace)

        # wait for index to be initialized  
        while not pc.describe_index(index_name).status['ready']:  
            logger.info(f"Pinecone: Wait for the index to be initialized")
            time.sleep(2) 

        # logger.info(f"Index stats:  {index.describe_index_stats()}")
        logger.info(f"Embeddings uploaded successfully for object: {data_filename}")
        return {"object_name": data_filename, "namespace": namespace, "success": True}
    except Exception as upload_error:
        logger.error(f"Error uploading embeddings to Pinecone: {upload_error}")
        raise upload_error

def generate_unique_file_id():
    """Function to generate unique file id

    Returns:
        string: returns the hex code of the unique file identifier
    """
    return uuid.uuid4().hex

def download_file_from_minio(data: OCRRequest):
    """Downloads a file from Minio based on the given bucket and file name

    Args:
        data (OCRRequest): OCRRequest object containing bucket and file name

    Raises:
        HTTPException: If the bucket or object does not exist or if the downloaded file is empty

    Returns:
        str: filepath of download file
    """
    try:
        # Check if the bucket exists
        if not minio_client.bucket_exists(data.bucket_name):
            logger.error(f"Bucket '{data.bucket_name}' does not exist")
            raise HTTPException(status_code=404, detail=f"Bucket '{data.bucket_name}' does not exist")
        else:
            logger.info(f"Bucket '{data.bucket_name}' exist")
        
        # Check if the object exists
        try:
            stat = minio_client.stat_object(data.bucket_name, data.object_name)
        except Exception as e:
            logger.error(f"Object '{data.object_name}' does not exist in bucket '{data.bucket_name}'")
            raise HTTPException(status_code=404, detail=f"Object '{data.object_name}' does not exist in bucket '{data.bucket_name}'")
            
        try:
            # Fetch the file from MinIO using the bucket_name and object_name
            logger.info("Fetching the file from MinIO using bucket name and object name")
            input_filepath = os.path.join(OUTPUT_DIR, data.object_name)
            minio_client.fget_object(data.bucket_name, data.object_name, input_filepath)
            time.sleep(1) #time for file download
        except:
            logger.error("Error while fetching the object from the Minio bucket, please try again")
            raise HTTPException(status_code=400, detail="Failed to fetch the file from the given bucket, please try again")
        
        # Check if the downloaded file is valid
        if os.path.getsize(input_filepath) == 0:
            logger.error(f"Downloaded file '{data.object_name}' is empty")
            raise HTTPException(status_code=400, detail=f"Downloaded file '{data.object_name}' is empty")
        
        return input_filepath
    except Exception as e:
        logger.error(f"Failed to fetch the file from the given bucket: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch the file from the given bucket: {e}")