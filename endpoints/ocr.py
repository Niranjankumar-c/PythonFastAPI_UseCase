"""
ocr.py: The file used to implement the OCR endpoint

"""
#import required packages
import os, time
import logging
import pytesseract, openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tqdm import tqdm
import PyPDF2, tifffile
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, OUTPUT_DIR, EMBEDDING_MODEL_ID, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX
from minio import Minio
import cv2, json, uuid
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np

# Configure OpenAI api
openai.api_key = OPENAI_API_KEY

# Initialize OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_ID)

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
                # logger.info("Creating embeddings for the OCR text using OpenAI")
                logger.info("Creating embeddings using OpenAI and uploading them to pinecone database")
                embeddings_result = upload_embeddings_to_pinecone(chunks, data.object_name)
                    
                logger.info(f"Embeddings processing completed for object: {data.object_name} and stored in namespace: {embeddings_result['namespace']}" )
                return OCRResponse(message=f"Embeddings processing successful for object: {data.object_name} and stored in namespace: {embeddings_result['namespace']}")
            else:
                logger.warning(f"No chunks generated for object: {data.object_name}. Skipping embeddings creation.")
                return OCRResponse(message=f"No chunks generated for object: {data.object_name}. Skipping embeddings creation.")
        else:
            logger.warning(f"OCR extraction status for the file: {data.object_name} is {ocr_data['status']}. Skipping embeddings creation.")
            return OCRResponse(message=f"OCR extraction status for the file: {data.object_name} is '{ocr_data['status']}'. Skipping embeddings creation.")

    except Exception as e:
        logger.error(f"Error occurred while creating and uploading embeddings: {e}")
        raise Exception(f"Error occurred while creating and uploading embeddings: {e}")
    
# OCR-Embeddings function
@router.post("/ocr", response_model=OCRResponse, include_in_schema=True)
def ocr_embeddings_process(data: OCRRequest):

    try:
        ocr_response = ocr_process(data)
        print(ocr_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")

    try:
        embeddings_response = create_embeddings(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    
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

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def get_embeddings_from_openai(text_to_embed: list[str]):
    """
    Generate embeddings for a list of texts/chunks using openai

    Args:
        text_to_embed (list): A list of texts to be embedded.

    Returns:
        list: A list of embeddings generated for the input texts.
    """
    try:
        all_embeddings = []  # List to store all embeddings
        
        logger.info(f"Generating embeddings for {len(text_to_embed)} texts using model '{EMBEDDING_MODEL_ID}'...")
        
        # Process each text chunk
        for index, text in enumerate(text_to_embed, start=1):
            logger.info(f"Processing text {index}/{len(text_to_embed)} to create embeddings...")

            # Embed the chunk using OpenAIEmbeddings
            chunk_embeddings = embedding_model.embed_documents([text])

            # add all the embeddings to the list
            all_embeddings.extend(chunk_embeddings)

            logger.info(f"Embedding generated for text {index}/{len(text_to_embed)}")
            
        logger.info("Embeddings generation completed.")
        return all_embeddings  # Return the list of all embeddings
    
    except Exception as e:
        # Log the error and retry
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
    
def upload_embeddings_to_pinecone(chunks_lst, data_filename):
    """Create embeddings for the list of texts and upload embeddings to the pinecone vector database

    Args:
        chunks_lst (list): list of embeddings to be uploaded
        data_filename (str): name of the input file

    Raises:
        Exception: Raise exception when embeddings failed to upload
    """
    try:
        
        index_name = PINECONE_INDEX #name of the index - pinecone
        dimension_length = len(embedding_model.embed_documents(["GET EMBEDDING LENGTH"])[0])  #get the dimensions to create the index
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
        #loop through all the chunks, create embeddings and upload them to the pinecone index
        batch_size = 50 #set a small batch size to avoid openai and pinecone rate errors

        logger.info("Iterating though each batch of chunks for creating embeddings") 
        for i in tqdm(range(0, len(chunks_lst), batch_size)):
            batch = chunks_lst[i:i + batch_size]
            
            # Prepare metadata and embeddings for the batch
            metadatas = [{'text': chunk} for chunk in batch]
            ids = [str(x) for x in range(i, i + len(batch))]
            
            # Get embeddings for the batch
            embeds = get_embeddings_from_openai(batch)

            if embeds:
                # Upsert into Pinecone index
                index.upsert(vectors=list(zip(ids, np.array(embeds), metadatas)), namespace=namespace)
                logger.info(f"Batch {i//batch_size + 1} upserted successfully.")
            else:
                raise ValueError("Failed to create embeddings, empty list returned")

        # wait for index to be initialized  
        while not pc.describe_index(index_name).status['ready']:  
            logger.info(f"Pinecone: Wait for the index to be initialized")
            time.sleep(1) 

        logger.info(f"Embeddings uploaded successfully for object: {data_filename}")
        return {"object_name": data_filename, "namespace": namespace}
    except Exception as upload_error:
        logger.error(f"Error uploading embeddings to Pinecone: {upload_error}")
        raise upload_error

def generate_unique_file_id():
    """Function to generate unique file id in the format filenum_xxxxxxxx

    Returns:
        string: returns the unique file identifier in the format filenum_xxxxxxxx
    """
    # Generate a UUID and take the first 8 characters of its hexadecimal representation
    unique_id = uuid.uuid4().hex[:8]
    return f"filenum_{unique_id}"

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
        raise