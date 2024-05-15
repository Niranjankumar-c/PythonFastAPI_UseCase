import os
from dotenv import load_dotenv

load_dotenv() #load the .env variables

#set the config parameters for minio
MINIO_ENDPOINT = "play.min.io:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = "my-bucket2"

# Define the output directory for saving files
OUTPUT_DIR = os.path.join(os.getcwd(), "ocr_results")

#pinecone parameters
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

#openai parameters
EMBEDDING_MODEL_ID = os.getenv("OPENAI_EMBEDDING_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")