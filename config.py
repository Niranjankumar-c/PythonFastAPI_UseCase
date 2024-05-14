import os

#set the config parameters
MINIO_ENDPOINT = "play.min.io:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET_NAME = "my-bucket2"

# Define the output directory for saving files
OUTPUT_DIR = os.path.join(os.getcwd(), "ocr_results")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_EN")