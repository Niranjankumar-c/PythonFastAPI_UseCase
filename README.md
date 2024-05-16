# PythonFastAPI_UseCase
**Objective**: Build a Python backend API using FastAPI for a document processing application.

**Functionalities**:
- File Upload: Allow users to upload documents for processing.
- Mock OCR: Simulate the process of extracting text from the uploaded document and create embeddings from the text & upload them to the pinecone vector database.
- Attribute Extraction: Utilize Retrieval-Augmented Generation (RAG) to extract relevant attributes from the embeddings stored in pinecone.

## Project Structure
```
root/
├── README.md
├── requirements.txt
├── __init__.py
├── .env
├── config.py
├── main.py
├── endpoints/
│   ├── __init__.py
│   ├── ocr.py
│   ├── upload_files.py
│   └── extract.py
├── tests/
│   ├── test_upload.py
│   ├── test_ocr.py
│   └── test_extract.py
├── ocr_results/
├── data/
│   ├── ocr
│   ├── sample_inputdata
```

- **main.py**: Serves as the entry point for the application, initializing FastAPI and configuring dependencies.
- **.env**: This file stores sensitive information like API keys and connection details (not commited to git).
- **config.py**: Contains the different configuration details for connecting to minio, openai and pinecone.
- **endpoints/**: Directory containing endpoint implementations.
- **tests/**: Directory containing the test cases for the endpoint implementations
- **requirements.txt**: List of Python dependencies

## Getting Started

### Prerequisites
- Ensure you have Python 3.9 or later installed

### Installation
The Installation process will get you a copy of the project up and running on your local machine.
1. Create a virtual environemnt
   - It is recommended to create a virtual environment to isolate project dependencies using ```venv``` and ```conda```
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ocr-embeddings-project.git
   cd ocr-embeddings-project
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

### Configure .env
- Set up all the sensitive information like API keys and connection details into .env file.
```
MINIO_ACCESS_KEY="Q3AM3UQ867SPQQA43P2F"
MINIO_SECRET_KEY="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG"
OPENAI_API_KEY=YOUR_API_KEY
PINECONE_API_KEY=YOUR_API_KEY
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```
## Running the APP
1. Start the FastAPI server:
   - Run FastAPI application using uvicorn and adjust the host and port options as needed
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
2. Access the API at `http://localhost:8000`

## EndPoints

### File Upload Endpoint
Path: **`/upload`**<br />
Method: **POST**<br />
Functionality:
- Accepts one or more file uploads (limited to pdf, tiff, png,jpeg formats).

### MOCK OCR Service EndPoint
Path: **`/ocr`**(In-Progress)<br />
Method: **POST**<br />
Functionality:
- Simulating the operation of an OCR service on a file and generating embeddings for uploading to a vector database.

### Attribute Extraction Endpoint
Path: **`/extract`**(Yet to Implement)<br />
Method: **POST**<br />
Functionality:
- Takes a query text and file_id as input, performs a vector search and returns matching attributes based on the embeddings.

## Testing the Application Functionality
Start the FastAPI server by running the main.py<br />
### Postman or CURL
- Use tools like Postman or curl to send a POST request to `http://localhost:8000/upload` with one or more files in the request body.
  - For example, using curl with a file named "test_file.pdf": `curl -X POST http://localhost:8000/upload -F "files=@test_file.pdf"`
  - The response should be a JSON object containing a list of uploaded file IDs if successful.
### Interactive FASTAPI Docs
- After starting the server go to `http://127.0.0.1:8000/docs`, You will see the automatic interactive API documentation provided by Swagger UI.
![image](https://github.com/Niranjankumar-c/PythonFastAPI_UseCase/assets/36450263/31a6b5db-9ec8-4e8e-8b26-cc1ee9ba24b3)
