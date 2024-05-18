from pydantic import BaseModel, Field
import openai
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import logging
from config import EMBEDDING_MODEL_ID, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, OPENAI_MODEL_ID
from fastapi import APIRouter

class ExtractRequest(BaseModel):
    query_text: str = Field(..., description="The query text", min_length=1)
    namespace: str = Field(..., description="The Pinecone namespace", min_length=3)

class OCRResponse(BaseModel):
    message: str

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

@router.post('/extract', response_model=OCRResponse, include_in_schema=True) 
def extract_attributes(data: ExtractRequest):
    """
    Extracts attributes based on the given query text and Pinecone namespace.

    Args:
        data (ExtractRequest): Input data containing query text and Pinecone namespace.

    Returns:
        OCRResponse: Response containing the extracted attributes or error message.
    """

    try:
        #read the user input
        query_text = data.query_text
        pc_namespace = data.namespace

        # Validate Pinecone index existence
        logger.info("Validating the Pinecone index existence.")
        pinecone_index = validate_pinecone_index()
        pinecone_index.list

        #validate the Pinecone namespace
        logger.info("Validating the given namespace presence in the Pinecone index.")
        if pc_namespace not in pinecone_index.list_indexes():
            logger.error(f"Pinecone namespace '{pc_namespace}' does not exist in the '{PINECONE_INDEX}' index.")
            raise ValueError(f"Pinecone namespace '{pc_namespace}' does not exist in the '{PINECONE_INDEX}' index.")

        #generate embeddings for the query text
        logger.info("Generating embeddings for the query text and performing vector search.")
        relevant_docs = perform_vector_search(pc_namespace, query_text)

        # Generate response using LangChain's QA chain
        if relevant_docs:
            response = generate_response_from_text(relevant_docs, query_text)
            return OCRResponse(message=response)
        else:
            return OCRResponse(message="No relevant attributes found. Try again with another query")

    except Exception as e:
        logger.error(f"Error occurred during attribute extraction. {e}")
        return OCRResponse(message=f"Error occurred during attribute extraction. {e}")

def perform_vector_search(namespace, query_text):
    """
    Performs vector search using PineconeVectorStore integration with LangChain.

    Args:
        namespace (str): Pinecone namespace.
        query_text (str): Query text.

    Returns:
        list: Relevant texts.
    """

    try:
        #pinecone vectorstore from index
        pinecone_index_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding_model=embedding_model)
        
        #perform vector search using pineconevectorstore integration with langchain
        search_results = pinecone_index_vectorstore.similarity_search(query_text, namespace=namespace, k=5)

        # Extract relevant parts of the file
        relevant_texts = [result['metadata']['text'] for result in search_results]
        return relevant_texts

    except Exception as e:
        logger.error(f"An error occurred during vector search: {str(e)}")
        return None

def generate_response_from_text(relevant_texts, query_text):
    """
    Generates response using LangChain's QA chain.

    Args:
        relevant_texts (list): Relevant texts.
        query_text (str): Query text.

    Returns:
        str: Extracted attributes.
    """

    try:
        llm = OpenAI(model_name=OPENAI_MODEL_ID)
        qa_chain = load_qa_chain(llm, chain_type="map_reduce")

        response = qa_chain.run(input_documents=relevant_texts, question=query_text)
        extracted_attributes = response['answer']
        logger.info(f"Extracted attributes: {extracted_attributes}")
        return extracted_attributes
    except Exception as e:
        logger.error(f"An error occurred during response generation: {str(e)}")
        return None

def validate_pinecone_index():
    """
    Validates the existence of the Pinecone index.
    """
    indexes = pc.list_indexes().get('indexes', [])
    index_names = [index['name'] for index in indexes]

    if not index_names:
        logger.error("No indexes found. Please create a Pinecone index.")
        raise ValueError("No indexes found. Please create a Pinecone index.")

    if PINECONE_INDEX not in index_names:
        logger.error(f"The given '{PINECONE_INDEX}' is not present in the DB. Provide a valid Pinecone index.")
        raise ValueError(f"The given '{PINECONE_INDEX}' is not present in the DB. Provide a valid Pinecone index.")
    else:
        logger.info(f"Pinecone index '{PINECONE_INDEX}' is present in the DB.")
        pinecone_index = pc.Index(PINECONE_INDEX)
        return pinecone_index
