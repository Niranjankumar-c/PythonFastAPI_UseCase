from pydantic import BaseModel, Field
import openai
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
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
openai_client = OpenAI()

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

        #validate the Pinecone namespace
        logger.info("Validating the given namespace exists in the Pinecone index.")
        validate_pinecone_namespace(pinecone_index, pc_namespace)

        #generate embeddings for the query text
        logger.info("Generating embeddings for the query text and performing vector search.")
        relevant_docs = perform_similarity_search(pinecone_index, pc_namespace, query_text)

        # Generate response using LangChain's QA chain
        if relevant_docs:
            response = generate_response_from_chatapi(relevant_docs, query_text)
            logger.info("Sucessfully generated response to the user query using OpenAI API.")
            return OCRResponse(message=response)
        else:
            return OCRResponse(message="An error occurred during response generation, coun't find any results matching your query. Try again with another query")

    except Exception as e:
        logger.error(f"Error occurred during attribute extraction. {e}")
        return OCRResponse(message=f"Error occurred during attribute extraction. {e}")

def perform_similarity_search(pc_index, pc_namespace, query_text):
    """
    Performs similarity search using PineconeVectorStore integration with LangChain.

    Args:
        namespace (str): Pinecone namespace.
        query_text (str): Query text.

    Returns:
        list: Relevant texts.
    """

    try:

        #create embeddings for the input query text
        logger.info(f"Creating embeddings for the query text")
        embed = embedding_model.embed_documents([query_text])
        logger.info(f"Peform similarity search on the index")
        res = pc_index.query(vector=embed, top_k=3, include_metadata=True, namespace=pc_namespace)

        #get the context
        contexts = res['matches'][0]['metadata']['text'] if res['matches'][0]['metadata']['text'] else ""

        return contexts

    except Exception as e:
        logger.error(f"An error occurred during vector search: {str(e)}")
        return None

def generate_response_from_chatapi(relevant_texts, query_text):
    """
    Generates response using LangChain's QA chain.

    Args:
        relevant_texts (list): Relevant texts.
        query_text (str): Query text.

    Returns:
        str: Extracted attributes.
    """

    try:

        #create a prompt template 
        messages = [
            {"role": "system", "content": """You are a helpful assistant. You are required to answer the user query question in form of bullet points based on the provided context. The answer should be in English unless specified in the query. 
            You shall not stray away from the context and only provide information which is derived from the context. Refer to the block of given text below: \n""" + relevant_texts},
            {"role": "user", "content": "Question: " + query_text}]

        result = openai_client.chat.completions.create(
            model=OPENAI_MODEL_ID,
            messages=messages,
            temperature=0.3,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        
        model_response = result.choices[0].message.content
        return model_response
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

def validate_pinecone_namespace(pinecone_index, namespace):
    """
    Checks if the Pinecone namespace is present within the specified index.

    Args:
        pinecone_index: Pinecone index object.
        pc_namespace (str): Pinecone namespace.

    Raises:
        ValueError: If the namespace does not exist in the index.
    """
    index_stats = pinecone_index.describe_index_stats()
    namespaces = index_stats.get("namespaces", {})

    if namespace not in namespaces:
        logger.error(f"Pinecone namespace '{namespace}' does not exist in the '{PINECONE_INDEX}' index.")
        raise ValueError(f"Pinecone namespace '{namespace}' does not exist in the '{PINECONE_INDEX}' index.")
    else:
        logger.info(f"Pinecone namespace '{namespace}' exists in the '{PINECONE_INDEX}' index.")