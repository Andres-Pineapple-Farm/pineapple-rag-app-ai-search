import os
from langchain_openai import AzureOpenAIEmbeddings
from config import get_logger

logger = get_logger(__name__)

def test_embedding():
    try:
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

        # Test embedding using the embed_query method
        test_text = "This is a test document to check if embeddings work."
        embedding_vector = embeddings.embed_query(test_text)
        
        print(f"Embedding generated successfully!")
        print(f"Vector dimension: {len(embedding_vector)}")
        print(f"First few values: {embedding_vector[:5]}...")
        
        return True
    except Exception as e:
        logger.error(f"Error testing embeddings: {e}")
        print(f"Error testing embeddings: {e}")
        return False

if __name__ == "__main__":
    print("Testing AzureOpenAIEmbeddings...")
    success = test_embedding()
    if success:
        print("Embedding test passed successfully!")
    else:
        print("Embedding test failed.")
