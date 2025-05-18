"""
Document Intelligence and Search App

This Streamlit application allows users to upload documents which are processed,
chunked, and indexed into Azure AI Search. Users can then ask questions about
the documents and get AI-powered responses based on the document content.
"""

import os
import streamlit as st
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

# Import utilities for file handling and processing
from file_handler import detect_file_type, validate_file
from create_index_from_file import (
    create_index_definition, 
    create_index_from_file,
    create_docs_from_markdown,
    index_client, 
    search_service_endpoint,
    search_api_key
)
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from config import get_logger
from azure.search.documents.indexes.models import SearchIndex
from pdftomarkdown import analyze_documents_output_in_markdown

# For OpenAI functionality
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# For PDF processing
import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = "real-estate-document-index"
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".pptx", ".md", ".txt"]

# Initialize the search client
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(search_api_key)
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Initialize the chat model
chat_model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Ensure this matches your GPT-4 deployment
    model="gpt-4o",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Define our own search function for document retrieval
def search_documents(question: str, top_k: int = 5) -> List[dict]:
    """Search for documents related to a question using Azure AI Search."""
    try:
        # Generate vector embeddings for the query
        query_vector = embeddings.embed_query(question)
        
        # Use Azure AI Search vector search
        from azure.search.documents.models import VectorizedQuery
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="contentVector")
        
        # Perform the search
        search_results = search_client.search(
            search_text=question,
            vector_queries=[vector_query],
            select=["id", "content", "filepath", "title", "url"]
        )
        
        # Format the results
        documents = [
            {
                "id": result["id"],
                "content": result["content"],
                "filepath": result.get("filepath", "Unknown"),
                "title": result.get("title", "Untitled"),
                "url": result.get("url", "")
            }
            for result in search_results
        ]
        
        return documents
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def init_session_state():
    """Initialize session state variables."""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "documents_indexed" not in st.session_state:
        st.session_state.documents_indexed = False
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = ""
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = []

def ensure_index_exists():
    """Ensure the search index exists, create it if it doesn't."""
    try:
        # Check if the index exists
        index_client.get_index(INDEX_NAME)
        logger.info(f"Index '{INDEX_NAME}' already exists")
    except Exception:
        # Create the index if it doesn't exist
        index_definition = create_index_definition(INDEX_NAME, model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002"))
        index_client.create_index(index_definition)
        logger.info(f"Created new index '{INDEX_NAME}'")

def process_file(file):
    """Process an uploaded file and prepare it for indexing."""
    try:
        # Save the file temporarily
        temp_file_path = f"temp_{file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Update processing status
        st.session_state.processing_status = f"Processing {file.name}..."
        
        # Detect file type
        file_type = detect_file_type(temp_file_path)
        valid, error_message, _ = validate_file(temp_file_path)
        
        if not valid:
            st.error(f"Error processing file: {error_message}")
            return None
        
        # Import specific file processing module dynamically based on file type
        if file_type == 'pdf':
            from archive.create_search_index import process_pdf
            document = process_pdf(temp_file_path)
        elif file_type == 'word':
            from archive.create_search_index import process_word
            document = process_word(temp_file_path)
        elif file_type == 'powerpoint':
            from archive.create_search_index import process_powerpoint
            document = process_powerpoint(temp_file_path)
        elif file_type in ['markdown', 'csv']:
            from archive.create_search_index import process_text
            document = process_text(temp_file_path)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        # Update status
        st.session_state.processing_status = f"Document extracted from {file.name}. Chunking..."
        
        # Get the document ready for indexing
        from archive.create_search_index import chunk_and_index_document
        document_info = {
            "path": temp_file_path,
            "title": file.name,
            "content": document.page_content if hasattr(document, 'page_content') else document,
        }
        
        # Index the document
        st.session_state.processing_status = f"Indexing {file.name}..."
        chunk_and_index_document(document_info)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        # Update session state
        st.session_state.documents_indexed = True
        st.session_state.indexed_documents.append({
            "name": file.name,
            "type": file_type,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        st.session_state.processing_status = f"{file.name} processed and indexed successfully!"
        return document_info
        
    except Exception as e:
        logger.error(f"Error processing {file.name}: {e}")
        st.error(f"An error occurred while processing {file.name}: {e}")
        return None

def ask_question(question):
    """Process a user question and get response from the indexed documents."""
    try:
        # Update processing status
        st.session_state.processing_status = "Processing your question..."
        
        # Get documents related to the question using our custom search function
        results = search_documents(question, top_k=5)
        
        if not results:
            st.warning("No relevant information found. Please try rephrasing your question.")
            return
        
        # Prepare the prompt for the AI model
        context = "\n\n".join([doc["content"] for doc in results])
        
        # Use the chat model (initialized earlier)
        response = chat_model.invoke([
            {
                "role": "system", 
                "content": "You are a helpful assistant. Answer the user's question based ONLY on the context provided. "
                           "If the answer is not in the context, say you don't know."
            },
            {
                "role": "user", 
                "content": f"Context: {context}\n\nQuestion: {question}"
            }
        ])
        
        # Save the conversation history
        st.session_state.conversation_history.append({
            "question": question,
            "answer": response.content,
            "references": results
        })
        
        # Clear processing status
        st.session_state.processing_status = ""
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        st.error(f"An error occurred while processing your question: {e}")

def recreate_index():
    """Recreate the search index, removing all indexed documents."""
    try:
        # Delete the existing index
        try:
            index_client.delete_index(INDEX_NAME)
            logger.info(f"Deleted existing index '{INDEX_NAME}'")
        except Exception:
            logger.info(f"No existing index '{INDEX_NAME}' found to delete")
        
        # Create a new index
        index_definition = create_index_definition(INDEX_NAME, model=os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002"))
        index_client.create_index(index_definition)
        logger.info(f"Created new index '{INDEX_NAME}'")
        
        # Reset session state
        st.session_state.documents_indexed = False
        st.session_state.indexed_documents = []
        st.session_state.processing_status = "Search index has been reset. Please upload new documents."
        
        return True
    except Exception as e:
        logger.error(f"Error recreating index: {e}")
        st.error(f"An error occurred while recreating the search index: {e}")
        return False

def main():
    """Main application function."""
    # Set page configuration
    st.set_page_config(
        page_title="Document Intelligence & Search",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Ensure the index exists
    ensure_index_exists()
    
    # Sidebar
    with st.sidebar:
        st.title("Document Intelligence")
        st.subheader("Upload and Search Documents")
        
        # Navigation
        page = st.radio("Navigation", ["Upload Documents", "Ask Questions"])
        
        # Display indexed documents
        if st.session_state.indexed_documents:
            st.subheader("Indexed Documents")
            for doc in st.session_state.indexed_documents:
                st.text(f"📄 {doc['name']} ({doc['type']})")
        
        # Reset index button
        if st.button("Reset Search Index"):
            if recreate_index():
                st.success("Search index has been reset successfully.")
    
    # Main content area
    if page == "Upload Documents":
        st.title("Upload Documents")
        st.write("Upload documents to process and index them for searching.")
        
        uploaded_files = st.file_uploader(
            "Choose documents to upload", 
            accept_multiple_files=True,
            type=["pdf", "docx", "pptx", "txt", "md"]
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    for file in uploaded_files:
                        st.info(f"Processing {file.name}...")
                        process_file(file)
                    st.success("All documents processed successfully!")
        
        # Display processing status if active
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
    
    elif page == "Ask Questions":
        st.title("Ask Questions")
        
        if not st.session_state.documents_indexed:
            st.warning("You need to upload and process documents first.")
        else:
            question = st.text_input("Enter your question about the documents:")
            
            if question:
                if st.button("Get Answer"):
                    with st.spinner("Getting answer..."):
                        ask_question(question)
            
            # Display conversation history
            if st.session_state.conversation_history:
                st.subheader("Conversation History")
                for i, entry in enumerate(st.session_state.conversation_history):
                    with st.expander(f"Q: {entry['question']}", expanded=(i == len(st.session_state.conversation_history) - 1)):
                        st.markdown(f"**Answer:** {entry['answer']}")
                        
                        if entry['references']:
                            st.markdown("**References:**")
                            for ref in entry['references']:
                                st.write(f"- {ref.get('title', 'Untitled document')}")
                                st.write(f"  Source: {ref.get('filepath', 'Unknown')}")
    
    # Display footer
    st.markdown("---")
    st.caption("Document Intelligence & Search App | Built with Azure AI Search, Document Intelligence, OpenAI, and Streamlit")

if __name__ == "__main__":
    main()
