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

# We'll define our own search function to avoid dependency issues
# instead of importing from get_documents

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
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".pptx", ".md", ".txt", ".csv"]

# Initialize the search client
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(search_api_key)
)

# Initialize embeddings
from langchain_openai import AzureOpenAIEmbeddings
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Initialize the chat model
from langchain_openai import AzureChatOpenAI
chat_model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Ensure this matches your GPT-4 deployment
    model="gpt-4o",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Define our own search function to replace get_product_documents
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
    if "index_name" not in st.session_state:
        st.session_state.index_name = INDEX_NAME

def ensure_index_exists():
    """Ensure the search index exists, create it if it doesn't."""
    try:
        # Check if the index exists
        index_client.get_index(INDEX_NAME)
        logger.info(f"Index '{INDEX_NAME}' already exists")
        st.session_state.documents_indexed = True
    except Exception:
        # Create the index if it doesn't exist
        index_definition = create_index_definition(INDEX_NAME, model=embeddings.model)
        index_client.create_index(index_definition)
        logger.info(f"Created new index '{INDEX_NAME}'")

def is_image_based_pdf(file_path):
    """Check if a PDF is an image-based PDF that requires OCR processing."""
    try:
        # Attempt to extract text using pdfminer
        text = extract_text(file_path)
        if text.strip():
            return False  # Native PDF with extractable text
        else:
            return True  # Image-based PDF (no extractable text)
    except PDFSyntaxError:
        return True  # Likely an image-based PDF

def process_pdf_document(file_path):
    """Process a PDF document and convert to markdown for indexing."""
    # Check if this is an image-based PDF that needs OCR
    if is_image_based_pdf(file_path):
        st.info(f"Processing image-based PDF using Document Intelligence...")
        
        # Get Document Intelligence credentials
        endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")
        
        if not endpoint or not key:
            st.error("Document Intelligence credentials not found in environment variables")
            return None
        
        # Use Azure Document Intelligence to convert image PDF to markdown
        connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
        container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        
        if not connection_string or not container_name:
            st.error("Azure Blob Storage credentials not found in environment variables")
            return None
          # Convert to markdown using Document Intelligence
        markdown_content = analyze_documents_output_in_markdown(
            endpoint, key, file_path, connection_string, container_name
        )
        
        # Write the markdown content to a temporary file
        temp_md_path = os.path.join(tempfile.gettempdir(), f"{os.path.basename(file_path)}.md")
        with open(temp_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)
            
        st.success(f"Created markdown file: {temp_md_path}")
        return temp_md_path
    else:
        st.info(f"Processing native PDF with extractable text...")
        
        # Extract text from native PDF
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages = []
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    pages.append(f"## Page {i+1}\n{page_text}")
            
            full_text = "\n\n".join(pages)
        
        # Create a temporary markdown file
        temp_md_path = f"{file_path}.md"
        with open(temp_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(f"# Converted PDF Document\n\n{full_text}")
        
        return temp_md_path

def process_file(file):
    """Process an uploaded file and prepare it for indexing."""
    try:
        # Save the file temporarily
        temp_file_path = os.path.join(tempfile.gettempdir(), file.name)
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
        
        # Process based on file type
        if file_type == 'pdf':
            # Process PDF and get markdown path
            markdown_path = process_pdf_document(temp_file_path)
            if not markdown_path:
                st.error("Failed to process PDF document")
                return None
            
            # Update the file type to markdown for indexing
            file_type = 'markdown'
        elif file_type in ['markdown', 'csv']:
            markdown_path = temp_file_path
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        # Update status
        st.session_state.processing_status = f"Document extracted. Creating index..."
          # Create or update search index
        try:
            # Verify the markdown file exists and is readable
            if not os.path.exists(markdown_path):
                raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
                
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Successfully read markdown file ({len(content)} chars)")
            
            # Create or update the index
            create_index_from_file(INDEX_NAME, markdown_path, file_type)
            
            # Update session state
            st.session_state.documents_indexed = True
            st.session_state.indexed_documents.append({
                "name": file.name,
                "type": file_type,
                "time": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.success(f"{file.name} processed and indexed successfully!")
            st.session_state.processing_status = f"{file.name} processed and indexed successfully!"
            return True
            
        except FileNotFoundError as e:
            logger.error(f"File error: {e}")
            st.error(f"File error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            st.error(f"Error creating index: {e}")
            # Show additional details for debugging
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return False
              except Exception as e:
        logger.error(f"Error processing {file.name}: {e}")
        st.error(f"An error occurred while processing {file.name}: {e}")
        st.error(f"Details: {str(e)}")
        return None
    finally:
        # Clean up temporary files
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                logger.info(f"Cleaning up temp file: {temp_file_path}")
                os.remove(temp_file_path)
                
            # Don't remove the markdown file immediately as it's needed for indexing
            # We'll rely on the OS to clean up the temp directory periodically
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

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
    
    # Main page title
    st.title("Document Intelligence & Search")
    st.subheader("Upload, process, and query your documents with Azure AI")
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        # Navigation
        page = st.radio("Select Option", ["Upload Documents", "Ask Questions"])
        
        # Display indexed documents
        if st.session_state.indexed_documents:
            st.subheader("Indexed Documents")
            for doc in st.session_state.indexed_documents:
                st.text(f"📄 {doc['name']} ({doc['time']})")
        
        # Reset index button
        if st.button("Reset Search Index"):
            if recreate_index():
                st.success("Search index has been reset successfully.")
    
    # Display processing status if active
    if st.session_state.processing_status:
        st.info(st.session_state.processing_status)
    
    # Main content area
    if page == "Upload Documents":
        st.header("Upload Documents")
        st.write("Upload documents to process and index them for searching.")
        
        # File uploader
        st.write("Supported file types: PDF, Markdown, CSV")
        uploaded_files = st.file_uploader(
            "Choose documents to upload", 
            accept_multiple_files=True,
            type=["pdf", "md", "csv"]
        )
        
        if uploaded_files:
            if st.button("Process & Index Documents"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    progress = (i) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    result = process_file(file)
                    
                    if result:
                        st.success(f"✅ {file.name} processed and indexed successfully!")
                    else:
                        st.error(f"❌ Failed to process {file.name}")
                
                progress_bar.progress(1.0)
                status_text.text("All documents processed!")
                st.success(f"Processed and indexed {len(uploaded_files)} documents.")
    
    elif page == "Ask Questions":
        st.header("Ask Questions About Your Documents")
        
        if not st.session_state.documents_indexed:
            st.warning("You need to upload and process documents first before asking questions.")
        else:
            question = st.text_input("Enter your question about the documents:")
            
            if st.button("Get Answer") and question:
                with st.spinner("Searching documents and generating answer..."):
                    ask_question(question)
            
            # Display conversation history
            if st.session_state.conversation_history:
                st.subheader("Conversation History")
                
                for i, entry in enumerate(reversed(st.session_state.conversation_history)):
                    with st.expander(f"Q: {entry['question']}", expanded=(i == 0)):
                        st.markdown(f"**Answer:** {entry['answer']}")
                        
                        st.markdown("**Sources:**")
                        for j, ref in enumerate(entry['references']):
                            source = ref.get('filepath', 'Unknown source')
                            title = ref.get('title', 'Untitled section')
                            st.markdown(f"**{j+1}.** {title} - *{source}*")
    
    # Display footer
    st.markdown("---")
    st.caption("Document Intelligence & Search App | Built with Azure AI Search, Document Intelligence, OpenAI, and Streamlit")

if __name__ == "__main__":
    main()
