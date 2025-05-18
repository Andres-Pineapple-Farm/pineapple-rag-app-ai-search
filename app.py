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
import uuid
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

# Import the improved PDF processor
from pdf_processor import process_pdf, is_image_based_pdf

# For PDF processing (fallback)
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

def process_file(file):
    """Process an uploaded file and prepare it for indexing."""
    temp_file_path = None
    markdown_path = None
    
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
            # Get Document Intelligence credentials
            endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
            key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")
            connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
            container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

            # Process PDF using our improved processor
            st.info(f"Processing PDF document...")
            markdown_path = process_pdf(
                temp_file_path,
                endpoint=endpoint,
                key=key,
                connection_string=connection_string,
                container_name=container_name
            )
            
            if not markdown_path:
                st.error("Failed to process PDF document")
                return None
            
            # Update the file type to markdown for indexing
            file_type = 'markdown'
            st.success("PDF processed successfully!")
            
        elif file_type == 'word':
            # For Word documents, pass directly to the word processor
            st.info(f"Processing Word document...")
            # The actual processing happens in create_index_from_file
            markdown_path = temp_file_path
            st.success("Word document ready for processing!")
            
        elif file_type == 'powerpoint':
            # For PowerPoint documents, pass directly to the powerpoint processor
            st.info(f"Processing PowerPoint presentation...")
            # The actual processing happens in create_index_from_file
            markdown_path = temp_file_path
            st.success("PowerPoint presentation ready for processing!")
            
        elif file_type in ['markdown', 'csv']:
            # For markdown and CSV files, just use the temp file as is
            markdown_path = temp_file_path
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        # Update status
        st.session_state.processing_status = f"Document extracted. Creating index..."
          # Create or update search index
        try:
            # Verify the markdown file exists
            if not os.path.exists(markdown_path):
                raise FileNotFoundError(f"File not found: {markdown_path}")
            
            # For text-based files, verify they are readable
            # Skip this check for binary files (Word, PowerPoint)
            if file_type in ['markdown', 'csv']:
                try:
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.info(f"Successfully read text file ({len(content)} chars)")
                except UnicodeDecodeError:
                    logger.warning(f"File {markdown_path} could not be read as text, but continuing as it might be binary")
            else:
                # For binary files, just log that we're proceeding
                logger.info(f"Processing binary file: {markdown_path}")
            
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
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Clean up temporary files
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                logger.info(f"Cleaning up temp file: {temp_file_path}")
                os.remove(temp_file_path)
                
            # Don't remove the markdown file immediately as it might still be used by the index
            # We'll rely on the OS to clean up the temp directory periodically
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

def ask_question(question):
    """Process a user question and get response from the indexed documents."""
    try:
        # Update processing status
        st.session_state.processing_status = "Processing your question..."
        
        # Search for relevant documents
        documents = search_documents(question)
        
        if not documents:
            return "I couldn't find any relevant information in the indexed documents."
        
        # Prepare context from the top documents
        context = "\n\n".join([doc["content"] for doc in documents])
        
        # Prepare the prompt
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=f"""You are an AI assistant that answers questions based on the provided document context.
            Use ONLY the provided context to answer questions. If the answer is not in the context, say "I don't have enough information to answer that question."
            Be concise and to the point. Try to extract specific facts from the context rather than giving general information.

            Context:
            {context}"""),
            HumanMessage(content=question)
        ]
        
        # Get the response from the chat model
        try:
            response = chat_model.invoke(messages)
            answer_content = response.content
        except AttributeError:
            # Fallback in case the model returns a string directly
            answer_content = str(response)
        
        # Update session state
        st.session_state.conversation_history.append({
            "question": question,
            "answer": answer_content,
            "documents": documents
        })
        st.session_state.processing_status = ""
        
        return answer_content
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        st.error(f"An error occurred while processing your question: {e}")
        return "I'm sorry, but an error occurred while processing your question."

def display_document_info(docs):
    """Display information about the documents used for the response."""
    if docs:
        st.write("**Sources:**")
        for i, doc in enumerate(docs[:3]):  # Show top 3 sources
            with st.expander(f"Document {i+1}: {doc.get('title', 'Untitled')}"):
                st.write(doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"])

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Document Q&A", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("Talk To Your Data App")
        st.write("Upload and process documents for Q&A")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "md", "txt", "csv", "pptx"])
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    process_file(uploaded_file)
        
        st.divider()
        
        # Show indexed documents
        st.subheader("Indexed Documents")
        if st.session_state.indexed_documents:
            for doc in st.session_state.indexed_documents:
                st.write(f"📄 {doc['name']} ({doc['type']}) - {doc['time']}")
        else:
            st.write("No documents indexed yet.")
    
    # Main area
    st.title("Talk To Your Data RCG Demo App")
    st.write("Upload documents and ask questions about their content.")
    
    # Display processing status
    if st.session_state.processing_status:
        st.info(st.session_state.processing_status)
    
    # Question input
    if st.session_state.documents_indexed:
        st.subheader("Ask a question")
        question = st.text_input("Enter your question")
        
        if question:
            with st.spinner("Thinking..."):
                answer = ask_question(question)
                
                # Display the answer
                st.subheader("Answer")
                st.write(answer)
                
                # Display document info if available
                if st.session_state.conversation_history:
                    last_conversation = st.session_state.conversation_history[-1]
                    display_document_info(last_conversation["documents"])
    else:
        st.warning("Please upload and process documents before asking questions.")
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Previous Questions")
        for i, conv in enumerate(reversed(st.session_state.conversation_history)):
            with st.expander(f"Q: {conv['question']}"):
                st.write("**Answer:**")
                st.write(conv["answer"])


    # Display footer
    st.markdown("---")
    st.caption("RCG Demo Talk To Your Data App | Built on Azure: AI Search, Document Intelligence, AOAI")

if __name__ == "__main__":
    ensure_index_exists()  # Make sure the index exists
    main()
