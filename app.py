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
from session_manager import register_session, update_session_activity, cleanup_session_resources, track_index, untrack_index, render_cleanup_settings
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
DEFAULT_INDEX_PREFIX = "doc-index-"
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".pptx", ".md", ".txt", ".csv"]

# We'll initialize search clients dynamically for each document index

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
def search_documents(question: str, doc_ids: List[str] = None, top_k: int = 5) -> List[dict]:
    """Search for documents related to a question using Azure AI Search.
    
    This function now searches across multiple indices, one per document.
    """
    try:
        # Validate inputs
        if not question or question.strip() == "":
            logger.warning("Empty question provided to search_documents")
            return []
            
        # Make sure doc_ids is properly formatted if provided
        if doc_ids is not None:
            # Handle edge cases
            if not isinstance(doc_ids, list):
                logger.warning(f"doc_ids is not a list: {type(doc_ids)}. Converting to list.")
                try:
                    doc_ids = list(doc_ids)
                except:
                    logger.error(f"Could not convert doc_ids to list: {doc_ids}")
                    doc_ids = []
            
            # Remove any None or empty values
            doc_ids = [doc_id for doc_id in doc_ids if doc_id]
            
            # Check if we have any document IDs to search
            if not doc_ids:
                logger.warning("No valid document IDs provided for search")
                return []
        else:
            logger.warning("No document IDs provided for search")
            return []
        
        # Generate vector embeddings for the query
        query_vector = embeddings.embed_query(question)
        
        # Use Azure AI Search vector search
        from azure.search.documents.models import VectorizedQuery
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=top_k, fields="contentVector")
        
        # Collect results from all selected document indices
        all_documents = []
        for doc_id in doc_ids:
            # Get the index name for this document
            if doc_id not in st.session_state.document_indices:
                logger.warning(f"No index found for document ID {doc_id}")
                continue
                
            index_name = st.session_state.document_indices[doc_id]
            logger.info(f"Searching index {index_name} for document {doc_id}")
            
            # Create a search client for this index
            search_client = SearchClient(
                endpoint=search_service_endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(search_api_key)
            )
            
            try:
                # Perform the search on this index
                search_results = search_client.search(
                    search_text=question,
                    vector_queries=[vector_query],
                    select=["id", "content", "filepath", "title", "url", "doc_id"]
                )
                
                # Format the results for this document
                documents = [
                    {
                        "id": result["id"],
                        "content": result["content"],
                        "filepath": result.get("filepath", "Unknown"),
                        "title": result.get("title", "Untitled"),
                        "url": result.get("url", ""),
                        "doc_id": result.get("doc_id", ""),
                        "index_name": index_name  # Add the index name for reference
                    }
                    for result in search_results
                ]
                
                logger.info(f"Search found {len(documents)} matching chunks in index {index_name}")
                all_documents.extend(documents)
                
            except Exception as e:
                logger.error(f"Error searching index {index_name}: {e}")
                # Continue with other indices even if one fails
        
        # Sort all results by relevance (we could implement a more sophisticated ranking here)
        # For now, we just return all results, limited by top_k
        top_results = all_documents[:top_k] if len(all_documents) > top_k else all_documents
        
        logger.info(f"Total search results across all indices: {len(top_results)}")
        return top_results
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        logger.error(f"Search error details: {traceback.format_exc()}")
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
    if "document_indices" not in st.session_state:
        st.session_state.document_indices = {}  # Maps doc_id to index_name
    if "selected_doc_ids" not in st.session_state:
        st.session_state.selected_doc_ids = []
    if "question" not in st.session_state:
        st.session_state.question = ""
    if "previous_uploaded_filename" not in st.session_state:
        st.session_state.previous_uploaded_filename = None
    if "select_all_state" not in st.session_state:
        st.session_state.select_all_state = True
    
    # Log initialized state
    logger.info(f"Session state initialized with {len(st.session_state.indexed_documents)} documents")
    if st.session_state.indexed_documents:
        logger.info(f"Selected doc IDs: {st.session_state.selected_doc_ids}")
        logger.info(f"Document indices: {st.session_state.document_indices}")

def create_index_for_document(doc_id, file_name):
    """Create a new search index for a document.
    
    Returns the index name.
    """
    # Generate a unique index name based on document ID
    # Ensure it follows Azure naming rules: lowercase letters, numbers, or dashes
    index_name = f"{DEFAULT_INDEX_PREFIX}{doc_id.replace('-', '')}"
    
    try:
        # Check if the index already exists
        try:
            index_client.get_index(index_name)
            logger.info(f"Index '{index_name}' already exists")
        except Exception:
            # Create the index if it doesn't exist
            index_definition = create_index_definition(index_name, model=embeddings.model)
            index_client.create_index(index_definition)
            logger.info(f"Created new index '{index_name}' for file '{file_name}'")
        
        # Store the mapping between doc_id and index_name
        st.session_state.document_indices[doc_id] = index_name
        st.session_state.documents_indexed = True
        
        # Track the index in our session manager
        track_index(index_name, doc_id)
        logger.info(f"Tracking index '{index_name}' in session {st.session_state.get('session_id', 'unknown')}")
        
        return index_name
        
    except Exception as e:
        logger.error(f"Error creating index for document: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_file(file):
    """Process an uploaded file and prepare it for indexing."""
    temp_file_path = None
    markdown_path = None
    
    try:
        # Generate a document ID
        doc_id = str(uuid.uuid4())
        logger.info(f"Processing file {file.name} with doc_id: {doc_id}")
        
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
        
        # Create a search index for this document
        index_name = create_index_for_document(doc_id, file.name)
        
        if index_name:
            # Index the document
            st.info(f"Creating index for {file.name}...")
            try:
                create_index_from_file(
                    index_name=index_name, 
                    file_path=markdown_path,
                    file_type=file_type,
                    doc_id=doc_id
                )
                
                # Update session state with document info
                document_info = {
                    "id": doc_id,
                    "name": file.name,
                    "file_type": file_type,
                    "path": markdown_path,
                    "index_name": index_name
                }
                
                # Only add if not already present
                if doc_id not in [doc["id"] for doc in st.session_state.indexed_documents]:
                    st.session_state.indexed_documents.append(document_info)
                
                # Select the document by default
                st.session_state.selected_doc_ids.append(doc_id)
                st.session_state[f"select_{doc_id}"] = True
                
                st.session_state.processing_status = f"{file.name} indexed successfully!"
                st.success(f"{file.name} indexed successfully!")
                
                # Update the previous filename to prevent re-indexing the same file
                st.session_state.previous_uploaded_filename = file.name
                
                logger.info(f"Document {file.name} indexed successfully with ID {doc_id}")
                return doc_id
                
            except Exception as e:
                st.error(f"Error indexing document: {e}")
                logger.error(f"Error indexing document: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        else:
            st.error("Failed to create search index")
            return None
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        logger.error(f"Error processing file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    finally:
        # Cleanup temp files if needed
        pass

def ask_question(question, selected_doc_ids=None):
    """Process a user question and get response from the indexed documents."""
    # Use the documents from the selected document IDs
    if not question.strip():
        return "Please enter a question."
        
    try:
        # Get relevant document chunks
        docs = search_documents(question, selected_doc_ids)
        
        if not docs:
            return "I couldn't find any relevant information to answer your question. Please try asking a different question or select different documents."
        
        # Count sources per document
        doc_counts = {}
        for doc in docs:
            doc_id = doc.get("doc_id", "unknown")
            if doc_id in doc_counts:
                doc_counts[doc_id] += 1
            else:
                doc_counts[doc_id] = 1
            
        # Build simple context
        context = "\n\n".join([f"Document: {doc.get('filepath', 'Unknown')}\nContent: {doc['content']}" for doc in docs])
        
        # Build the prompt
        if len(docs) == 1:
            # Single document
            prompt = f"""You are an AI assistant helping to answer questions based on the provided document. 
            Answer the following question using only the information from the document. If you don't know, say so.
            
            DOCUMENT CONTENT:
            {context}
            
            QUESTION: {question}
            
            ANSWER:"""
        else:
            # Multiple documents
            prompt = f"""You are an AI assistant helping to answer questions based on the provided documents.
            Answer the following question using only the information from the documents. If you don't know, say so.
            
            DOCUMENT CONTENT:
            {context}
            
            QUESTION: {question}
            
            ANSWER:"""
        
        # Get response from model
        response = chat_model.invoke(prompt)
        
        # Store the conversation
        st.session_state.conversation_history.append({
            "question": question,
            "answer": response.content,
            "documents": docs
        })
        
        logger.info(f"Generated answer from {len(docs)} document chunks across {len(doc_counts)} documents")
        
        # Return the answer
        return response.content
    except Exception as e:
        logger.error(f"Error asking question: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"An error occurred while generating an answer: {str(e)}"

def display_document_info(docs):
    """Display information about documents used to answer a question."""
    if not docs:
        return
    
    # Count chunks per document and organize by document
    doc_chunks = {}
    for doc in docs:
        doc_id = doc.get("doc_id", "unknown")
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = {
                "name": "Unknown",
                "chunks": []
            }
            # Find the document info in indexed_documents
            for indexed_doc in st.session_state.indexed_documents:
                if indexed_doc.get("id") == doc_id:
                    doc_chunks[doc_id]["name"] = indexed_doc.get("name", "Unknown")
                    break
        
        doc_chunks[doc_id]["chunks"].append(doc)
    
    # Display document info
    st.info(f"Answer based on {len(docs)} text chunks from {len(doc_chunks)} documents")
    
    # Create expandable sections for each document
    for doc_id, info in doc_chunks.items():
        with st.expander(f"{info['name']} ({len(info['chunks'])} chunks)"):
            for i, chunk in enumerate(info['chunks']):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk['content'])
                st.divider()

def delete_document_index(doc_id, index_name):
    """Delete a document's index and remove it from the session state."""
    try:
        # Delete the index from Azure AI Search
        try:
            index_client.delete_index(index_name)
            logger.info(f"Successfully deleted index {index_name} for document {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting index {index_name}: {e}")
            st.error(f"Error deleting index: {e}")
            return False
        
        # Remove the document from session state
        st.session_state.indexed_documents = [doc for doc in st.session_state.indexed_documents if doc["id"] != doc_id]
        
        # Remove the document from the selected docs if it was selected
        if doc_id in st.session_state.selected_doc_ids:
            st.session_state.selected_doc_ids.remove(doc_id)
        
        # Remove the mapping from document_indices
        if doc_id in st.session_state.document_indices:
            del st.session_state.document_indices[doc_id]
            
        # Untrack the index in our session manager
        untrack_index(index_name, doc_id)
        logger.info(f"Removed index '{index_name}' from session tracking")
            
        # Remove the checkbox state
        checkbox_key = f"select_{doc_id}"
        if checkbox_key in st.session_state:
            del st.session_state[checkbox_key]
        
        st.success(f"Document deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error removing document from state: {e}")
        st.error(f"Error removing document: {e}")
        return False

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Document Q&A", layout="wide")
    
    # Initialize session state
    init_session_state()
    
    # Register session for tracking and cleanup
    register_session(index_client)
    
    # Update session activity timestamp
    update_session_activity()
      
    # Sidebar
    with st.sidebar:
        st.title("Talk To Your Data App")
        st.write("Upload and process documents for Q&A")
          
        # Define a callback for when a user uploads a file
        def on_file_upload():
            st.session_state.question = ""  # Clear question on file upload
            
        # Use the on_change parameter to detect file changes
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "md", "txt", "csv", "pptx"], 
                                         on_change=on_file_upload)
        
        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    process_file(uploaded_file)
                    st.session_state.question = ""  # Also clear question after processing
          
        st.divider()
        
        # Show indexed documents with selection
        st.subheader("Indexed Documents")
        if st.session_state.indexed_documents:
            # Add Select All / Clear All buttons
            select_col, clear_col, delete_col = st.columns(3)
            
            def select_all_docs():
                # Set all document checkboxes to checked
                for doc in st.session_state.indexed_documents:
                    doc_key = f"select_{doc['id']}"
                    st.session_state[doc_key] = True
                # Update selected_doc_ids with a new list to avoid reference issues
                doc_ids = [doc["id"] for doc in st.session_state.indexed_documents]
                logger.info(f"Select All: Setting {len(doc_ids)} documents as selected")
                st.session_state.selected_doc_ids = doc_ids.copy()
                
            def clear_all_docs():
                # Set all document checkboxes to unchecked
                for doc in st.session_state.indexed_documents:
                    doc_key = f"select_{doc['id']}"
                    st.session_state[doc_key] = False
                # Update selected_doc_ids with a new empty list
                logger.info("Clear All: Clearing all document selections")
                st.session_state.selected_doc_ids = []
                
            def delete_all_docs():
                # Delete all indices and clear the session state
                indices_to_delete = []
                for doc in st.session_state.indexed_documents:
                    indices_to_delete.append((doc['id'], doc['index_name']))
                
                # Delete each index
                for doc_id, index_name in indices_to_delete:
                    delete_document_index(doc_id, index_name)
                
                # Clear all state
                st.session_state.indexed_documents = []
                st.session_state.document_indices = {}
                st.session_state.selected_doc_ids = []
                st.success("All documents deleted")
                
            with select_col:
                if st.button("Select All"):
                    select_all_docs()
            with clear_col:
                if st.button("Clear All"):
                    clear_all_docs()
            with delete_col:
                if st.button("Delete All", type="primary"):
                    if st.session_state.indexed_documents:
                        delete_all_docs()
            
            # List documents with checkboxes
            for doc in st.session_state.indexed_documents:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Create a key for this document's checkbox
                    checkbox_key = f"select_{doc['id']}"
                    
                    # Handle checkbox state
                    if checkbox_key not in st.session_state:
                        # Default to checked
                        st.session_state[checkbox_key] = True
                        # Add to selected doc IDs
                        if doc['id'] not in st.session_state.selected_doc_ids:
                            st.session_state.selected_doc_ids.append(doc['id'])
                    
                    # Display the checkbox
                    is_checked = st.checkbox(doc["name"], key=checkbox_key)
                    
                    # Update selected doc IDs based on checkbox
                    if is_checked:
                        if doc['id'] not in st.session_state.selected_doc_ids:
                            st.session_state.selected_doc_ids.append(doc['id'])
                    else:
                        if doc['id'] in st.session_state.selected_doc_ids:
                            st.session_state.selected_doc_ids.remove(doc['id'])
                
                with col2:
                    # Delete button for each document
                    if st.button("🗑️", key=f"delete_{doc['id']}"):
                        delete_document_index(doc['id'], doc['index_name'])
        else:
            st.write("No documents indexed yet. Upload a document to begin.")
            
        # Show session and cleanup settings
        st.divider()
        render_cleanup_settings()
    
    # Main area
    st.title("Document Q&A")
    
    if st.session_state.documents_indexed:
        # Show number of selected documents
        selected_count = len(st.session_state.selected_doc_ids)
        total_count = len(st.session_state.indexed_documents)
        st.write(f"Selected {selected_count} out of {total_count} documents for search")
        
        # Question input
        question = st.text_input("Ask a question about your documents:", key="question")
        selected_doc_ids = st.session_state.selected_doc_ids
        
        # Button to ask question
        if st.button("Ask") and question:
            if selected_doc_ids:
                with st.spinner("Thinking..."):
                    # Pass selected document IDs to search function
                    answer = ask_question(question, selected_doc_ids)
                    
                    # Display the answer
                    st.subheader("Answer")
                    st.write(answer)
                    
                    # Display document info if available
                    if st.session_state.conversation_history:
                        last_conversation = st.session_state.conversation_history[-1]
                        display_document_info(last_conversation["documents"])
            else:
                st.warning("Please select at least one document to search.")
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
    main()
