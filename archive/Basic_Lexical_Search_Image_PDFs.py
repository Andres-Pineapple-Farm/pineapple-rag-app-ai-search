import os
import mimetypes
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdftomarkdown import analyze_documents_output_in_markdown

# Load environment variables from .env file
load_dotenv()

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Split the PDF text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Function to check if a PDF is native or image-based
def is_image_based_pdf(file_path):
    try:
        # Attempt to extract text using pdfminer
        text = extract_text(file_path)
        if text.strip():
            return False  # Native PDF
        else:
            return True  # Image-based PDF
    except PDFSyntaxError:
        return True  # Likely an image-based PDF

def upload_and_save_file(uploaded_file) -> str:
    """Save the uploaded file locally and return the file path."""
    local_file_path = f"temp_{uploaded_file.name}"
    with open(local_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return local_file_path

def process_pdf(file_path: str) -> Document:
    """Process a PDF file and return a Document object."""
    if is_image_based_pdf(file_path):
        st.info("The uploaded PDF is an image-based PDF. " \
        "Processing it using document intelligence to Markdown format.")
        return process_image_based_pdf(file_path)
    else:
        st.info("The uploaded PDF is a native PDF. Extracting text directly.")
        return process_native_pdf(file_path)

def process_image_based_pdf(file_path: str) -> Document:
    """Process an image-based PDF and return a Document object."""
    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")
    connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

    if not all([endpoint, key, connection_string, container_name]):
        raise ValueError("Missing required environment variables for Document Intelligence.")

    md_file = analyze_documents_output_in_markdown(endpoint, key, file_path, connection_string, container_name)
    return Document(page_content=md_file)

def process_native_pdf(file_path: str) -> Document:
    """Process a native PDF and return a Document object."""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = "".join(page.extract_text() for page in reader.pages)
    return Document(page_content=pdf_text)

def split_and_index_document(document: Document, file_name: str):
    """Split a document into chunks and update the FAISS index."""
    # Ensure the document is wrapped in a list
    if not isinstance(document, list):
        document = [document]
    
    # Split the document into chunks
    chunks = text_splitter.split_documents(document)
    
    # Initialize or update the FAISS index
    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = FAISS.from_documents(chunks, embeddings)
    else:
        st.session_state.faiss_index.add_texts(
            [chunk.page_content for chunk in chunks], 
            metadatas=[{"source": file_name}] * len(chunks)
        )

def handle_question(user_question: str):
    """Handle the question-answering logic."""
    search_results = st.session_state.faiss_index.similarity_search(user_question, k=1)
    if search_results:
        for i, res in enumerate(search_results):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(res.page_content)
    else:
        st.warning("No relevant answer found in the indexed documents.")

def main():
    # Initialize session state for FAISS index
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None

    st.title("PDF Upload, Vectorization, and Q&A App")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    if st.button("Upload and Vectorize"):
        if not uploaded_file:
            st.error("Please upload a PDF document.")
        else:
            try:
                file_path = upload_and_save_file(uploaded_file)
                document = process_pdf(file_path)
                split_and_index_document(document, uploaded_file.name)
                st.success("Document processed and indexed successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.header("Ask a Question")
    user_question = st.text_input("Enter your question about the uploaded documents:")
    if st.button("Get Answer"):
        if not st.session_state.faiss_index:
            st.error("No documents have been indexed yet.")
        elif not user_question.strip():
            st.error("Please enter a valid question.")
        else:
            try:
                handle_question(user_question)
            except Exception as e:
                st.error(f"An error occurred while processing your question: {e}")

if __name__ == "__main__":
    main()
