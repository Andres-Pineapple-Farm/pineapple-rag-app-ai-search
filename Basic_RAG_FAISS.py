import os
import mimetypes
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdftomarkdown import analyze_documents_output_in_markdown
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import logging

# Configure logging
logging.basicConfig(filename="app.log", 
                    level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

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

# Correct the initialization of AzureChatOpenAI to use the GPT-4 model
chat_model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Ensure this matches the GPT-4 deployment
    model="gpt-4o",  # Correct model name for GPT-4
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# Adjust chunking logic to reduce fragmentation
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increased chunk size to 1000
    chunk_overlap=500  # Increased overlap to 200
)

# Initialize Markdown header text splitter
# This is used to split the document based on headers
md_text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
)

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

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
        document = process_image_based_pdf(file_path)
        document.metadata["is_image_based"] = True
        return document
    else:
        st.info("The uploaded PDF is a native PDF. Extracting text directly.")
        document = process_native_pdf(file_path)
        document.metadata["is_image_based"] = False
        return document

def process_image_based_pdf(file_path: str) -> Document:
    """Process an image-based PDF and return a Document object."""
    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")
    connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

    if not all([endpoint, key, connection_string, container_name]):
        raise ValueError("Missing required environment variables for Document Intelligence.")

    md_file = analyze_documents_output_in_markdown(endpoint, key, file_path, connection_string, container_name)
    
    # Set the source metadata to the Markdown file name
    document = Document(page_content=md_file)
    document.metadata = {"source": os.path.basename(md_file)}
    return document

def process_native_pdf(file_path: str) -> Document:
    """Process a native PDF and return a Document object."""
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = "".join(page.extract_text() for page in reader.pages)
    return Document(page_content=pdf_text)

# Log indexed chunks for debugging
def split_and_index_document(document: Document, file_name: str):
    """Split a document into chunks and update the FAISS index."""

    # Ensure the document is wrapped in a list
    if not isinstance(document, list):
        document = [document]

    # Check if the document content contains Markdown-specific syntax
    if any(header in document[0].page_content for header in ["#", "##", "###"]):
        chunks = md_text_splitter.split_text(document[0].page_content)  # Pass the page_content for markdown
        chunks = [Document(page_content=str(chunk)) for chunk in chunks]  # Ensure each chunk is a string
    else:
        chunks = text_splitter.split_documents(document)

    # Add or preserve source metadata for each chunk
    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata = {"source": file_name}

    # Log chunks for debugging
    for i, chunk in enumerate(chunks):
        logging.debug(f"Chunk {i + 1}: {chunk.page_content[:100]}... (Source: {chunk.metadata['source']})")

    # Initialize or update the FAISS index
    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = FAISS.from_documents(chunks, embeddings)
    else:
        st.session_state.faiss_index.add_texts(
            [chunk.page_content for chunk in chunks], 
            metadatas=[chunk.metadata for chunk in chunks]
        )

# Increase the number of retrieved chunks in similarity search
def handle_question_with_gpt(user_question: str):
    """Handle the question-answering logic using GPT-4."""
    search_results = st.session_state.faiss_index.similarity_search(user_question, k=3)  # Adjust k as needed
    if search_results:
        # Combine the chunks into a single context
        context = "\n".join([res.page_content for res in search_results])

        # Log metadata to inspect unique identifiers
        logging.debug("Metadata for search results:")
        for i, res in enumerate(search_results):
            logging.debug(f"Chunk {i + 1} Metadata: {res.metadata}")

        # Update references to include unique identifiers if available
        references = [
            {
                "source": res.metadata.get("source", "Unknown source"),
                "chunk_number": i + 1,
                "chunk_id": res.metadata.get("id", f"chunk_{i + 1}"),  # Use 'id' if available, fallback to a generated ID
                "chunk_content": res.page_content
            }
            for i, res in enumerate(search_results)
        ]

        try:
            # Ensure the GPT-4 model is explicitly used
            response = chat_model.invoke([{"role": "system", "content": "You are a helpful assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say you don't know."},
                               {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_question}"}])
            # Extract the answer content
            answer = response.content  # Correctly access the content of the message
        except Exception as e:
            logging.error(f"Error during GPT-4 processing: {e}")
            st.error(f"An error occurred while processing your question with GPT-4.{e}")

            return

        # Update conversation history
        st.session_state.conversation_history.append({
            "question": user_question,
            "answer": answer,
            "references": references
        })

        # Display the answer and references
        st.markdown(f"**Answer:** {answer}")
        st.markdown("**References:**")
        for ref in st.session_state.conversation_history[-1]["references"]:
            # Display only the source, chunk_number, and metadata
            st.write(f"- Source: {ref['source']}, Chunk Number: {ref['chunk_number']}, Metadata: {ref.get('metadata', 'No metadata')}")
    else:
        st.warning("No relevant answer found in the indexed documents.")

def main():
    # Initialize session state for FAISS index
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None

    # Sidebar for navigation
    st.sidebar.title("Talk To Your Data App Using FAISS, AOAI [GPT-4o, Ada-Embedding], LangChain, Azure Doc Intel, Streamlit")
    tab = st.sidebar.radio("Go to", ["Upload & Vectorize", "Conversation"])

    if tab == "Upload & Vectorize":
        st.title("PDF Upload and Vectorization")
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

    elif tab == "Conversation":
        st.title("Conversation")

        # Container for question input
        with st.container():
            st.header("Ask a Question")
            user_question = st.text_input("Enter your question about the uploaded documents:")
            if st.button("Get Answer"):
                if not st.session_state.faiss_index:
                    st.error("No documents have been indexed yet.")
                elif not user_question.strip():
                    st.error("Please enter a valid question.")
                else:
                    try:
                        handle_question_with_gpt(user_question)
                    except Exception as e:
                        st.error(f"An error occurred while processing your question: {e}")

        # Container for conversation history
        with st.container():
            st.header("Conversation History")
            for i, entry in enumerate(st.session_state.conversation_history):
                st.markdown(f"**Q{i+1}:** {entry['question']}")
                st.markdown(f"**A{i+1}:** {entry['answer']}")
                st.markdown("**References:**")
                for ref in entry["references"]:
                    st.write(f"- Source: {ref['source']}, Chunk Number: {ref['chunk_number']}, Metadata: {ref.get('metadata', 'No metadata')}")

if __name__ == "__main__":
    main()
