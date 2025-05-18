import os
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

def main():
    st.title("PDF Upload, Vectorization, and Q&A App")

    # Initialize FAISS index in session state if not already present
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None

    # File uploader for PDF documents
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if st.button("Upload and Vectorize"):
        if not uploaded_file:
            st.error("Please upload a PDF document.")
        else:
            try:
                # Save uploaded file locally
                local_file_path = f"temp_{uploaded_file.name}"
                with open(local_file_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Extract text from the PDF
                with open(local_file_path, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    pdf_text = ""
                    for page in reader.pages:
                        pdf_text += page.extract_text()
                
                # Step 2: Wrap in a Document object
                document = Document(page_content=pdf_text)

                if not document.strip():
                    st.error("The PDF appears to be empty or non-readable.")
                    return

                # Step 3: Split the document into chunks
                chunks = text_splitter.split_documents([document])

                st.success(f"Document split into {len(chunks)} chunks.")
                st.write("Here are the chunks:")
               
                for i, chunk in enumerate(chunks):
                    st.write(f"Chunk {i+1}:")
                    st.write(chunk.page_content)
                    st.write("-" * 40)

                # Step 4: Create or update the FAISS index
                if st.session_state.faiss_index is None:
                    st.write("Creating a new FAISS index...")
                    st.session_state.faiss_index = FAISS.from_documents(chunks, embeddings)
                    st.success("FAISS index initialized successfully!")
                else:
                    st.info("Adding document to existing FAISS index...")
                    st.session_state.faiss_index.add_texts(
                        chunks,
                        metadatas=[{"source": uploaded_file.name}]
                    )
                    st.success("Document added to FAISS index successfully!")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Question-answering section
    st.header("Ask a Question")
    user_question = st.text_input("Enter your question about the uploaded documents:")

    if st.button("Get Answer"):
        if "faiss_index" not in st.session_state or st.session_state.faiss_index is None:
            st.error("No documents have been indexed yet. Please upload and vectorize a document first.")
        elif not user_question.strip():
            st.error("Please enter a valid question.")
        else:
            try:
                # Step 5: Perform similarity search
                st.info("Searching for relevant documents...")
                search_results = st.session_state.faiss_index.similarity_search(user_question, k=1)

                if search_results:
                    st.success("✅ Top matching chunk(s):")
                    for i, res in enumerate(search_results):
                         st.markdown(f"**Chunk {i+1}:**")
                         st.write(res.page_content)

                else:
                    st.warning("No relevant answer found in the indexed documents.")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {e}")


if __name__ == "__main__":
    main()
