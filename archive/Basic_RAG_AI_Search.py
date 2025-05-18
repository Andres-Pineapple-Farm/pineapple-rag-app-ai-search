import os
import mimetypes
import streamlit as st

from azure.core.credentials import AzureKeyCredential

# Helps us convert image PDFs to Markdown
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult

# Helps us create the AI Search Index
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

# Helps us extract text from PDFs
import PyPDF2
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdftomarkdown import analyze_documents_output_in_markdown

# Helps us chunk the text and store it in a vector store
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
#from langchain_community.vectorstores import FAISS

from config import get_logger

from dotenv import load_dotenv

# initialize logging object
logger = get_logger(__name__)

from azure.search.documents.indexes.models import (SemanticSearch,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    SearchIndex,
    VectorizedQuery
)

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

# Azure Cognitive Search configuration
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = "document-index"

# Initialize SearchIndexClient and SearchClient
index_client = SearchIndexClient(endpoint=search_service_endpoint, 
                                 credential=AzureKeyCredential(search_api_key))

search_client = SearchClient(endpoint=search_service_endpoint, 
                             index_name=index_name, 
                             credential=AzureKeyCredential(search_api_key))

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
    """Split a document into chunks and update the index."""
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
        logger.debug(f"Chunk {i + 1}: {chunk.page_content[:100]}... (Source: {chunk.metadata['source']})")

    # Index documents in Azure Cognitive Search
    index_documents(chunks)

    # Update session state to indicate documents have been indexed
    st.session_state.index = True

# Increase the number of retrieved chunks in similarity search
def handle_question_with_gpt(user_question: str):
    references = search_documents_with_vector(user_question)
    if references:
        context = "\n".join([ref["chunk_content"] for ref in references])

        try:
            response = chat_model.invoke([
                {"role": "system", "content": "You are a helpful assistant. Use ONLY the following context to answer the question. If the answer is not in the context, say you don't know."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_question}"}
            ])
            answer = response.content
        except Exception as e:
            logger.error(f"Error during GPT-4 processing: {e}")
            st.error(f"An error occurred while processing your question with GPT-4.{e}")
            return

        st.session_state.conversation_history.append({
            "question": user_question,
            "answer": answer,
            "references": references
        })

        st.markdown(f"**Answer:** {answer}")
        st.markdown("**References:**")
        for ref in references:
            st.write(f"- Source: {ref['source']}, Chunk Number: {ref['chunk_number']}, Metadata: {ref.get('metadata', 'No metadata')}")
    else:
        st.warning("No relevant answer found in the indexed documents.")

def search_documents_with_vector(user_question: str):
    # search the index for products matching the search query
    query_vector = VectorizedQuery(vector=user_question, k_nearest_neighbors=5, fields="contentVector")
    
    results = search_client.search(
        search_text=user_question, vector_queries=[query_vector], select=["id", "content", "filepath", "title", "url"]
    )
    
    documents = [
        {
            "id": result["id"],
            "content": result["content"],
            "filepath": result["filepath"],
            "title": result["title"],
            "url": result["url"],
        }
        for result in results
    ]
    return documents

## Define the index
def create_index_definition(index_name: str, model: str) -> SearchIndex:
      
    dimensions = 1536  # text-embedding-ada-002
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="filepath", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="url", type=SearchFieldDataType.String),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            # Size of the vector created by the text-embedding-ada-002 model.
            vector_search_dimensions=dimensions,
            vector_search_profile_name="myHnswProfile",
        ),
    ]

    # The "content" field should be prioritized for semantic ranking.
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            keywords_fields=[],
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=1000,
                    ef_search=1000,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
            ExhaustiveKnnAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            ),
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm_configuration_name="myExhaustiveKnn",
            ),
        ],
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

     # Create the search index definition
    return SearchIndex(
        name=index_name,
        fields=fields,
        semantic_search=semantic_search,
        vector_search=vector_search,
    )

def index_documents(chunks):
    actions = [
        {
            "id": str(i),
            "content": chunk.page_content,
            "source": chunk.metadata.get("source", "Unknown source"),
            "metadata": chunk.metadata.get("metadata", "No metadata"),
            "vector": embeddings.embed_query(chunk.page_content)  # Generate vector embeddings
        }
        for i, chunk in enumerate(chunks)
    ]

    try:
        result = search_client.upload_documents(documents=actions)
        logger.info(f"Uploaded {len(actions)} documents to '{index_name}' index")
        print(f"Documents indexed successfully: {result}")
    except Exception as e:
        print(f"Error indexing documents: {e}")
        logger.error(f"Error indexing documents: {e}")

    # try:
    #     index_client.delete_index(index_name)
    #     print(f"Index '{index_name}' deleted successfully.")
    # except Exception as e:
    #     print(f"Error deleting index: {e}")

    # try:
    #     index_client.create_index(index_name)
    #     print(f"Index '{index_name}' created successfully.")
    # except Exception as e:
    #     print(f"Error creating index: {e}")

def recreate_index():
    try:
        index_definition = index_client.get_index(index_name)
        index_client.delete_index(index_name)
        logger.info(f"🗑️  Found existing index named '{index_name}', and deleted it")
    except Exception:
        logger.info(f"🗑️  No existing index named '{index_name}' found, so no need to delete it")
     # create an empty search index
    index_definition = create_index_definition(index_name, model=os.environ["EMBEDDINGS_MODEL"])
    index_client.create_index(index_definition)


# Streamlit app that allows users to upload a PDF document, chunks & vectorize it, and ask questions
# using the indexed content and Azure OpenAI gpt-4o model.
def main():
    # Create the index if it doesn't exist
    try:
        create_index_definition(index_name, model="text-embedding-ada-002")
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        st.error(f"Error creating index: {e}")

    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar for navigation
    st.sidebar.title("Talk To Your Data App")
    st.sidebar.title("Using Azure AI Search, AOAI [GPT-4o, Ada-Embedding], LangChain, Azure Doc Intel, Streamlit")
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
                if not st.session_state.index:
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
