# Document Intelligence and Search Application

This application combines Azure Document Intelligence, Azure AI Search, and Azure OpenAI services to create a powerful document processing and question answering system. Users can upload various document types (PDF, Word, PPT, Markdown, CSV), which are processed, chunked, and indexed in Azure AI Search. The application then allows users to ask questions about the documents and receive AI-powered responses. All file types are converted to markdown and then chunked.

## Features

### Document Processing
- Upload and process multiple file formats (PDF, DOCX, PPTX, MD, TXT, CSV)
- Smart PDF handling that automatically detects image-based PDFs
- OCR processing using Azure Document Intelligence for image-based documents
- Automatic conversion of all documents to searchable text chunks

### Search Capabilities
- **Vector-Based Semantic Search**: Find content based on meaning, not just keywords
- **Per-Document Indexing**: Each document gets its own dedicated search index
- **Multi-Document Queries**: Ask questions across multiple selected documents
- **Document Selection**: Choose which documents to include in each search
- **Consistent Metadata**: Each chunk maintains connection to its source document

### User Experience
- **Conversation History**: View all previous questions and answers
- **Source References**: See exactly which parts of documents were used for answers
- **Document Management**: Add, remove, and select documents through the UI
- **Session Management**: Configure automatic cleanup of search indices
- **Expandable Results**: Drill down into the content chunks used for each answer

## Setup Instructions

### 1. Prerequisites

- Azure subscription with the following services:
  - Azure OpenAI
  - Azure AI Search
  - Azure Document Intelligence
  - Azure Blob Storage (optional, for storing processed files)
- Python 3.8 or higher

### 2. Clone the Repository

```bash
git clone <repository-url>
cd pineapple-rag-app-ai-search
```

### 3. Install Dependencies

Run the setup script to install all required dependencies:

```bash
python setup.py
```

### 4. Configure Environment Variables

Edit the `.env` file with your Azure service credentials:

```
# Azure OpenAI Settings
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_API_BASE=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2023-12-01-preview

# Azure AI Search Settings
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=real-estate-document-index

# Azure Document Intelligence Settings
DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-doc-intelligence.cognitiveservices.azure.com/
DOCUMENT_INTELLIGENCE_API_KEY=your_doc_intelligence_key

# Azure Blob Storage Settings (optional, for storing processed files)
AZURE_BLOB_CONNECTION_STRING=your_blob_connection_string
AZURE_BLOB_CONTAINER_NAME=your_container_name
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501

Note: The main application file is `app.py`. If you're using an older version of the codebase, you might need to run `streamlit run app_fixed.py` instead.

## Usage

### Upload Documents

1. Use the file uploader in the sidebar to select documents from your computer
2. Click "Process Document" to upload, process, and index the document
3. Wait for processing to complete - the document will appear in the "Indexed Documents" list

### Select Documents for Search

1. In the sidebar, use the checkboxes next to each document to select which ones to include in your search
2. You can use "Select All" or "Clear All" buttons to quickly manage selections
3. Only selected documents will be searched when asking questions

### Ask Questions

1. Enter your question in the text input box in the main area
2. Click "Ask" to submit your question
3. View the generated answer and the sources it was derived from
4. Expand the source references to see exactly which document chunks were used

### Review Conversation History

1. Previous questions and answers are stored in the session
2. Scroll down to the "Previous Questions" section
3. Click on any question to expand and view the answer and sources

### Session Management

1. The application automatically manages document indices during your session
2. You can configure automatic cleanup settings in the sidebar
3. Use the "Delete" button next to any document to remove it from the index
4. Use "Delete All" to clear all indexed documents

## Testing

You can test specific components of the application:

- Test PDF processing: `python test_pdf_processor.py --pdf_path sample_docs/sample_image_pdf_lease_agreement.pdf`

## Architecture

The application consists of the following components:

1. **Document Processing Pipeline**:
   - File upload and validation
   - Document text extraction (using Azure Document Intelligence for image-based PDFs)
   - Text chunking and embedding generation

2. **Vector Search Index**:
   - Azure AI Search with vector search capabilities
   - Semantic ranking for improved results

3. **Question Answering System**:
   - Vector search to retrieve relevant document chunks
   - Azure OpenAI to generate answers based on retrieved content

## Search Process in Detail

### Document Indexing and Storage

1. **Per-Document Index Creation**
   - Each uploaded document gets its own unique Azure AI Search index
   - Index names use the pattern `doc-index-{document_id}` where document_id is a UUID
   - This approach prevents cross-contamination between document content

2. **Chunking and Metadata**
   - Documents are split into smaller chunks for more effective retrieval
   - Each chunk maintains the following metadata fields:
     - `id`: Unique identifier for the chunk
     - `content`: The actual text content
     - `filepath`: Original document path
     - `title`: Generated title based on document structure
     - `url`: Reference path to the specific section
     - `doc_id`: Parent document identifier (used for filtering)
     - `contentVector`: Vector embedding for semantic search

3. **Vector Embeddings**
   - Each text chunk is converted to a vector using Azure OpenAI embeddings
   - Default model: text-embedding-ada-002 (1536 dimensions)
   - These vectors enable semantic similarity search

### Search and Retrieval

1. **Document Selection**
   - Users can select specific documents to include in search
   - Selection is managed via checkboxes in the UI
   - Document filtering happens at the index level

2. **Query Processing**
   - User questions are converted to vector embeddings
   - Semantic search is performed across selected document indices
   - The top_k most relevant chunks are retrieved from each selected document

3. **Result Ranking**
   - Results are combined across all selected document indices
   - The most semantically similar chunks across all documents are returned
   - Results maintain their source document metadata for reference

## Advanced Features

### Session Management

The application includes a session management system that keeps track of created indices and resources:

1. **Session Tracking**
   - Each browser session gets a unique session ID
   - All indices created during the session are tracked
   - Session activity is monitored for timeout management

2. **Automatic Cleanup**
   - Configure whether indices should be automatically deleted when:
     - The session ends (browser is closed)
     - The session times out (idle for a specified period)
   - Adjust the timeout period (default: 60 minutes)

3. **Resource Management**
   - The application creates one Azure AI Search index per document
   - This approach improves performance and organization
   - Indices can be manually deleted using the delete buttons in the UI

### Multi-Document Querying

The application allows querying across multiple documents simultaneously:

1. **Document Selection**
   - Select which documents to include in each search
   - Results are aggregated across all selected documents
   - The most relevant chunks across all documents are presented

2. **Result Visualization**
   - See which documents contributed to the answer
   - View the specific text chunks that were used
   - Understand the distribution of information across documents

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**:
   - Check that Azure Document Intelligence credentials are correct
   - Ensure PDF files are not password-protected
   - Try with different PDF types (native text vs. image-based)

2. **Search Index Issues**:
   - Verify Azure AI Search credentials
   - Check that your Azure AI Search service has vector search enabled
   - Look for any quota limitations in your Azure Search service
   - Check logs for specific error messages

3. **Question Answering Issues**:
   - Ensure documents have been successfully indexed
   - Try rephrasing your question
   - Check Azure OpenAI credentials and deployment names
   - Verify that you've selected documents to search against

### Logging

The application logs information to `app.log`. Check this file for detailed error messages and debugging information.

## License

[MIT License](LICENSE)

## Acknowledgements

- Azure Document Intelligence
- Azure AI Search
- Azure OpenAI
- Streamlit
