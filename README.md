# Document Intelligence and Search Application

This application combines Azure Document Intelligence, Azure AI Search, and Azure OpenAI services to create a powerful document processing and question answering system. Users can upload various document types (PDF, Word, PPT, Markdown, CSV), which are processed, chunked, and indexed in Azure AI Search. The application then allows users to ask questions about the documents and receive AI-powered responses. All file types are converted to markdown and then chunked.

## Features

- **Document Processing**: Upload PDF, DOCX, PPTX, MD, TXT, and CSV files
- **Smart PDF Handling**: Automatically detects image-based PDFs and processes them with Azure Document Intelligence
- **Vector Indexing**: Documents are chunked and indexed with vector embeddings for semantic search
- **Question Answering**: Ask questions about your documents and get AI-powered answers
- **Conversation History**: View your previous questions and answers
- **Source References**: See which parts of your documents were used to generate answers

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

## Usage

### Upload Documents

1. Go to the "Upload Documents" section
2. Click "Browse Files" to select documents from your computer
3. Click "Process Document" to upload and index the documents
4. Wait for processing to complete

### Ask Questions

1. Go to the "Ask Questions" section
2. Type your question in the input box
3. Click "Get Answer"
4. View the answer and the sources it was derived from

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

## Troubleshooting

### Common Issues

1. **PDF Processing Errors**:
   - Check that Azure Document Intelligence credentials are correct
   - Ensure PDF files are not password-protected
   - Try with different PDF types (native text vs. image-based)

2. **Search Index Issues**:
   - Verify Azure AI Search credentials
   - Try recreating the index using the "Reset Search Index" button
   - Check logs for specific error messages

3. **Question Answering Issues**:
   - Ensure documents have been successfully indexed
   - Try rephrasing your question
   - Check Azure OpenAI credentials and deployment names

### Logging

The application logs information to `app.log`. Check this file for detailed error messages and debugging information.

## License

[MIT License](LICENSE)

## Acknowledgements

- Azure Document Intelligence
- Azure AI Search
- Azure OpenAI
- Streamlit
