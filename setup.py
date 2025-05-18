"""
Setup script to install required dependencies for the Document Intelligence and Search App
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    requirements = [
        "streamlit>=1.32.0",
        "azure-ai-documentintelligence>=1.0.0",
        "azure-search-documents>=11.4.0",
        "azure-storage-blob>=12.19.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-core>=0.1.16",
        "python-dotenv>=1.0.0",
        "pypdf2>=3.0.1",
        "pdfminer.six>=20221105",
        "pandas>=2.0.0"
    ]
    
    for requirement in requirements:
        print(f"Installing {requirement}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
    
    print("\nAll dependencies installed successfully.")

def check_env_file():
    """Check if .env file exists, create template if not."""
    if not os.path.exists(".env"):
        print("Creating template .env file...")
        with open(".env", "w") as f:
            f.write("""# Azure OpenAI Settings
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
""")
        print(".env template created. Please edit the file with your Azure service credentials.")
    else:
        print(".env file already exists.")

if __name__ == "__main__":
    install_requirements()
    check_env_file()
    
    print("\nSetup complete!")
    print("To run the application, use: streamlit run app_fixed.py")
    print("\nMake sure you have updated the .env file with your Azure service credentials.")
    print("If you haven't already created the required Azure resources, please refer to the README.md file.")
