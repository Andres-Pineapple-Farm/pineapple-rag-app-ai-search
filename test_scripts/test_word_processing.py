"""
Test Word document processing for search index creation.

This script tests the Word document processing functionality
in the create_search_index.py module.

Usage:
    python test_word_processing.py path/to/sample.docx
"""

import os
import sys
import tempfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_test():
    """Check if required dependencies are installed and return status."""
    try:
        import docx
        return True
    except ImportError:
        logger.error("python-docx is not installed. Install with: pip install python-docx")
        return False

def test_docx_to_markdown(docx_path):
    """Test conversion of Word document to markdown."""
    try:
        import docx
        from archive.create_search_index import create_docs_from_word
        from langchain_openai import AzureOpenAIEmbeddings
        
        # Initialize embeddings with minimal functionality for testing
        class DummyEmbeddings:
            def embed_query(self, text):
                # Return a dummy vector of proper length
                return [0.0] * 1536
                
        # Use dummy embeddings to avoid API calls during testing
        embeddings = DummyEmbeddings()
        
        # Test the function
        logger.info(f"Testing Word document processing with file: {docx_path}")
        docs = create_docs_from_word(
            path=docx_path,
            embedding_model=embeddings,
            chunk_size=800,
            chunk_overlap=150
        )
        
        logger.info(f"Successfully processed Word document into {len(docs)} chunks")
        
        # Print sample of the first document
        if docs:
            first_doc = docs[0]
            logger.info("Sample content from first chunk:")
            print(f"Title: {first_doc['title']}")
            print(f"Content preview: {first_doc['content'][:200]}...")
            print(f"Content type: {first_doc.get('content_type', 'not set')}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error testing Word document processing: {e}")
        return False

if __name__ == "__main__":
    if not setup_test():
        sys.exit(1)
        
    if len(sys.argv) < 2:
        print("Usage: python test_word_processing.py path/to/sample.docx")
        sys.exit(1)
        
    docx_path = sys.argv[1]
    if not os.path.exists(docx_path):
        logger.error(f"File not found: {docx_path}")
        sys.exit(1)
        
    if not docx_path.lower().endswith(('.docx', '.doc')):
        logger.error(f"File must be a Word document (.docx or .doc): {docx_path}")
        sys.exit(1)
        
    success = test_docx_to_markdown(docx_path)
    if success:
        logger.info("Word document processing test completed successfully")
    else:
        logger.error("Word document processing test failed")
        sys.exit(1)
