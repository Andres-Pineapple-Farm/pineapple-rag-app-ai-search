"""
Test PowerPoint document processing for search index creation.

This script tests the PowerPoint document processing functionality
in the create_search_index.py module.

Usage:
    python test_powerpoint_processing.py path/to/sample.pptx
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
        import pptx
        return True
    except ImportError:
        logger.error("python-pptx is not installed. Install with: pip install python-pptx")
        return False

def test_pptx_to_markdown(pptx_path):
    """Test conversion of PowerPoint to markdown."""
    try:
        import pptx
        from archive.create_search_index import create_docs_from_powerpoint
        from langchain_openai import AzureOpenAIEmbeddings
        
        # Initialize embeddings with minimal functionality for testing
        class DummyEmbeddings:
            def embed_query(self, text):
                # Return a dummy vector of proper length
                return [0.0] * 1536
                
        # Use dummy embeddings to avoid API calls during testing
        embeddings = DummyEmbeddings()
        
        # Test the function
        logger.info(f"Testing PowerPoint processing with file: {pptx_path}")
        docs = create_docs_from_powerpoint(
            path=pptx_path,
            embedding_model=embeddings,
            chunk_size=800,
            chunk_overlap=150
        )
        
        logger.info(f"Successfully processed PowerPoint into {len(docs)} chunks")
        
        # Print sample of the first document
        if docs:
            first_doc = docs[0]
            logger.info("Sample content from first chunk:")
            print(f"Title: {first_doc['title']}")
            print(f"Content preview: {first_doc['content'][:200]}...")
            print(f"Content type: {first_doc.get('content_type', 'not set')}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error testing PowerPoint processing: {e}")
        return False

if __name__ == "__main__":
    if not setup_test():
        sys.exit(1)
        
    if len(sys.argv) < 2:
        print("Usage: python test_powerpoint_processing.py path/to/sample.pptx")
        sys.exit(1)
        
    pptx_path = sys.argv[1]
    if not os.path.exists(pptx_path):
        logger.error(f"File not found: {pptx_path}")
        sys.exit(1)
        
    if not pptx_path.lower().endswith(('.pptx', '.ppt')):
        logger.error(f"File must be a PowerPoint document (.pptx or .ppt): {pptx_path}")
        sys.exit(1)
        
    success = test_pptx_to_markdown(pptx_path)
    if success:
        logger.info("PowerPoint processing test completed successfully")
    else:
        logger.error("PowerPoint processing test failed")
        sys.exit(1)
