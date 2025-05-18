"""
Test script to verify Word document processing
"""
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_word_upload():
    """Test Word document upload to index"""
    from create_index_from_file import create_docs_from_word, embeddings
    
    # Path to test Word file
    docx_file = "sample_docs/Commercial_Office_Lease_Agreement.docx"
    
    if not os.path.exists(docx_file):
        print(f"Error: Test file not found: {docx_file}")
        return
    
    print(f"Testing Word document processing with file: {docx_file}")
    
    try:
        # Process the Word document using our function
        docs = create_docs_from_word(
            path=docx_file,
            model=embeddings.model
        )
        
        print(f"Successfully created {len(docs)} document chunks from Word")
        
        # Log some details about the chunks
        for i, doc in enumerate(docs[:3]):  # Show first 3 chunks
            print(f"Chunk {i+1}:")
            print(f"  ID: {doc.get('id', 'No ID')}")
            print(f"  Title: {doc.get('title', 'No title')}")
            print(f"  Content length: {len(doc['content'])}")
            print(f"  Content preview: {doc['content'][:100]}...")
        
        if len(docs) > 3:
            print(f"... and {len(docs) - 3} more chunks")
            
        return docs
        
    except Exception as e:
        logger.error(f"Error processing Word document: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_word_upload()
