"""
Debugging script for PowerPoint and Word processing
"""
import os
import logging
from pathlib import Path
import tempfile

# Configure logging to write to file
log_file = Path(tempfile.gettempdir()) / "document_processing_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also output to console
    ]
)
logger = logging.getLogger(__name__)

def test_word_processing():
    """Test Word document processing"""
    from create_index_from_file import create_docs_from_word, embeddings
    
    # Log file path
    logger.info(f"Logging to: {log_file}")
    
    # Path to test Word file
    word_file = "sample_docs/Commercial_Office_Lease_Agreement.docx"
    
    if not os.path.exists(word_file):
        logger.error(f"Word test file not found: {word_file}")
        return
    
    logger.info(f"Testing Word document processing with: {word_file}")
    
    try:
        # Process Word document
        docs = create_docs_from_word(path=word_file, model=embeddings.model)
        logger.info(f"Successfully created {len(docs)} document chunks from Word")
        
        # Add more detailed logging
        if docs:
            logger.info(f"First document chunk ID: {docs[0].get('id', 'None')}")
            logger.info(f"First document chunk title: {docs[0].get('title', 'None')}")
        else:
            logger.warning("No document chunks were created from Word file")
        
        return docs
    except Exception as e:
        logger.exception(f"Error processing Word document: {e}")
        return None

def test_powerpoint_processing():
    """Test PowerPoint document processing"""
    from create_index_from_file import create_docs_from_powerpoint, embeddings
    
    # Path to test PowerPoint file
    ppt_file = "sample_docs/Industrial_Real_Estate_Trends_Midwest_2020_2025.pptx"
    
    if not os.path.exists(ppt_file):
        logger.error(f"PowerPoint test file not found: {ppt_file}")
        return
    
    logger.info(f"Testing PowerPoint document processing with: {ppt_file}")
    
    try:
        # Process PowerPoint document
        docs = create_docs_from_powerpoint(path=ppt_file, model=embeddings.model)
        logger.info(f"Successfully created {len(docs)} document chunks from PowerPoint")
        
        # Add more detailed logging
        if docs:
            logger.info(f"First document chunk ID: {docs[0].get('id', 'None')}")
            logger.info(f"First document chunk title: {docs[0].get('title', 'None')}")
        else:
            logger.warning("No document chunks were created from PowerPoint file")
        
        return docs
    except Exception as e:
        logger.exception(f"Error processing PowerPoint document: {e}")
        return None

def run_tests():
    logger.info("Starting document processing tests")
    
    # Test Word processing
    logger.info("=== WORD DOCUMENT TEST ===")
    word_docs = test_word_processing()
    
    # Add spacing between tests
    logger.info("\n\n")
    
    # Test PowerPoint processing
    logger.info("=== POWERPOINT DOCUMENT TEST ===")
    ppt_docs = test_powerpoint_processing()
    
    # Log final results
    logger.info("=== TEST RESULTS ===")
    logger.info(f"Word document chunks: {len(word_docs) if word_docs else 0}")
    logger.info(f"PowerPoint document chunks: {len(ppt_docs) if ppt_docs else 0}")
    
    # Print the log file path for the user
    print(f"\nTest complete. Full log written to: {log_file}")

if __name__ == "__main__":
    run_tests()
    print(f"Log file path: {log_file}")
