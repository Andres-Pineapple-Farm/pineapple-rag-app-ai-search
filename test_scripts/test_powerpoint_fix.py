"""
Test script to validate PowerPoint processing
"""
import logging
from create_index_from_file import create_docs_from_powerpoint, embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_powerpoint_processing():
    """Test the PowerPoint processing function"""
    pptx_path = "sample_docs/Industrial_Real_Estate_Trends_Midwest_2020_2025.pptx"
    
    logger.info(f"Testing PowerPoint processing with file: {pptx_path}")
    
    try:
        docs = create_docs_from_powerpoint(
            path=pptx_path,
            model=embeddings.model
        )
        
        logger.info(f"Successfully created {len(docs)} document chunks from PowerPoint")
        
        # Log the first few documents
        for i, doc in enumerate(docs[:3]):
            logger.info(f"Document {i+1}:")
            logger.info(f"  Title: {doc['title']}")
            logger.info(f"  Content length: {len(doc['content'])} chars")
            logger.info(f"  Content preview: {doc['content'][:100]}...")
        
        if not docs:
            logger.warning("No document chunks were created!")
        
    except Exception as e:
        logger.error(f"Error processing PowerPoint file: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_powerpoint_processing()
