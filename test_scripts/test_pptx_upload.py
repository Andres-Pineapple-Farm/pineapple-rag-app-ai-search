"""
Test script to verify PowerPoint document processing
"""
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ppt_upload():
    """Test PowerPoint document upload to index"""
    from create_index_from_file import create_index_from_file
    
    # Path to test PowerPoint file
    pptx_file = "sample_docs/Industrial_Real_Estate_Trends_Midwest_2020_2025.pptx"
    
    if not os.path.exists(pptx_file):
        logger.error(f"Test file not found: {pptx_file}")
        return
    
    logger.info(f"Testing PowerPoint upload with file: {pptx_file}")
    
    try:
        # Create a test index for PowerPoint
        create_index_from_file(
            index_name="test-powerpoint-index",
            file_path=pptx_file,
            file_type='powerpoint'
        )
        logger.info("Successfully processed and indexed PowerPoint file")
    except Exception as e:
        logger.error(f"Error processing PowerPoint file: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_ppt_upload()
