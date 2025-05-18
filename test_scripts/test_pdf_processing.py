#!/usr/bin/env python
"""
Test script to demonstrate the PDF processing capabilities of create_search_index.py.

This script:
1. Tests a native PDF with extractable text
2. Tests an image-based PDF that requires OCR
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_processing():
    """
    Test the PDF processing functionality with both types of PDFs.
    """
    # Get the current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Define the test PDFs
    # A native PDF with extractable text (main document)
    native_pdf = current_dir / "Decatur TX Lease.pdf"
    
    # Image-based PDFs that need OCR - we look for both options
    image_pdf_options = [
        current_dir / "temp_Decatur TX Lease.pdf",  # Assuming this is a scanned version
        current_dir / "temp_conda-cheatsheet.pdf"   # Another potential scanned PDF
    ]
    
    # Find the first available image PDF
    image_pdf = None
    for pdf_path in image_pdf_options:
        if pdf_path.exists():
            image_pdf = pdf_path
            break
    
    # Verify we have our PDFs
    if not native_pdf.exists():
        logger.error(f"Native PDF not found: {native_pdf}")
        return
    
    if not image_pdf:
        logger.warning("No image-based PDF found. Will only test native PDF processing.")
        logger.warning("Looked for: " + ", ".join(str(p) for p in image_pdf_options))
    else:
        logger.info(f"Found image-based PDF to test: {image_pdf}")
    # Test 1: Process native PDF
    logger.info("\n" + "=" * 80)
    logger.info("TESTING NATIVE PDF PROCESSING")
    logger.info("=" * 80)
    
    cmd = f"python create_search_index.py --index-name pdf-native-test --file {native_pdf} --file-type pdf"
    logger.info(f"Running command: {cmd}")
    
    os.system(cmd)
      # Test 2: Process image PDF if it exists
    if image_pdf and all(os.getenv(env) for env in ["DOCUMENTINTELLIGENCE_ENDPOINT", "DOCUMENTINTELLIGENCE_API_KEY"]):
        logger.info("\n" + "=" * 80)
        logger.info("TESTING IMAGE-BASED PDF PROCESSING")
        logger.info("=" * 80)
        
        cmd = f"python create_search_index.py --index-name pdf-image-test --file {image_pdf} --file-type pdf"
        logger.info(f"Running command: {cmd}")
        
        os.system(cmd)
    elif image_pdf:
        logger.warning("\n" + "=" * 80)
        logger.warning("SKIPPING IMAGE-BASED PDF TEST - MISSING DOCUMENT INTELLIGENCE CREDENTIALS")
        logger.warning("=" * 80)
        logger.warning("To test image PDFs, set DOCUMENTINTELLIGENCE_ENDPOINT and DOCUMENTINTELLIGENCE_API_KEY")
    
    logger.info("\nTest completed. Check the logs for any errors.")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Native PDF test: {'Completed' if native_pdf.exists() else 'Skipped - PDF not found'}")
    
    if image_pdf:
        has_di_creds = all(os.getenv(env) for env in ["DOCUMENTINTELLIGENCE_ENDPOINT", "DOCUMENTINTELLIGENCE_API_KEY"])
        logger.info(f"Image PDF test: {'Completed' if has_di_creds else 'Skipped - Missing Document Intelligence credentials'}")
    else:
        logger.info("Image PDF test: Skipped - No image PDF found")

if __name__ == "__main__":
    # Verify that all required environment variables are set
    required_envs = [
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_API_BASE",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY"
    ]
    
    # For image-based PDFs, also need Document Intelligence credentials
    image_pdf_envs = [
        "DOCUMENTINTELLIGENCE_ENDPOINT",
        "DOCUMENTINTELLIGENCE_API_KEY"
    ]
    
    missing_envs = [env for env in required_envs if not os.getenv(env)]
    if missing_envs:
        logger.error(f"Missing required environment variables: {', '.join(missing_envs)}")
        logger.error("Please set these environment variables and try again.")
        sys.exit(1)
    
    missing_image_envs = [env for env in image_pdf_envs if not os.getenv(env)]
    if missing_image_envs:
        logger.warning(f"Missing environment variables for image PDF processing: {', '.join(missing_image_envs)}")
        logger.warning("If you want to test image-based PDFs, please set these variables.")
    
    # Run the test
    test_pdf_processing()
