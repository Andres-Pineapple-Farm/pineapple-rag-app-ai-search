"""
Test script for PDF processing with Azure Document Intelligence
This script tests the pdf_processor.py module with a sample PDF document
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import tempfile
import argparse

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom PDF processor
from pdf_processor import process_pdf, is_image_based_pdf

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Process a PDF file and convert to markdown')
    parser.add_argument('--pdf_path', help='Path to the PDF file to process', 
                        default='sample_docs/sample_image_pdf_lease_agreement.pdf')
    args = parser.parse_args()

    # Set the PDF file path
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    # Get Azure Document Intelligence credentials from environment variables
    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY")
    connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")

    if not endpoint or not key:
        print("Error: Azure Document Intelligence credentials not found in environment variables")
        return

    print(f"Processing PDF: {pdf_path}")
    
    # Detect if this is an image-based PDF
    is_image_pdf = is_image_based_pdf(pdf_path)
    print(f"Is image-based PDF: {is_image_pdf}")

    # Process the PDF
    try:
        markdown_path = process_pdf(
            pdf_path, 
            endpoint=endpoint,
            key=key,
            connection_string=connection_string,
            container_name=container_name
        )
        
        print(f"PDF processed successfully. Markdown file created at {markdown_path}")
        
        # Display the first 500 characters of the markdown file
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
            print("\nMarkdown Preview (first 500 characters):")
            print(markdown_content[:500] + "...")
            
        print(f"\nTotal markdown content length: {len(markdown_content)} characters")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
