"""
PDF Processing Utilities

This module contains functions to process PDFs, including:
1. Detecting if a PDF is image-based or has extractable text
2. Converting PDFs to markdown format
3. Processing with Azure Document Intelligence for image-based PDFs
"""

import os
import tempfile
import logging
from pathlib import Path
import PyPDF2
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult
from azure.storage.blob import BlobServiceClient

from config import get_logger

logger = get_logger(__name__)

def is_image_based_pdf(file_path):
    """Check if a PDF is an image-based PDF that requires OCR processing."""
    try:
        # Attempt to extract text using pdfminer
        text = extract_text(file_path)
        if text.strip():
            return False  # Native PDF with extractable text
        else:
            return True  # Image-based PDF (no extractable text)
    except PDFSyntaxError:
        return True  # Likely an image-based PDF

def process_native_pdf(file_path):
    """Process a native PDF with extractable text and convert to markdown."""
    logger.info(f"Processing native PDF: {file_path}")
    
    try:
        # Extract text from native PDF
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pages = []
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    pages.append(f"## Page {i+1}\n{page_text}")
            
            full_text = "\n\n".join(pages)
        
        # Create a temporary markdown file
        temp_md_path = os.path.join(tempfile.gettempdir(), f"{Path(file_path).stem}.md")
        with open(temp_md_path, "w", encoding="utf-8") as md_file:
            md_file.write(f"# Converted PDF Document\n\n{full_text}")
        
        logger.info(f"Created markdown file: {temp_md_path}")
        return temp_md_path
    
    except Exception as e:
        logger.error(f"Error processing native PDF: {e}")
        raise

def process_image_pdf(file_path, endpoint, key, connection_string=None, container_name=None):
    """Process an image-based PDF using Azure Document Intelligence."""
    logger.info(f"Processing image-based PDF: {file_path}")
    
    try:
        # Open and read the file
        with open(file_path, 'rb') as file:
            content = file.read()

        # Initialize Document Intelligence client
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )
        
        # Process the document
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout",
            AnalyzeDocumentRequest(url_source=None, bytes_source=content),
            output_content_format=DocumentContentFormat.MARKDOWN, 
        )
        result: AnalyzeResult = poller.result()

        # Extract and format the information into markdown
        markdown_content = "# Converted from PDF To Markdown\n\n"

        for page in result.pages:
            markdown_content += f"## Page {page.page_number}\n"
            for line in page.lines:
                markdown_content += f"{line.content}\n\n"

        # Create a temporary markdown file
        temp_md_path = os.path.join(tempfile.gettempdir(), f"{Path(file_path).stem}.md")
        with open(temp_md_path, "w", encoding="utf-8") as file:
            file.write(markdown_content)
        
        # Optionally upload to blob storage if credentials provided
        if connection_string and container_name:
            try:
                # Initialize the Blob Service client
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_name = f"{Path(file_path).stem}.md"
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

                # Upload the markdown file to Blob Storage
                with open(temp_md_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.info(f"Uploaded markdown to blob: {blob_name}")
            except Exception as e:
                logger.warning(f"Failed to upload to blob storage: {e}")
        
        logger.info(f"Created markdown file: {temp_md_path}")
        return temp_md_path
        
    except Exception as e:
        logger.error(f"Error processing image-based PDF: {e}")
        raise

def process_pdf(file_path, endpoint=None, key=None, connection_string=None, container_name=None):
    """Process a PDF document, determining if it needs OCR or not."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    # Check if this is an image-based PDF that needs OCR
    if is_image_based_pdf(file_path):
        logger.info(f"Detected image-based PDF: {file_path}")
        
        if not endpoint or not key:
            raise ValueError("Document Intelligence credentials required for image-based PDFs")
            
        # Process with Document Intelligence for OCR
        return process_image_pdf(
            file_path, endpoint, key, connection_string, container_name
        )
    else:
        logger.info(f"Detected native PDF with extractable text: {file_path}")
        
        # Process native PDF by extracting text
        return process_native_pdf(file_path)
