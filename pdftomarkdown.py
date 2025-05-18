# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: pdf-to-markdown.py

DESCRIPTION:
    Demonstrates how to analyze a pdf document of images (ie scanned in pages) into markdown format and 
    place the resulting file into blob storage

USAGE:
    python pdf-to-markdown.py

    Set the environment variables with your own values before running the sample:
    1) DOCUMENTINTELLIGENCE_ENDPOINT - the endpoint to your Document Intelligence resource.
    2) DOCUMENTINTELLIGENCE_API_KEY - your Document Intelligence API key.
"""

import os
import streamlit as st

def main():
    st.title("PDF to Markdown Converter")

    # Input fields for user to provide necessary details
    endpoint = st.text_input("Document Intelligence Endpoint", "")
    key = st.text_input("Document Intelligence API Key", "", type="password")
    file_path = st.text_input("Path to PDF File", "")
    connection_string = st.text_input("Azure Blob Storage Connection String", "", type="password")
    container_name = st.text_input("Blob Storage Container Name", "")

    if st.button("Convert and Upload"):
        if not endpoint or not key or not file_path or not connection_string or not container_name:
            st.error("Please fill in all the fields.")
        else:
            try:
                analyze_documents_output_in_markdown(endpoint, key, file_path, connection_string, container_name)
                st.success("File successfully converted and uploaded to Blob Storage.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Modify the function to accept parameters
def analyze_documents_output_in_markdown(endpoint, key, file_path, connection_string, container_name):
    # [START analyze_documents_output_in_markdown]
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult
    from azure.storage.blob import BlobServiceClient

    # Open and read the file
    with open(file_path, 'rb') as file:
        content = file.read()

    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
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

    # Write the markdown content to a local file
    local_file_name = "decatur_tx_lease.md"
    with open(local_file_name, "w", encoding="utf-8") as file:
        file.write(markdown_content)

    # Initialize the Blob Service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

    # Upload the markdown file to Blob Storage
    with open(local_file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    # Return the markdown content
    return markdown_content
    # [END analyze_documents_output_in_markdown]

if __name__ == "__main__":
    main()