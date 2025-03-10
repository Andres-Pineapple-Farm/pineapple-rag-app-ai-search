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

def analyze_documents_output_in_markdown():
    # [START analyze_documents_output_in_markdown]
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat, AnalyzeResult
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

    endpoint = ""
    key = ""
    # sample file below
    #url = "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/main/sdk/documentintelligence/azure-ai-documentintelligence/samples/sample_forms/forms/Invoice_1.pdf"

    # Reference a specific file on the desktop
    file_path = ""

    # Open and read the file
    with open(file_path, 'rb') as file:
        content = file.read()
        print(content)

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
    blob_service_client = BlobServiceClient.from_connection_string("")
    container_name = ""
    blob_name = local_file_name

    # Upload the markdown file to Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open(local_file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print("Document Intelligence processed file has been uploaded to Blob Storage successfully.")


    # print(f"completed upload to blob storage")
    # print(result.content)
    # # [END analyze_documents_output_in_markdown]


if __name__ == "__main__":
    from azure.core.exceptions import HttpResponseError
    from dotenv import find_dotenv, load_dotenv

    try:
        load_dotenv(find_dotenv())
        analyze_documents_output_in_markdown()
    except HttpResponseError as error:
        # Examples of how to check an HttpResponseError
        # Check by error code:
        if error.error is not None:
            if error.error.code == "InvalidImage":
                print(f"Received an invalid image error: {error.error}")
            if error.error.code == "InvalidRequest":
                print(f"Received an invalid request error: {error.error}")
            # Raise the error again after printing it
            raise
        # If the inner error is None and then it is possible to check the message to get more information:
        if "Invalid request".casefold() in error.message.casefold():
            print(f"Uh-oh! Seems there was an invalid request: {error}")
        # Raise the error again
        raise