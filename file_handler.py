import os
from pathlib import Path
import logging
from typing import Literal, Optional

# Configure logging
logger = logging.getLogger(__name__)

# File type definitions
FileType = Literal['csv', 'markdown', 'pdf', 'word', 'powerpoint', 'unknown']

# File extension mappings
FILE_EXTENSIONS = {
    '.csv': 'csv',
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.pdf': 'pdf',
    '.docx': 'word',
    '.doc': 'word',
    '.pptx': 'powerpoint',
    '.ppt': 'powerpoint'
}

def detect_file_type(file_path: str) -> FileType:
    """
    Detect the type of file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        A string indicating the file type: 'csv', 'markdown', 'pdf', 'word', 'powerpoint', or 'unknown'
    """
    # Get the file extension
    extension = Path(file_path).suffix.lower()
    
    # Return the mapped file type or 'unknown' if not recognized
    return FILE_EXTENSIONS.get(extension, 'unknown')

def validate_file(file_path: str, expected_type: Optional[str] = None) -> tuple[bool, str, FileType]:
    """
    Validate that a file exists and optionally matches an expected type.
    
    Args:
        file_path: Path to the file to validate
        expected_type: Optional expected file type
        
    Returns:
        A tuple containing:
        - Boolean indicating if validation passed
        - Error message (empty string if validation passed)
        - Detected file type
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        return False, f"File not found: {file_path}", 'unknown'
    
    # Detect file type
    detected_type = detect_file_type(file_path)
    
    # If expected_type is specified, validate against it
    if expected_type and detected_type != expected_type and detected_type != 'unknown':
        return False, (f"Expected file type '{expected_type}' but detected '{detected_type}' "
                      f"for file: {file_path}"), detected_type
    
    # If the type is unknown, warn but don't fail
    if detected_type == 'unknown':
        logger.warning(f"Unknown file type for {file_path}. Processing may fail.")
    
    return True, "", detected_type
