"""
Session Management for Azure AI Search Indices

This module provides functions to manage Azure AI Search indices 
during Streamlit sessions, including tracking, cleanup, and timeout handling.
"""

import streamlit as st
import time
import uuid
from typing import Dict, List, Set, Tuple, Optional
from azure.search.documents.indexes import SearchIndexClient
from config import get_logger

logger = get_logger(__name__)

def update_session_activity():
    """Update the session's last activity timestamp."""
    if "last_activity_time" in st.session_state:
        st.session_state.last_activity_time = time.time()
    
def register_session(index_client: SearchIndexClient):
    """Register a new Streamlit session for resource tracking.
    
    Args:
        index_client: Azure AI Search index client
    """
    # Generate a unique session ID if not already present
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session started with ID: {st.session_state.session_id}")
    
    # Initialize last activity timestamp if not present
    if "last_activity_time" not in st.session_state:
        st.session_state.last_activity_time = time.time()
    
    # Initialize timeout settings (default: 60 minutes)
    if "session_timeout_minutes" not in st.session_state:
        st.session_state.session_timeout_minutes = 60
    
    # Initialize cleanup settings
    if "cleanup_on_exit" not in st.session_state:
        st.session_state.cleanup_on_exit = True
    
    # Store indices created in this session if not already tracking
    if "session_indices" not in st.session_state:
        st.session_state.session_indices = set()
        # Add any existing indices to track
        for doc_id, index_name in st.session_state.get("document_indices", {}).items():
            st.session_state.session_indices.add(index_name)
    
    # Check for timed out session
    check_session_timeout(index_client)
    
    # Log session information
    logger.debug(f"Session {st.session_state.session_id} active, tracking {len(st.session_state.session_indices)} indices")

def check_session_timeout(index_client: SearchIndexClient):
    """Check if the session has timed out and needs cleanup.
    
    Args:
        index_client: Azure AI Search index client
    """
    # Skip if cleanup is disabled
    if not st.session_state.get("cleanup_on_exit", True):
        return
    
    # Get the current time
    current_time = time.time()
    
    # Get the last activity time (default to current time if not set)
    last_activity = st.session_state.get("last_activity_time", current_time)
    
    # Check if session has timed out (default to 1 hour)
    timeout_minutes = st.session_state.get("session_timeout_minutes", 60)
    timeout_seconds = timeout_minutes * 60
    
    if current_time - last_activity > timeout_seconds:
        logger.info(f"Session {st.session_state.get('session_id', 'unknown')} timed out after {timeout_minutes} minutes of inactivity")
        cleanup_session_resources(index_client)
    else:
        # Update the last activity time
        st.session_state.last_activity_time = current_time

def cleanup_session_resources(index_client: SearchIndexClient):
    """Clean up resources associated with the session.
    
    Args:
        index_client: Azure AI Search index client
    """
    if not st.session_state.get("cleanup_on_exit", True):
        logger.info("Automatic cleanup disabled, skipping resource cleanup")
        return
        
    logger.info(f"Cleaning up resources for session {st.session_state.get('session_id', 'unknown')}")
    
    # Get indices to delete
    indices_to_delete = []
    for doc_id, index_name in st.session_state.get("document_indices", {}).items():
        indices_to_delete.append((doc_id, index_name))
    
    # Delete each index
    for doc_id, index_name in indices_to_delete:
        try:
            index_client.delete_index(index_name)
            logger.info(f"Cleaned up index {index_name} for document {doc_id}")
        except Exception as e:
            logger.error(f"Error cleaning up index {index_name}: {e}")
    
    # Clear document-related session state
    if "indexed_documents" in st.session_state:
        st.session_state.indexed_documents = []
    if "document_indices" in st.session_state:
        st.session_state.document_indices = {}
    if "selected_doc_ids" in st.session_state:
        st.session_state.selected_doc_ids = []
    if "session_indices" in st.session_state:
        st.session_state.session_indices = set()
        
    logger.info("Session cleanup completed")

def track_index(index_name: str, doc_id: str):
    """Track a new index in the current session.
    
    Args:
        index_name: Name of the Azure AI Search index
        doc_id: ID of the document associated with the index
    """
    # Add the index to the session indices set
    if "session_indices" not in st.session_state:
        st.session_state.session_indices = set()
    
    st.session_state.session_indices.add(index_name)
    logger.debug(f"Added index {index_name} to session tracking for document {doc_id}")

def untrack_index(index_name: str, doc_id: str):
    """Remove an index from tracking in the current session.
    
    Args:
        index_name: Name of the Azure AI Search index
        doc_id: ID of the document associated with the index
    """
    if "session_indices" in st.session_state and index_name in st.session_state.session_indices:
        st.session_state.session_indices.remove(index_name)
        logger.debug(f"Removed index {index_name} from session tracking for document {doc_id}")

def render_cleanup_settings():
    """Render cleanup settings UI in the Streamlit app."""
    with st.expander("Session & Cleanup Settings"):
        # Automatic cleanup toggle
        cleanup_enabled = st.toggle(
            "Automatically clean up indices when session ends",
            value=st.session_state.get("cleanup_on_exit", True),
            help="When enabled, document indices will be automatically deleted when your session times out"
        )
        if cleanup_enabled != st.session_state.get("cleanup_on_exit", True):
            st.session_state.cleanup_on_exit = cleanup_enabled
            logger.info(f"Updated automatic cleanup setting to: {cleanup_enabled}")
        
        # Timeout setting
        timeout_minutes = st.slider(
            "Session timeout (minutes)",
            min_value=10,
            max_value=240,
            value=st.session_state.get("session_timeout_minutes", 60),
            step=10,
            help="Session will be considered inactive after this period of no activity"
        )
        if timeout_minutes != st.session_state.get("session_timeout_minutes", 60):
            st.session_state.session_timeout_minutes = timeout_minutes
            logger.info(f"Updated session timeout to: {timeout_minutes} minutes")
        
        # Session info
        st.caption(f"Session ID: {st.session_state.get('session_id', 'Not initialized')}")
        st.caption(f"Active indices: {len(st.session_state.get('session_indices', set()))}")
