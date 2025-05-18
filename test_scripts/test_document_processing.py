"""
Test script to verify document processing works by writing results to a file
"""
import os
import tempfile
from pathlib import Path

def test_processing():
    """Test document processing and write results to a file"""
    from create_index_from_file import create_docs_from_word, create_docs_from_powerpoint, embeddings
    
    # Paths to test files
    word_file = "sample_docs/Commercial_Office_Lease_Agreement.docx"
    ppt_file = "sample_docs/Industrial_Real_Estate_Trends_Midwest_2020_2025.pptx"
    
    # Output file
    output_file = Path(tempfile.gettempdir()) / "doc_processing_test_results.txt"
    
    with open(output_file, "w") as f:
        f.write("Document Processing Test Results\n")
        f.write("==============================\n\n")
        
        # Test Word processing
        f.write("WORD DOCUMENT PROCESSING\n")
        f.write("------------------------\n")
        
        if not os.path.exists(word_file):
            f.write(f"Error: Word test file not found: {word_file}\n")
        else:
            try:
                # Process Word document
                word_docs = create_docs_from_word(path=word_file, model=embeddings.model)
                f.write(f"Successfully created {len(word_docs)} document chunks from Word\n\n")
                
                # Write details about a few chunks
                for i, doc in enumerate(word_docs[:3]):
                    f.write(f"Chunk {i+1}:\n")
                    f.write(f"  ID: {doc.get('id', 'No ID')}\n")
                    f.write(f"  Title: {doc.get('title', 'No title')}\n")
                    f.write(f"  Content length: {len(doc['content'])} characters\n")
                    f.write(f"  Content preview: {doc['content'][:100]}...\n\n")
                    
                if len(word_docs) > 3:
                    f.write(f"... and {len(word_docs) - 3} more chunks\n")
            except Exception as e:
                f.write(f"Error processing Word document: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
        
        # Add spacing between tests
        f.write("\n\n")
        
        # Test PowerPoint processing
        f.write("POWERPOINT DOCUMENT PROCESSING\n")
        f.write("-----------------------------\n")
        
        if not os.path.exists(ppt_file):
            f.write(f"Error: PowerPoint test file not found: {ppt_file}\n")
        else:
            try:
                # Process PowerPoint document
                ppt_docs = create_docs_from_powerpoint(path=ppt_file, model=embeddings.model)
                f.write(f"Successfully created {len(ppt_docs)} document chunks from PowerPoint\n\n")
                
                # Write details about a few chunks
                for i, doc in enumerate(ppt_docs[:3]):
                    f.write(f"Chunk {i+1}:\n")
                    f.write(f"  ID: {doc.get('id', 'No ID')}\n")
                    f.write(f"  Title: {doc.get('title', 'No title')}\n")
                    f.write(f"  Content length: {len(doc['content'])} characters\n")
                    f.write(f"  Content preview: {doc['content'][:100]}...\n\n")
                    
                if len(ppt_docs) > 3:
                    f.write(f"... and {len(ppt_docs) - 3} more chunks\n")
            except Exception as e:
                f.write(f"Error processing PowerPoint document: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
    
    print(f"Results written to {output_file}")
    return output_file

if __name__ == "__main__":
    output_path = test_processing()
    print(f"Testing complete. Results written to {output_path}")
    print("Opening the file for review...")
    os.startfile(output_path)  # This will open the file in the default text editor
