import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List
import logging
import mimetypes

router = APIRouter()

DOCUMENTS_DIR = "documents_store"

@router.get("/documents", response_model=List[str])
async def list_documents():
    doc_dir_path = DOCUMENTS_DIR 

    if not os.path.exists(doc_dir_path) or not os.path.isdir(doc_dir_path):
        logging.warning(f"Documents directory '{doc_dir_path}' not found or is not a directory.")
        return []
    try:
        files = [
            f
            for f in os.listdir(doc_dir_path)
            if os.path.isfile(os.path.join(doc_dir_path, f)) and not f.startswith('.')
        ]
        return files
    except Exception as e:
        logging.error(f"Error listing documents in '{doc_dir_path}': {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@router.get("/documents/{filename}")
async def get_document(filename: str):
    """Serves a specific document from the documents_store directory."""
    # Basic security: prevent path traversal
    if ".." in filename or filename.startswith("/") or filename.startswith("\\\\"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = os.path.join(DOCUMENTS_DIR, filename)

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")
    
    # Determine media type and disposition based on file extension
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    # Get proper MIME type
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        mime_type = 'application/octet-stream'
    
    # Create response
    response = FileResponse(
        path=file_path, 
        filename=filename, 
        media_type=mime_type
    )
    
    if file_extension == 'pdf':
        # For PDFs, use inline disposition to view in browser
        response.headers["Content-Disposition"] = f"inline; filename=\"{filename}\""
    else:
        # For other files, use attachment disposition to download
        response.headers["Content-Disposition"] = f"attachment; filename=\"{filename}\""
    
    return response