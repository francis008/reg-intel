#!/usr/bin/env python3
"""
Download Large Embedding Files
==============================

Downloads pre-computed embeddings from cloud storage or regenerates them locally.
This avoids storing large files in Git while keeping them accessible.
"""

import os
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_embeddings():
    """Download or generate embeddings"""
    embeddings_dir = Path("./embeddings")
    embeddings_dir.mkdir(exist_ok=True)
    
    # Check if embeddings already exist
    required_files = [
        "legal_embeddings_complete.pkl",
        "legal_chunks_metadata.json", 
        "legal_embeddings.npy"
    ]
    
    existing_files = [f for f in required_files if (embeddings_dir / f).exists()]
    
    if len(existing_files) == len(required_files):
        logger.info("✅ All embedding files already exist")
        return True
    
    logger.info("📦 Embedding files not found - options:")
    logger.info("1. Download from cloud storage (if available)")
    logger.info("2. Generate embeddings locally (requires parsed data)")
    
    choice = input("Choose option (1/2): ").strip()
    
    if choice == "1":
        return download_from_cloud()
    elif choice == "2":
        return generate_locally()
    else:
        logger.error("Invalid choice")
        return False

def download_from_cloud():
    """Download from cloud storage (implement your cloud storage URLs)"""
    logger.info("🌥️ Cloud download not implemented yet")
    logger.info("💡 You can:")
    logger.info("   - Upload files to Google Drive/Dropbox")
    logger.info("   - Share download links")
    logger.info("   - Add URLs to this script")
    return False

def generate_locally():
    """Generate embeddings locally"""
    logger.info("🔄 Generating embeddings locally...")
    
    # Check if parsed data exists
    parsed_dir = Path("./parsed")
    if not parsed_dir.exists():
        logger.error("❌ Parsed data not found")
        logger.info("📋 Run: python pipeline/parse_pdf_folder.py")
        return False
    
    # Run chunking and embedding
    try:
        import subprocess
        result = subprocess.run([
            "python", "pipeline/chunk_and_embed.py"
        ], check=True)
        
        logger.info("✅ Embeddings generated successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to generate embeddings: {e}")
        return False
    except ImportError:
        logger.error("❌ Required packages not installed")
        logger.info("📋 Run: pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    print("🇲🇾 MALAYSIAN LEGAL EMBEDDINGS DOWNLOADER")
    print("=" * 50)
    
    if download_embeddings():
        print("🎉 Embeddings ready!")
        print("📋 Next: python pipeline/upload_to_vectordb.py")
    else:
        print("❌ Failed to get embeddings")
