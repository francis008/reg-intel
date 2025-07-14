#!/usr/bin/env python3
"""
Repository Cleanup Script
Removes temporary files, test files, and generated directories
"""

import os
import shutil
from pathlib import Path

def cleanup_repository():
    """Remove temporary and test files from repository"""
    
    print("🧹 CLEANING UP REPOSITORY")
    print("=" * 50)
    
    # Files to remove (temporary test files we created)
    # KEEP legal_ai_system.py and build_malaysian_legal_base.py - they are ESSENTIAL
    files_to_remove = [
        "./test_rag.py",
        "./quick_rag_test.py", 
        "./minimal_rag_test.py",
        # KEEP: "./legal_ai_system.py",  # ESSENTIAL - Core Malaysian Legal AI
        # KEEP: "./build_malaysian_legal_base.py",  # ESSENTIAL - Knowledge base builder  
        "./temp_test_contract.txt",
        "./tiny_contract.txt",
        ".DS_Store"
    ]
    
    # Directories to remove (generated data and caches)
    dirs_to_remove = [
        "./secure_data",
        "./legal_knowledge_base", 
        "./src/__pycache__",
        "./src/test_scraper.ipynb",  # Jupyter notebook we don't need
        "./temp",
        "./.pytest_cache",
        "./models",
        "./embeddings_cache"
    ]
    
    # Remove files
    removed_files = 0
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✅ Removed file: {file_path}")
                removed_files += 1
            except Exception as e:
                print(f"❌ Could not remove {file_path}: {e}")
    
    # Remove directories
    removed_dirs = 0
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"✅ Removed directory: {dir_path}")
                    removed_dirs += 1
                else:
                    os.remove(dir_path)
                    print(f"✅ Removed file: {dir_path}")
                    removed_files += 1
            except Exception as e:
                print(f"❌ Could not remove {dir_path}: {e}")
    
    # Remove Python cache files everywhere
    cache_files_removed = 0
    for root, dirs, files in os.walk("."):
        # Remove __pycache__ directories
        for dir_name in dirs[:]:  # Use slice to avoid modifying while iterating
            if dir_name == "__pycache__":
                cache_dir = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_dir)
                    print(f"✅ Removed cache: {cache_dir}")
                    dirs.remove(dir_name)  # Don't traverse into removed directory
                    cache_files_removed += 1
                except Exception as e:
                    print(f"❌ Could not remove cache {cache_dir}: {e}")
        
        # Remove .pyc files
        for file_name in files:
            if file_name.endswith(('.pyc', '.pyo')):
                cache_file = os.path.join(root, file_name)
                try:
                    os.remove(cache_file)
                    cache_files_removed += 1
                except Exception as e:
                    print(f"❌ Could not remove cache file {cache_file}: {e}")
    
    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"✅ Files removed: {removed_files}")
    print(f"✅ Directories removed: {removed_dirs}")
    print(f"✅ Cache files removed: {cache_files_removed}")
    
    print(f"\n🎉 Repository cleanup complete!")
    print(f"📁 Your repository is now clean and ready for production")

def show_clean_structure():
    """Show what the clean repository structure should look like"""
    print(f"\n📁 CLEAN REPOSITORY STRUCTURE:")
    print("=" * 50)
    print("""
reg-intel/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── LICENSE               # License file
├── .gitignore            # Git ignore rules
├── setup.sh              # Setup script
├── start-api.sh          # API server starter
├── start-web.sh          # Web app starter  
├── start-both.sh         # Start both services
├── src/                  # Source code
│   ├── api.py            # FastAPI application
│   ├── web_app.py        # Streamlit web interface
│   ├── secure_rag_llamaindex.py  # RAG system
│   ├── text_extractor.py # Document processing
│   ├── extract_text.py   # Text extraction utilities
│   └── scraper.py        # Web scraping utilities
└── docs/                 # Documentation files
    └── a125pdf.pdf       # Sample document

REMOVED (temporary/generated files):
❌ test_rag.py            # Temporary test file
❌ quick_rag_test.py      # Temporary test file  
❌ minimal_rag_test.py    # Temporary test file
❌ legal_ai_system.py     # Temporary implementation
❌ build_legal_knowledge_base.py  # Temporary builder
❌ secure_data/           # Generated RAG data
❌ legal_knowledge_base/  # Generated legal docs
❌ __pycache__/           # Python cache files
❌ .DS_Store              # macOS system file
    """)

if __name__ == "__main__":
    cleanup_repository()
    show_clean_structure()
