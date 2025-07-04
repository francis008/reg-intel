# src/scraper.py - Legal Document Scraper

import os
import requests
import pathlib
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re

# Use absolute path based on script location
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DOCS_DIR = os.path.join(SCRIPT_DIR, "..", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

# Legal document sources in Malaysia
LEGAL_SOURCES = {
    'judiciary': 'https://www.kehakiman.gov.my',
    'agc': 'https://www.agc.gov.my',  # Attorney General's Chambers
    'parlimen': 'https://www.parlimen.gov.my',  # Parliament
    'ssm': 'https://www.ssm.com.my'  # Companies Commission
}

# Legal document types and their indicators
LEGAL_DOC_TYPES = {
    'contract': ['agreement', 'contract', 'mou', 'memorandum of understanding'],
    'court_filing': ['motion', 'brief', 'pleading', 'petition', 'complaint'],
    'judgment': ['judgment', 'ruling', 'decision', 'order'],
    'legislation': ['act', 'law', 'statute', 'regulation', 'ordinance'],
    'legal_memo': ['memo', 'memorandum', 'opinion', 'advice']
}

def download_pdf_from_url(pdf_url):
    filename = pdf_url.split("/")[-1]
    filepath = os.path.join(DOCS_DIR, filename)
    
    # Print the absolute path where file will be saved
    abs_filepath = os.path.abspath(filepath)
    print(f"Will save file to: {abs_filepath}")

    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            # Verify file exists and print size
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024  # KB
                print(f"✅ Downloaded: {filename} ({file_size:.2f} KB)")
                print(f"File location: {abs_filepath}")
            else:
                print(f"⚠️ Something went wrong - file not found after download")
        else:
            print(f"Failed to download. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")

class LegalDocumentProcessor:
    """Processes and classifies legal documents for LLM training"""
    
    def __init__(self, docs_dir):
        self.docs_dir = docs_dir
        
    def classify_document(self, filename, content_snippet=""):
        """
        Classify document type based on filename and content
        Returns: document type (contract, court_filing, etc.)
        """
        filename_lower = filename.lower()
        content_lower = content_snippet.lower()
        
        for doc_type, keywords in LEGAL_DOC_TYPES.items():
            for keyword in keywords:
                if keyword in filename_lower or keyword in content_lower:
                    return doc_type
        
        return 'unknown'
    
    def extract_metadata(self, filepath):
        """Extract basic metadata from legal document"""
        stat = os.stat(filepath)
        filename = os.path.basename(filepath)
        
        metadata = {
            'filename': filename,
            'filepath': filepath,
            'size_kb': stat.st_size / 1024,
            'created_date': stat.st_ctime,
            'doc_type': self.classify_document(filename),
            'source': 'scraped'
        }
        
        return metadata
    
    def prepare_for_training(self, documents_metadata):
        """Prepare documents for LLM training"""
        training_data = []
        
        for doc in documents_metadata:
            # This will be expanded to extract text and create training examples
            training_item = {
                'document_id': doc['filename'],
                'document_type': doc['doc_type'],
                'content_path': doc['filepath'],
                'metadata': doc
            }
            training_data.append(training_item)
        
        return training_data

def download_legal_documents(source_urls):
    """Download multiple legal documents and classify them"""
    processor = LegalDocumentProcessor(DOCS_DIR)
    downloaded_docs = []
    
    for url in source_urls:
        try:
            filename = url.split("/")[-1]
            filepath = os.path.join(DOCS_DIR, filename)
            
            # Download the document
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                # Process and classify
                metadata = processor.extract_metadata(filepath)
                downloaded_docs.append(metadata)
                
                print(f"✅ Downloaded: {filename}")
                print(f"   Type: {metadata['doc_type']}")
                print(f"   Size: {metadata['size_kb']:.2f} KB")
                print(f"   Location: {filepath}")
                print()
            else:
                print(f"❌ Failed to download {url}: Status {response.status_code}")
        
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
    
    return downloaded_docs

if __name__ == "__main__":
    # Example legal document URLs - you can add more
    legal_urls = [
        "https://www.ssm.com.my/acts/a125pdf.pdf",  # Companies Act
        # Add more URLs as you find them
    ]
    
    print("🏗️ Starting Legal Document Collection...")
    print(f"Documents will be saved to: {os.path.abspath(DOCS_DIR)}")
    print()
    
    # Download and process documents
    documents = download_legal_documents(legal_urls)
    
    # Create processor and prepare training data
    processor = LegalDocumentProcessor(DOCS_DIR)
    training_data = processor.prepare_for_training(documents)
    
    # Summary
    print("📊 Collection Summary:")
    print(f"Total documents downloaded: {len(documents)}")
    
    # Group by document type
    type_counts = {}
    for doc in documents:
        doc_type = doc['doc_type']
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    for doc_type, count in type_counts.items():
        print(f"  {doc_type}: {count} documents")
    
    print(f"\nDocuments directory: {os.path.abspath(DOCS_DIR)}")
    print("Ready for LLM training! 🚀")