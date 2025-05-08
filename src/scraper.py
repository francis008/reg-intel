# src/scraper.py

import os
import requests
import pathlib

# Use absolute path based on script location
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
DOCS_DIR = os.path.join(SCRIPT_DIR, "..", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

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

if __name__ == "__main__":
    # Replace with your actual PDF URL
    url = "https://www.ssm.com.my/acts/a125pdf.pdf"
    download_pdf_from_url(url)
    
    # Also print working directory for reference
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Documents directory: {os.path.abspath(DOCS_DIR)}")