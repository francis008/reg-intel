#!/usr/bin/env python3
"""
Step-by-Step PDF Text Extraction Debug
=====================================

Simple script to understand how unstructured extracts text from a single page
of a Malaysian legal PDF, showing the exact flow and accuracy.
"""

import sys
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from unstructured.partition.pdf import partition_pdf
    print("✅ Unstructured library loaded successfully")
except ImportError:
    print("❌ ERROR: unstructured library not found")
    print("📦 Install with: pip install unstructured[pdf]")
    exit(1)

def debug_single_page():
    """Debug text extraction from a single page of one PDF"""
    
    # Choose one specific PDF
    pdf_path = Path("malaysian_acts/EN/Act_514_EN_Act 514 - Final (1.6.2024).pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("📁 Available PDFs:")
        for pdf in Path("malaysian_acts/EN").glob("*.pdf"):
            print(f"   {pdf.name}")
        return
    
    print(f"🔍 Analyzing: {pdf_path.name}")
    print("=" * 60)
    
    # Extract from just the first page to keep it manageable
    print("\n📄 STEP 1: Extracting elements from page 1...")
    
    try:
        # Use basic extraction settings first
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",  # High resolution for accuracy
            infer_table_structure=True,
            extract_images_in_pdf=False,
        )
        
        print(f"✅ Successfully extracted {len(elements)} elements from entire PDF")
        
        # Filter to only page 1
        page1_elements = []
        for element in elements:
            page_num = 1  # default
            if hasattr(element, 'metadata') and element.metadata:
                if hasattr(element.metadata, 'page_number'):
                    page_num = element.metadata.page_number
            
            if page_num == 1:
                page1_elements.append(element)
        
        print(f"📋 Filtered to page 1: {len(page1_elements)} elements")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return
    
    # Convert to JSON-serializable format
    print("\n📋 STEP 2: Converting to JSON format...")
    
    json_data = {
        "pdf_name": pdf_path.name,
        "page_analyzed": 1,
        "total_elements": len(page1_elements),
        "elements": []
    }
    
    # Process each element
    for i, element in enumerate(page1_elements):
        # Get element details
        element_type = str(type(element).__name__)
        text = str(element).strip()
        
        # Get page number (should be 1)
        page_num = 1
        if hasattr(element, 'metadata') and element.metadata:
            if hasattr(element.metadata, 'page_number'):
                page_num = element.metadata.page_number
        
        # Create JSON element
        json_element = {
            "index": i,
            "page_number": page_num,
            "element_type": element_type,
            "text": text,
            "character_count": len(text),
            "metadata": {}
        }
        
        # Add metadata if available
        if hasattr(element, 'metadata') and element.metadata:
            if hasattr(element.metadata, 'coordinates'):
                json_element["metadata"]["coordinates"] = str(element.metadata.coordinates)
            if hasattr(element.metadata, 'filename'):
                json_element["metadata"]["filename"] = element.metadata.filename
        
        json_data["elements"].append(json_element)
    
    # Save to JSON file
    json_output_path = Path("page1_extraction_debug.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON output saved to: {json_output_path}")
    
    # Show detailed breakdown in console
    print("\n📄 STEP 3: Detailed element breakdown...")
    print(f"PAGE 1 - {len(page1_elements)} elements:")
    print("-" * 60)
    
    for i, element in enumerate(page1_elements):
        element_type = str(type(element).__name__)
        text = str(element).strip()
        
        print(f"\n{i+1:2d}. {element_type} ({len(text)} chars)")
        print(f"    Text: {text}")
        
        # Show coordinates if available
        if hasattr(element, 'metadata') and element.metadata:
            if hasattr(element.metadata, 'coordinates'):
                print(f"    Coordinates: {element.metadata.coordinates}")
    
    print(f"\n📊 STEP 4: Summary statistics...")
    total_chars = sum(len(str(elem)) for elem in page1_elements)
    print(f"Total characters extracted: {total_chars:,}")
    
    # Show element type distribution
    type_counts = {}
    for element in page1_elements:
        element_type = str(type(element).__name__)
        type_counts[element_type] = type_counts.get(element_type, 0) + 1
    
    print(f"Element types found:")
    for elem_type, count in sorted(type_counts.items()):
        print(f"  {elem_type}: {count}")
    
    print(f"\n✅ Analysis complete! Check '{json_output_path}' for full details.")

if __name__ == "__main__":
    debug_single_page()
