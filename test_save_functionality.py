#!/usr/bin/env python3
"""
Test Save Functionality
=======================

Simple test to verify that parsing and saving works correctly
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.parse_pdf_folder import MalaysianLegalPDFParser

def test_save_functionality():
    """Test if parsing and saving works for a single document"""
    
    print("🔍 TESTING SAVE FUNCTIONALITY")
    print("=" * 50)
    
    # Initialize parser
    parser = MalaysianLegalPDFParser(
        source_dir="./malaysian_acts",
        output_dir="./parsed_test"
    )
    
    # Find a test PDF
    test_pdf = None
    for pdf_file in Path("./malaysian_acts/EN").glob("*.pdf"):
        test_pdf = pdf_file
        break
    
    if not test_pdf:
        print("❌ No test PDF found")
        return
    
    print(f"🔍 Testing with: {test_pdf.name}")
    print(f"📁 Output directory: {parser.output_dir}")
    
    # Parse the document
    print("🔄 Parsing document...")
    doc = parser.parse_single_pdf(test_pdf, "EN")
    
    if doc:
        print(f"✅ Parsed successfully!")
        print(f"📄 Act: {doc.act_number} - {doc.act_title}")
        print(f"📑 Sections: {len(doc.sections)}")
        print(f"🌐 Language: {doc.language}")
        
        # Save the document
        print("\n💾 Saving document...")
        save_success = parser.save_parsed_document(doc)
        
        if save_success:
            output_file = parser.output_dir / doc.language / f"{doc.act_number}_{doc.language}.json"
            print(f"✅ Save successful!")
            print(f"📁 File path: {output_file}")
            
            # Check if file exists
            if output_file.exists():
                file_size = output_file.stat().st_size
                print(f"✅ File confirmed to exist")
                print(f"📊 File size: {file_size:,} bytes")
                
                # Show first few lines of the file
                print("\n📄 File content preview:")
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    for i, line in enumerate(lines, 1):
                        print(f"   {i:2d}: {line.strip()}")
                    if len(lines) >= 10:
                        print("   ... (truncated)")
                
            else:
                print(f"❌ File not found: {output_file}")
        else:
            print("❌ Failed to save document")
    else:
        print("❌ Failed to parse document")
    
    # List all files in output directory
    print("\n📁 FILES IN OUTPUT DIRECTORY:")
    print("-" * 40)
    
    for subdir in ["EN", "BM"]:
        subdir_path = parser.output_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.json"))
            print(f"📂 {subdir}/ ({len(files)} files):")
            for file in files:
                size = file.stat().st_size
                print(f"   • {file.name} ({size:,} bytes)")
        else:
            print(f"📂 {subdir}/ (directory not found)")

if __name__ == "__main__":
    test_save_functionality()
