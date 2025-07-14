#!/usr/bin/env python3
"""
Simple debug script to see what unstructured actually extracts
"""

from pathlib import Path
from pipeline.parse_pdf_folder import MalaysianLegalPDFParser

# Test with the same PDF that's failing
parser = MalaysianLegalPDFParser()
pdf_path = Path("malaysian_acts/BM/Act_867_BM_Akta 867-Akta Iltizam Kecekapan Perkhidmatan Kerajaan 2025.pdf")

print("🔍 Extracting raw elements...")
raw_elements = parser.parse_with_unstructured(pdf_path)

print(f"📊 Found {len(raw_elements)} raw elements")
print("\n" + "="*60)

for i, element in enumerate(raw_elements):
    print(f"ELEMENT {i+1}:")
    print(f"  Type: {element.get('element_type', 'unknown')}")
    print(f"  Page: {element.get('page_number', '?')}")
    print(f"  Text: {element.get('text', '')[:200]}...")
    print(f"  Is boilerplate: {parser.is_boilerplate_or_noise(element.get('text', ''), element.get('page_number'))}")
    print("-" * 40)

print("\n🔍 Testing section detection...")
for i, element in enumerate(raw_elements):
    text = element.get('text', '')
    section_type, section_number, section_title = parser.detect_section_type(text)
    if section_type != 'text':
        print(f"ELEMENT {i+1} - SECTION DETECTED:")
        print(f"  Type: {section_type}")
        print(f"  Number: {section_number}")
        print(f"  Title: {section_title}")
        print(f"  Text: {text[:100]}...")
        print()
