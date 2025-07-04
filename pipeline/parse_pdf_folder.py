#!/usr/bin/env python3
"""
Malaysian Legal PDF Parser
==========================

Parses Malaysian Principal Act PDFs from EN and BM folders using unstructured.
Extracts text while maintaining logical sections, headings, and legal structure.

Features:
- Robust PDF parsing with fallback mechanisms
- Malaysian legal document structure recognition
- Metadata extraction (Act numbers, titles, sections)
- Language detection and handling
- Section/heading preservation for legal citations
- Error handling and recovery
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import hashlib

# PDF processing libraries
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("⚠️ unstructured not available, using fallback PDF parser")

try:
    from pdfminer.high_level import extract_text
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from io import StringIO
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LegalSection:
    """Represents a section of a Malaysian legal document"""
    section_id: str
    heading: str
    content: str
    section_number: Optional[str] = None
    subsection: Optional[str] = None
    page_number: Optional[int] = None
    element_type: str = "text"  # text, table, list_item, title

@dataclass
class MalaysianActDocument:
    """Represents a parsed Malaysian Principal Act"""
    act_number: str
    act_title: str
    language: str  # EN or BM
    filename: str
    file_path: str
    total_pages: int
    sections: List[LegalSection]
    metadata: Dict[str, Any]
    parsing_method: str
    document_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'sections': [asdict(section) for section in self.sections]
        }

class MalaysianLegalPDFParser:
    """
    Advanced PDF parser for Malaysian legal documents
    
    Handles:
    - Malaysian Principal Acts structure
    - Bilingual content (EN/BM)
    - Legal section numbering
    - Headings and subheadings
    - Tables and lists
    - Error recovery and fallbacks
    """
    
    def __init__(self, source_dir: str = "./malaysian_acts", output_dir: str = "./parsed"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create language-specific output directories
        (self.output_dir / "EN").mkdir(exist_ok=True)
        (self.output_dir / "BM").mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'unstructured_used': 0,
            'fallback_used': 0,
            'total_sections': 0
        }
        
        logger.info(f"Initialized Malaysian Legal PDF Parser")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
    
    def extract_act_info_from_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Extract Act number, title, and language from filename
        
        Expected patterns:
        - Act_123_Some_Title_EN.pdf
        - Act_456_BM.pdf
        - 123_Act_Title_English.pdf
        """
        base_name = Path(filename).stem
        
        # Default values
        act_number = "Unknown"
        act_title = base_name
        language = "EN"  # Default to English
        
        # Detect language from filename
        if any(lang_indicator in base_name.upper() for lang_indicator in ['_BM', '_BAHASA', '_MALAY']):
            language = "BM"
        elif any(lang_indicator in base_name.upper() for lang_indicator in ['_EN', '_ENGLISH']):
            language = "EN"
        
        # Extract Act number - look for patterns like "Act_123" or "123"
        act_patterns = [
            r'[Aa]ct[_\s]*(\d+)',  # Act_123, Act 123
            r'^(\d+)[_\s]',        # 123_Title
            r'(\d{1,4})[_\s]'      # Any 1-4 digit number
        ]
        
        for pattern in act_patterns:
            match = re.search(pattern, base_name)
            if match:
                act_number = match.group(1)
                break
        
        # Clean up title - remove act number and language indicators
        title_clean = base_name
        for remove_pattern in [f'Act_{act_number}', f'{act_number}_', '_EN', '_BM', '_English', '_Bahasa']:
            title_clean = title_clean.replace(remove_pattern, '')
        
        # Clean up underscores and extra spaces
        title_clean = re.sub(r'[_]+', ' ', title_clean).strip()
        if title_clean:
            act_title = title_clean
        
        return act_number, act_title, language
    
    def detect_content_language(self, text: str) -> str:
        """Detect language from content using langdetect"""
        if not LANGDETECT_AVAILABLE or not text.strip():
            return "EN"  # Default fallback
        
        try:
            # Sample first 1000 characters for detection
            sample_text = text[:1000].strip()
            if not sample_text:
                return "EN"
            
            detected = detect(sample_text)
            
            # Map detected languages to EN/BM
            if detected in ['ms', 'id']:  # Malay or Indonesian (similar)
                return "BM"
            else:
                return "EN"
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "EN"
    
    def parse_with_unstructured(self, pdf_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Parse PDF using unstructured library (preferred method)"""
        if not UNSTRUCTURED_AVAILABLE:
            return None
        
        try:
            logger.debug(f"Parsing with unstructured: {pdf_path.name}")
            
            # Use unstructured to partition the PDF
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",  # High resolution for better text extraction
                infer_table_structure=True,  # Detect tables
                chunking_strategy="by_title",  # Group by headings
                max_characters=4000,  # Reasonable chunk size
                new_after_n_chars=3000,  # Split long sections
                combine_text_under_n_chars=200,  # Combine short fragments
            )
            
            parsed_elements = []
            for i, element in enumerate(elements):
                element_data = {
                    'element_id': i,
                    'element_type': str(type(element).__name__),
                    'text': str(element),
                    'metadata': getattr(element, 'metadata', {})
                }
                
                # Try to extract page number from metadata
                if hasattr(element, 'metadata') and element.metadata:
                    element_data['page_number'] = element.metadata.get('page_number')
                
                parsed_elements.append(element_data)
            
            self.stats['unstructured_used'] += 1
            return parsed_elements
            
        except Exception as e:
            logger.error(f"Unstructured parsing failed for {pdf_path.name}: {e}")
            return None
    
    def parse_with_pdfminer(self, pdf_path: Path) -> Optional[str]:
        """Fallback PDF parsing using pdfminer"""
        if not PDFMINER_AVAILABLE:
            return None
        
        try:
            logger.debug(f"Parsing with pdfminer: {pdf_path.name}")
            
            text = extract_text(str(pdf_path))
            self.stats['fallback_used'] += 1
            return text
            
        except Exception as e:
            logger.error(f"PDFMiner parsing failed for {pdf_path.name}: {e}")
            return None
    
    def structure_legal_content(self, raw_elements: List[Dict[str, Any]]) -> List[LegalSection]:
        """
        Convert parsed elements into structured legal sections
        Recognizes Malaysian legal document patterns
        """
        sections = []
        
        for element in raw_elements:
            text = element.get('text', '').strip()
            if not text:
                continue
            
            element_type = element.get('element_type', 'text')
            page_number = element.get('page_number')
            
            # Detect section headings and numbers
            section_number = None
            heading = ""
            
            # Malaysian legal section patterns
            section_patterns = [
                r'^(\d+)\.\s*(.+)',  # "1. Section title"
                r'^Section\s+(\d+)[.\s]*(.+)',  # "Section 1. Title"
                r'^(\d+[A-Z]?)\.\s*(.+)',  # "1A. Subsection"
                r'^Part\s+([IVX\d]+)[.\s]*(.+)',  # "Part I. Title"
                r'^Chapter\s+(\d+)[.\s]*(.+)',  # "Chapter 1. Title"
            ]
            
            is_heading = False
            for pattern in section_patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    section_number = match.group(1)
                    heading = match.group(2).strip()
                    is_heading = True
                    break
            
            # If it's a title/heading element from unstructured
            if 'Title' in element_type and not is_heading:
                heading = text
                is_heading = True
            
            # Create section
            section_id = f"section_{len(sections) + 1}"
            if section_number:
                section_id = f"section_{section_number}"
            
            section = LegalSection(
                section_id=section_id,
                heading=heading if is_heading else "",
                content=text,
                section_number=section_number,
                page_number=page_number,
                element_type=element_type.lower()
            )
            
            sections.append(section)
        
        return sections
    
    def structure_plain_text(self, text: str) -> List[LegalSection]:
        """Structure plain text into legal sections (fallback method)"""
        sections = []
        
        # Split by common legal section patterns
        # This is a simplified approach for fallback parsing
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Try to detect if it's a heading
            is_heading = (
                len(paragraph) < 200 and  # Short text
                (paragraph.isupper() or  # ALL CAPS
                 re.match(r'^\d+\.', paragraph) or  # Starts with number
                 re.match(r'^(PART|CHAPTER|SECTION)', paragraph, re.IGNORECASE))
            )
            
            section = LegalSection(
                section_id=f"para_{i + 1}",
                heading=paragraph if is_heading else "",
                content=paragraph,
                element_type="heading" if is_heading else "text"
            )
            
            sections.append(section)
        
        return sections
    
    def parse_single_pdf(self, pdf_path: Path, language_dir: str) -> Optional[MalaysianActDocument]:
        """Parse a single PDF and return structured document"""
        logger.info(f"📄 Parsing: {pdf_path.name}")
        
        try:
            # Extract metadata from filename
            act_number, act_title, detected_language = self.extract_act_info_from_filename(pdf_path.name)
            
            # Override language if specified by directory
            if language_dir in ['EN', 'BM']:
                detected_language = language_dir
            
            # Create document hash for tracking
            with open(pdf_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:12]
            
            # Try parsing with unstructured first
            parsed_elements = self.parse_with_unstructured(pdf_path)
            parsing_method = "unstructured"
            
            if parsed_elements:
                sections = self.structure_legal_content(parsed_elements)
            else:
                # Fallback to pdfminer
                raw_text = self.parse_with_pdfminer(pdf_path)
                parsing_method = "pdfminer"
                
                if not raw_text:
                    logger.error(f"❌ Failed to parse {pdf_path.name} with any method")
                    return None
                
                # Detect language from content if needed
                if detected_language == "EN":  # Only override if default
                    content_language = self.detect_content_language(raw_text)
                    if content_language != detected_language:
                        detected_language = content_language
                
                sections = self.structure_plain_text(raw_text)
            
            # Count pages (rough estimate)
            total_pages = max([s.page_number for s in sections if s.page_number], default=1)
            
            # Create document object
            document = MalaysianActDocument(
                act_number=act_number,
                act_title=act_title,
                language=detected_language,
                filename=pdf_path.name,
                file_path=str(pdf_path),
                total_pages=total_pages,
                sections=sections,
                metadata={
                    'parsing_timestamp': str(Path(pdf_path).stat().st_mtime),
                    'file_size_bytes': pdf_path.stat().st_size,
                    'original_language_dir': language_dir
                },
                parsing_method=parsing_method,
                document_hash=file_hash
            )
            
            self.stats['total_sections'] += len(sections)
            self.stats['successful_parses'] += 1
            
            logger.info(f"   ✅ Parsed: {len(sections)} sections, Language: {detected_language}")
            return document
            
        except Exception as e:
            logger.error(f"❌ Error parsing {pdf_path.name}: {e}")
            self.stats['failed_parses'] += 1
            return None
    
    def save_parsed_document(self, document: MalaysianActDocument) -> bool:
        """Save parsed document as JSON"""
        try:
            # Determine output file
            output_subdir = self.output_dir / document.language
            output_file = output_subdir / f"{document.act_number}_{document.language}.json"
            
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save {document.filename}: {e}")
            return False
    
    def process_folder(self, folder_path: Path) -> List[MalaysianActDocument]:
        """Process all PDFs in a folder"""
        if not folder_path.exists():
            logger.warning(f"⚠️ Folder not found: {folder_path}")
            return []
        
        language_dir = folder_path.name
        pdf_files = list(folder_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"⚠️ No PDF files found in {folder_path}")
            return []
        
        logger.info(f"📁 Processing {len(pdf_files)} PDFs from {language_dir} folder")
        
        documents = []
        for pdf_file in tqdm(pdf_files, desc=f"Processing {language_dir} PDFs"):
            self.stats['total_processed'] += 1
            
            document = self.parse_single_pdf(pdf_file, language_dir)
            if document:
                if self.save_parsed_document(document):
                    documents.append(document)
        
        return documents
    
    def process_all_acts(self) -> Dict[str, List[MalaysianActDocument]]:
        """Process all Malaysian Principal Acts from EN and BM folders"""
        logger.info("🇲🇾 Starting Malaysian Legal PDF Processing Pipeline")
        logger.info("=" * 60)
        
        all_documents = {'EN': [], 'BM': []}
        
        # Process English acts
        en_folder = self.source_dir / "EN"
        if en_folder.exists():
            all_documents['EN'] = self.process_folder(en_folder)
        else:
            logger.warning("⚠️ English folder not found: malaysian_acts/EN")
        
        # Process Bahasa Malaysia acts
        bm_folder = self.source_dir / "BM"
        if bm_folder.exists():
            all_documents['BM'] = self.process_folder(bm_folder)
        else:
            logger.warning("⚠️ Bahasa Malaysia folder not found: malaysian_acts/BM")
        
        # Generate summary
        self.print_processing_summary()
        
        return all_documents
    
    def print_processing_summary(self):
        """Print processing statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MALAYSIAN LEGAL PDF PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📄 Total PDFs processed: {self.stats['total_processed']}")
        logger.info(f"✅ Successfully parsed: {self.stats['successful_parses']}")
        logger.info(f"❌ Failed to parse: {self.stats['failed_parses']}")
        logger.info(f"📚 Total sections extracted: {self.stats['total_sections']}")
        logger.info(f"🔧 Unstructured used: {self.stats['unstructured_used']}")
        logger.info(f"🔄 Fallback used: {self.stats['fallback_used']}")
        
        success_rate = (self.stats['successful_parses'] / max(self.stats['total_processed'], 1)) * 100
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        # Check output
        en_files = len(list((self.output_dir / "EN").glob("*.json")))
        bm_files = len(list((self.output_dir / "BM").glob("*.json")))
        logger.info(f"💾 Output files: {en_files} EN, {bm_files} BM")


def main():
    """Main function to run the PDF parsing pipeline"""
    print("🇲🇾 MALAYSIAN LEGAL PDF PARSER")
    print("=" * 50)
    print("Parsing Malaysian Principal Acts from PDF to structured JSON")
    print()
    
    # Check if source folders exist
    source_dir = Path("./malaysian_acts")
    if not source_dir.exists():
        print("❌ Source directory 'malaysian_acts' not found")
        print("📋 Expected structure:")
        print("   malaysian_acts/")
        print("   ├── EN/     (English PDFs)")
        print("   └── BM/     (Bahasa Malaysia PDFs)")
        return
    
    # Initialize parser
    parser = MalaysianLegalPDFParser()
    
    # Process all documents
    results = parser.process_all_acts()
    
    # Final summary
    total_docs = len(results['EN']) + len(results['BM'])
    print(f"\n🎉 Processing complete! {total_docs} documents parsed")
    print(f"📁 Parsed documents saved to: ./parsed/")
    print(f"📋 Next step: Run chunk_and_embed.py to prepare for vector database")

if __name__ == "__main__":
    main()
