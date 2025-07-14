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
from collections import defaultdict
from datetime import datetime

# PDF processing libraries - UNSTRUCTURED ONLY for maximum accuracy
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.chunking.title import chunk_by_title
    UNSTRUCTURED_AVAILABLE = True
    print("✅ Unstructured library loaded - prioritizing accuracy over speed")
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    print("❌ ERROR: unstructured library required for legal document processing")
    print("📦 Install with: pip install unstructured[pdf]")
    exit(1)

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Token counting for chunk size optimization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  tiktoken not available - using word count approximation for chunk sizes")

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
    token_count: Optional[int] = None
    word_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate token and word counts after initialization"""
        if self.content:
            self.word_count = len(self.content.split())
            self.token_count = self.estimate_tokens(self.content)
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text"""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
                return len(encoding.encode(text))
            except:
                pass
        
        # Fallback: rough approximation (1 token ≈ 0.75 words)
        return int(len(text.split()) * 1.33)

@dataclass
class LegalChunk:
    """Represents a final chunk ready for embedding"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    token_count: int
    word_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "word_count": self.word_count
        }

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
    parse_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp after initialization"""
        if not self.parse_timestamp:
            self.parse_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'sections': [asdict(section) for section in self.sections]
        }
    
    def to_chunks(self, max_tokens: int = 2000) -> List[LegalChunk]:
        """Convert sections to embedding-ready chunks with enhanced metadata"""
        chunks = []
        
        for section in self.sections:
            # Extract part information if available
            part_info = None
            if hasattr(section, 'part_info') and section.part_info:
                part_info = f"PART {section.part_info['part_number']}"
            
            # Base metadata for all chunks from this section
            base_metadata = {
                "section_number": section.section_number,
                "section_title": section.heading,
                "content": section.content,
                "part": part_info,
                "act_number": self.act_number,
                "act_title": self.act_title,
                "language": self.language,
                "filename": self.filename,
                "section_type": section.element_type,
                "page_number": section.page_number,
                "total_pages": self.total_pages,
                "parse_timestamp": self.parse_timestamp,
                "document_hash": self.document_hash
            }
            
            # Create full text combining heading and content
            full_text = ""
            if section.heading and section.content:
                # Both heading and content exist
                full_text = f"{section.heading}\n\n{section.content}"
            elif section.heading and not section.content:
                # Only heading exists - skip this section as it's likely incomplete
                continue
            elif section.content and not section.heading:
                # Only content exists - use content as is
                full_text = section.content
            else:
                # Neither heading nor content - skip
                continue
            
            # Ensure we have substantial content
            if len(full_text.strip()) < 100:
                continue
            
            # If section fits in one chunk, use it as is
            if section.token_count and section.token_count <= max_tokens:
                chunk = LegalChunk(
                    chunk_id=f"{self.act_number}_section_{section.section_number}" if section.section_number else f"{self.act_number}_{section.section_id}",
                    text=full_text.strip(),
                    metadata=base_metadata,
                    token_count=section.token_count,
                    word_count=section.word_count
                )
                chunks.append(chunk)
            else:
                # Split large sections into smaller chunks
                split_chunks = self._split_section_into_chunks(section, base_metadata, max_tokens)
                chunks.extend(split_chunks)
        
        return chunks
    
    def _split_section_into_chunks(self, section: LegalSection, base_metadata: Dict[str, Any], max_tokens: int) -> List[LegalChunk]:
        """Split a large section into smaller chunks while preserving context"""
        chunks = []
        content_lines = section.content.split('\n')
        
        current_chunk_lines = []
        current_tokens = 0
        chunk_num = 1
        
        # Always include the section heading
        heading_tokens = LegalSection.estimate_tokens(section.heading)
        
        for line in content_lines:
            line_tokens = LegalSection.estimate_tokens(line)
            
            # Check if adding this line would exceed token limit
            if current_tokens + line_tokens + heading_tokens > max_tokens and current_chunk_lines:
                # Create chunk with current content
                chunk_text = f"{section.heading}\n\n" + '\n'.join(current_chunk_lines)
                
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_part"] = chunk_num
                chunk_metadata["is_partial"] = True
                
                chunk = LegalChunk(
                    chunk_id=f"{base_metadata['act_number']}_{section.section_id}_part_{chunk_num}",
                    text=chunk_text.strip(),
                    metadata=chunk_metadata,
                    token_count=current_tokens + heading_tokens,
                    word_count=len(chunk_text.split())
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_lines = [line]
                current_tokens = line_tokens
                chunk_num += 1
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        # Add remaining content
        if current_chunk_lines:
            chunk_text = f"{section.heading}\n\n" + '\n'.join(current_chunk_lines)
            
            chunk_metadata = base_metadata.copy()
            if chunk_num > 1:
                chunk_metadata["chunk_part"] = chunk_num
                chunk_metadata["is_partial"] = True
            
            chunk = LegalChunk(
                chunk_id=f"{base_metadata['act_number']}_{section.section_id}_part_{chunk_num}" if chunk_num > 1 else f"{base_metadata['act_number']}_{section.section_id}",
                text=chunk_text.strip(),
                metadata=chunk_metadata,
                token_count=current_tokens + heading_tokens,
                word_count=len(chunk_text.split())
            )
            chunks.append(chunk)
        
        return chunks

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
            'total_sections': 0,
            'total_elements_extracted': 0
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
        """Parse PDF using unstructured library with optimized settings for Malaysian legal documents"""
        if not UNSTRUCTURED_AVAILABLE:
            logger.error("Unstructured library not available!")
            return None
        
        try:
            
            # Enhanced settings for Malaysian legal documents
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy="hi_res",  # Maximum accuracy for legal documents
                infer_table_structure=True,  # Detect tables in legal documents
                extract_images_in_pdf=False,  # Skip images to focus on text
                extract_image_block_types=["Image", "Table"],  # Only extract relevant blocks
                chunking_strategy="by_title",  # Group content by legal headings
                max_characters=8000,  # Larger chunks for complete legal sections
                new_after_n_chars=6000,  # More generous section breaks
                combine_text_under_n_chars=100,  # Combine very short fragments
                overlap=50,  # Small overlap to preserve context
            )
            
            parsed_elements = []
            for i, element in enumerate(elements):
                element_data = {
                    'element_id': i,
                    'element_type': str(type(element).__name__),
                    'text': str(element),
                    'metadata': {}
                }
                
                # Extract additional metadata for legal documents
                if hasattr(element, 'metadata') and element.metadata:
                    metadata_dict = {}
                    
                    # Safely extract metadata attributes
                    if hasattr(element.metadata, 'page_number'):
                        metadata_dict['page_number'] = element.metadata.page_number
                        element_data['page_number'] = element.metadata.page_number
                    
                    if hasattr(element.metadata, 'filename'):
                        metadata_dict['filename'] = element.metadata.filename
                        element_data['filename'] = element.metadata.filename
                    
                    if hasattr(element.metadata, 'filetype'):
                        metadata_dict['filetype'] = element.metadata.filetype
                        element_data['filetype'] = element.metadata.filetype
                    
                    if hasattr(element.metadata, 'coordinates'):
                        metadata_dict['coordinates'] = element.metadata.coordinates
                        element_data['coordinates'] = element.metadata.coordinates
                    
                    # Store the metadata dictionary
                    element_data['metadata'] = metadata_dict
                
                parsed_elements.append(element_data)
            
            self.stats['unstructured_used'] += 1
            return parsed_elements
            
        except Exception as e:
            logger.error(f"Unstructured parsing failed for {pdf_path.name}: {e}")
            return None
    
    def is_boilerplate_or_noise(self, text: str, page_number: Optional[int] = None) -> bool:
        """
        SIMPLE APPROACH: Only filter truly empty content
        """
        if not text or len(text.strip()) < 5:
            return True
        
        # Only filter page numbers
        if re.match(r'^\d+\s*$', text.strip()):
            return True
        
        return False

    def detect_section_type(self, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Detect the type of legal section and extract identifiers
        Enhanced to properly detect real section headers vs TOC entries
        
        Returns:
            tuple: (section_type, section_number, section_title)
        """
        text_strip = text.strip()
        
        # Enhanced Malaysian legal section patterns - PRIORITIZE REAL SECTIONS
        section_patterns = [
            # Real section headers (most important - should capture actual legal content)
            (r'^(\d{1,3}[A-Z]?)\.\s+(.+)$', 'section', 1, 2),  # "20. General duties of manufacturers"
            (r'^Section\s+(\d{1,3}[A-Z]?)\.\s+(.+)', 'section', 1, 2),  # "Section 20. Title"
            
            # Subsections within sections (only if they have substantial content)
            (r'^\((\d+[a-z]*)\)\s+(.{20,})', 'subsection', 1, 2),  # "(1) Any person who..." with min 20 chars
            (r'^(\d+[A-Z]*)\s*\((\d+[a-z]*)\)\s*(.{20,})', 'subsection', 1, 3),  # "20(1) Content" with min 20 chars
            
            # Parts and chapters (higher level structure)
            (r'^PART\s+([IVX\d]+)[:\.\s]*(.+)', 'part', 1, 2),  # "PART V: SAFETY"
            (r'^Part\s+([IVX\d]+)[:\.\s]*(.+)', 'part', 1, 2),
            (r'^CHAPTER\s+(\d+)[:\.\s]*(.+)', 'chapter', 1, 2),
            (r'^Chapter\s+(\d+)[:\.\s]*(.+)', 'chapter', 1, 2),
            
            # Schedules
            (r'^(?:THE\s+)?(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)\s+SCHEDULE[:\.\s]*(.+)', 'schedule', 0, 1),
            (r'^SCHEDULE\s+([IVX\d]+)[:\.\s]*(.+)', 'schedule', 1, 2),
            (r'^Schedule\s+([IVX\d]+)[:\.\s]*(.+)', 'schedule', 1, 2),
            (r'^SCHEDULE[:\.\s]*(.+)', 'schedule', 0, 1),
            
            # Regulations and rules
            (r'^REGULATION\s+(\d+[A-Z]*)[:\.\s]*(.+)', 'regulation', 1, 2),
            (r'^Regulation\s+(\d+[A-Z]*)[:\.\s]*(.+)', 'regulation', 1, 2),
            (r'^RULE\s+(\d+[A-Z]*)[:\.\s]*(.+)', 'rule', 1, 2),
            (r'^Rule\s+(\d+[A-Z]*)[:\.\s]*(.+)', 'rule', 1, 2),
            
            # Definitions and interpretations
            (r'^(?:INTERPRETATION|DEFINITIONS?)[:\.\s]*(.+)', 'definition', 0, 1),
            (r'^(?:Interpretation|Definitions?)[:\.\s]*(.+)', 'definition', 0, 1),
            
            # Repealed/deleted sections
            (r'^(\d+[A-Z]*)\.\s*(?:\[REPEALED\]|\[DELETED\]|\[OMITTED\])', 'repealed', 1, 0),
            (r'^(\d+[A-Z]*)\.\s*(?:\[repealed\]|\[deleted\]|\[omitted\])', 'repealed', 1, 0),
        ]
        
        for pattern, section_type, num_group, title_group in section_patterns:
            match = re.match(pattern, text_strip, re.IGNORECASE | re.DOTALL)
            if match:
                section_number = match.group(num_group) if num_group > 0 else None
                section_title = match.group(title_group) if title_group > 0 else None
                
                # Clean up title
                if section_title:
                    section_title = section_title.strip(' .-:')
                    # Remove common suffixes that might be artifacts
                    section_title = re.sub(r'\s+\d+$', '', section_title)  # Remove trailing page numbers
                    if len(section_title) > 200:  # Truncate very long titles
                        section_title = section_title[:200] + "..."
                
                return section_type, section_number, section_title
        
        return 'text', None, None

    def merge_section_content(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge related elements into complete legal sections
        
        ENHANCED: Properly captures full section content, not just TOC entries
        Each section includes ALL content until the next section/part header
        """
        if not elements:
            return []
        
        merged_sections = []
        current_section = None
        current_part = None  # Track current part for metadata
        
        for element in elements:
            text = element.get('text', '').strip()
            if not text or self.is_boilerplate_or_noise(text, element.get('page_number')):
                continue
            
            section_type, section_number, section_title = self.detect_section_type(text)
            
            # Track current part for metadata
            if section_type == 'part':
                current_part = {
                    'part_number': section_number,
                    'part_title': section_title
                }
                # Don't create a section for parts - they're structural metadata
                continue
            
            # Check if this starts a new real section (numbered sections)
            if section_type == 'section' and section_number:
                # Save previous section if exists
                if current_section and current_section['content_parts']:
                    merged_sections.append(current_section)
                
                # Start new section with full content
                current_section = {
                    'section_type': 'section',
                    'section_number': section_number,
                    'section_title': section_title or f"Section {section_number}",
                    'content_parts': [text],  # Include the header line
                    'page_numbers': [element.get('page_number')] if element.get('page_number') else [],
                    'element_types': [element.get('element_type', 'text')],
                    'metadata': element.get('metadata', {}),
                    'is_repealed': False,
                    'part_info': current_part  # Include part metadata
                }
            
            # Handle subsections - they belong to current section
            elif section_type == 'subsection' and current_section:
                current_section['content_parts'].append(text)
                if element.get('page_number'):
                    current_section['page_numbers'].append(element.get('page_number'))
                current_section['element_types'].append(element.get('element_type', 'text'))
            
            # Handle schedules as separate sections
            elif section_type in ['schedule', 'regulation', 'rule'] and section_number:
                # Save previous section if exists
                if current_section and current_section['content_parts']:
                    merged_sections.append(current_section)
                
                # Start new schedule/regulation section
                current_section = {
                    'section_type': section_type,
                    'section_number': section_number,
                    'section_title': section_title or f"{section_type.title()} {section_number}",
                    'content_parts': [text],
                    'page_numbers': [element.get('page_number')] if element.get('page_number') else [],
                    'element_types': [element.get('element_type', 'text')],
                    'metadata': element.get('metadata', {}),
                    'is_repealed': False,
                    'part_info': current_part
                }
            
            # Handle general text content - belongs to current section
            elif section_type == 'text' and current_section:
                # Only add if it's substantial content (not just formatting)
                if len(text.strip()) > 10 and not re.match(r'^[_\-\=\*\+\s]+$', text):
                    # Skip obvious header repetitions
                    if text.strip() != current_section['section_title'].strip():
                        current_section['content_parts'].append(text)
                        if element.get('page_number'):
                            current_section['page_numbers'].append(element.get('page_number'))
                        current_section['element_types'].append(element.get('element_type', 'text'))
            
            # Handle definitions sections
            elif section_type == 'definition':
                # Save previous section if exists
                if current_section and current_section['content_parts']:
                    merged_sections.append(current_section)
                
                # Start new definition section
                current_section = {
                    'section_type': 'definition',
                    'section_number': '0',  # Special number for definitions
                    'section_title': section_title or "Interpretation",
                    'content_parts': [text],
                    'page_numbers': [element.get('page_number')] if element.get('page_number') else [],
                    'element_types': [element.get('element_type', 'text')],
                    'metadata': element.get('metadata', {}),
                    'is_repealed': False,
                    'part_info': current_part
                }
            
            # Handle repealed sections
            elif section_type == 'repealed' and section_number:
                # Save previous section if exists
                if current_section and current_section['content_parts']:
                    merged_sections.append(current_section)
                
                # Create minimal section for repealed content
                current_section = {
                    'section_type': 'repealed',
                    'section_number': section_number,
                    'section_title': f"[REPEALED] Section {section_number}",
                    'content_parts': [text],
                    'page_numbers': [element.get('page_number')] if element.get('page_number') else [],
                    'element_types': [element.get('element_type', 'text')],
                    'metadata': element.get('metadata', {}),
                    'is_repealed': True,
                    'part_info': current_part
                }
        
        # Add final section
        if current_section and current_section['content_parts']:
            merged_sections.append(current_section)
        
        # Post-process to ensure quality
        return self._post_process_sections(merged_sections)
    
    def _post_process_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process sections to ensure quality and completeness - LESS AGGRESSIVE"""
        processed_sections = []
        
        for section in sections:
            # Combine all content parts
            combined_content = "\n\n".join(section['content_parts'])
            
            # Skip sections that are too short or empty - LOWER THRESHOLD
            if len(combined_content.strip()) < 20:  # Reduced from 50
                continue
            
            # Remove duplicate page numbers
            section['page_numbers'] = sorted(list(set(p for p in section['page_numbers'] if p)))
            
            # More lenient content validation
            content_lines = [line.strip() for line in combined_content.split('\n') if line.strip()]
            if len(content_lines) < 1:  # Reduced from 2
                continue
            
            processed_sections.append(section)
        
        logger.info(f"Post-processing: {len(sections)} -> {len(processed_sections)} sections")
        return processed_sections
    
    def _is_placeholder_content(self, content: str) -> bool:
        """Check if content appears to be placeholder or meaningless"""
        content_lower = content.lower().strip()
        
        # Check for placeholder patterns
        placeholder_patterns = [
            r'^[\.\s]*$',  # Only dots and spaces
            r'^[_\-\=\*\+\s]+$',  # Only formatting characters
            r'^(\w+\s*){1,3}$',  # Very short content (1-3 words)
            r'^\d+\s*$',  # Only numbers
            r'^[^\w\s]*$',  # Only non-alphanumeric characters
        ]
        
        for pattern in placeholder_patterns:
            if re.match(pattern, content_lower):
                return True
        
        # Check for repetitive content
        words = content_lower.split()
        if len(words) > 3:
            unique_words = set(words)
            if len(unique_words) < len(words) * 0.3:  # Less than 30% unique words
                return True
        
        # Check for common placeholder phrases
        placeholder_phrases = [
            'to be continued',
            'see section',
            'refer to',
            'as defined',
            'not applicable',
            'reserved',
            'placeholder',
            'tbd',
            'todo',
        ]
        
        for phrase in placeholder_phrases:
            if phrase in content_lower:
                return True
        
        return False

    def _is_toc_content(self, content: str) -> bool:
        """Check if content appears to be from Table of Contents"""
        content_lower = content.lower()
        
        # TOC indicators
        toc_indicators = [
            'arrangement of sections',
            'table of contents',
            'contents',
            'page',
            '...',  # Dots common in TOC
        ]
        
        # Check for TOC patterns
        lines = content.split('\n')
        if len(lines) > 3:
            # If more than half the lines contain numbers at the end (page refs)
            page_ref_lines = sum(1 for line in lines if re.search(r'\s+\d+\s*$', line))
            if page_ref_lines > len(lines) * 0.5:
                return True
        
        # Check for TOC keywords
        for indicator in toc_indicators:
            if indicator in content_lower:
                return True
        
        return False
    
    def _extract_title_from_text(self, text: str) -> str:
        """Extract a reasonable title from text content"""
        # Take first line or first 100 characters as title
        first_line = text.split('\n')[0].strip()
        if len(first_line) > 100:
            return first_line[:100] + "..."
        return first_line

    def generate_embeddings_ready_chunks(self, documents: List[MalaysianActDocument], output_file: str = "legal_chunks.json") -> Dict[str, Any]:
        """
        Generate final embeddings-ready chunks from parsed documents
        
        This is the final step that creates JSON output suitable for vector databases
        """
        all_chunks = []
        chunk_stats = defaultdict(int)
        
        logger.info("🔄 Generating embeddings-ready chunks...")
        
        for doc in tqdm(documents, desc="Processing documents"):
            try:
                # Convert document to chunks
                doc_chunks = doc.to_chunks(max_tokens=2000)
                
                for chunk in doc_chunks:
                    # Add to collection
                    all_chunks.append(chunk.to_dict())
                    
                    # Track statistics
                    chunk_stats['total_chunks'] += 1
                    chunk_stats['total_tokens'] += chunk.token_count
                    chunk_stats['total_words'] += chunk.word_count
                    
                    # Track by language
                    lang = chunk.metadata.get('language', 'unknown')
                    chunk_stats[f'{lang}_chunks'] += 1
                    
                    # Track by section type
                    section_type = chunk.metadata.get('section_type', 'unknown')
                    chunk_stats[f'{section_type}_sections'] += 1
                
            except Exception as e:
                logger.error(f"Error generating chunks for {doc.filename}: {e}")
                continue
        
        # Calculate averages
        if chunk_stats['total_chunks'] > 0:
            chunk_stats['avg_tokens_per_chunk'] = chunk_stats['total_tokens'] / chunk_stats['total_chunks']
            chunk_stats['avg_words_per_chunk'] = chunk_stats['total_words'] / chunk_stats['total_chunks']
        
        # Create output structure
        output_data = {
            "chunks": all_chunks,
            "metadata": {
                "total_documents": len(documents),
                "total_chunks": len(all_chunks),
                "generation_timestamp": datetime.now().isoformat(),
                "chunk_stats": dict(chunk_stats),
                "chunk_schema": {
                    "chunk_id": "Unique identifier for the chunk",
                    "text": "Full text content including section heading",
                    "metadata": {
                        "act_number": "Malaysian Act number (e.g., 'Act_1')",
                        "act_title": "Full title of the Act",
                        "language": "EN or BM",
                        "filename": "Source PDF filename",
                        "section_number": "Legal section number",
                        "section_title": "Section heading/title",
                        "section_type": "Type of section (section, part, chapter, etc.)",
                        "page_number": "Page number in original PDF",
                        "total_pages": "Total pages in document",
                        "parse_timestamp": "When document was parsed",
                        "document_hash": "Hash of source document",
                        "chunk_part": "Part number if section was split",
                        "is_partial": "True if section was split into multiple chunks"
                    },
                    "token_count": "Estimated token count for LLM processing",
                    "word_count": "Word count"
                }
            }
        }
        
        # Save to file
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Generated {len(all_chunks)} chunks from {len(documents)} documents")
        logger.info(f"📊 Average tokens per chunk: {chunk_stats.get('avg_tokens_per_chunk', 0):.1f}")
        logger.info(f"📊 Average words per chunk: {chunk_stats.get('avg_words_per_chunk', 0):.1f}")
        logger.info(f"💾 Saved to: {output_path}")
        
        return output_data
    
    def structure_legal_content(self, raw_elements: List[Dict[str, Any]]) -> List[LegalSection]:
        """
        Convert parsed elements into structured legal sections with robust chunking
        ENHANCED: Creates proper individual section chunks with full content
        """
        if not raw_elements:
            logger.warning("No raw elements to process")
            return []
        
        logger.info(f"Processing {len(raw_elements)} raw elements")
        
        # Filter out boilerplate and merge related content
        merged_sections = self.merge_section_content(raw_elements)
        
        logger.info(f"After merging: {len(merged_sections)} sections")
        
        structured_sections = []
        
        for section_data in merged_sections:
            # Combine all content parts into a single text
            combined_content = "\n\n".join(section_data['content_parts'])
            
            # Skip if content is too short or meaningless
            if len(combined_content.strip()) < 50:
                continue
            
            # Create clean section ID
            section_number = section_data['section_number']
            section_id = f"section_{len(structured_sections) + 1}"
            if section_number:
                section_id = f"section_{section_number}"
            
            # Extract page range
            page_numbers = [p for p in section_data['page_numbers'] if p]
            page_number = min(page_numbers) if page_numbers else None
            
            # Clean the content - separate header from body content
            content_lines = combined_content.split('\n')
            header_line = ""
            body_content = ""
            
            if len(content_lines) > 1:
                # Check if first line is the section header
                first_line = content_lines[0].strip()
                if re.match(r'^\d+[A-Z]?\.\s+', first_line):
                    # Remove header from content, keep separate
                    header_line = first_line
                    body_content = '\n'.join(content_lines[1:]).strip()
                else:
                    # No clear header, use title as header
                    header_line = section_data['section_title']
                    body_content = combined_content.strip()
            else:
                # Single line - might be header only
                header_line = section_data['section_title']
                body_content = combined_content.strip()
            
            # CRITICAL: Ensure we have substantial body content beyond just the header
            if len(body_content.strip()) < 50:  # Reduced from 100 to 50
                continue
            
            # Additional validation: ensure it's not just a header repeated
            if body_content.strip() == header_line.strip():
                continue
            
            # Check for meaningful content (not just whitespace, numbers, or symbols)
            meaningful_chars = len(re.sub(r'[^\w\s]', '', body_content))
            if meaningful_chars < 30:  # Reduced from 50 to 30
                continue
            
            # Create structured section with enhanced metadata
            section = LegalSection(
                section_id=section_id,
                heading=header_line,
                content=body_content,
                section_number=section_number,
                subsection=None,
                page_number=page_number,
                element_type=section_data['section_type']
            )
            
            structured_sections.append(section)
        
        return structured_sections
    
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
            
            # Parse with unstructured (accuracy-focused approach)
            parsed_elements = self.parse_with_unstructured(pdf_path)
            parsing_method = "unstructured"
            
            if not parsed_elements:
                logger.error(f"❌ Failed to parse {pdf_path.name} - unstructured parsing returned no elements")
                return None
            
            # Structure the content into legal sections
            sections = self.structure_legal_content(parsed_elements)
            
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
            self.stats['total_elements_extracted'] += len(parsed_elements)
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
        
        # Generate summary and chunks
        self.print_processing_summary(all_documents)
        
        return all_documents
    
    def print_processing_summary(self, documents: Dict[str, List[MalaysianActDocument]]):
        """Print processing statistics"""
        total_docs = len(documents['EN']) + len(documents['BM'])
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 MALAYSIAN LEGAL PDF PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📄 Total PDFs processed: {self.stats['total_processed']}")
        logger.info(f"✅ Successfully parsed: {self.stats['successful_parses']}")
        logger.info(f"❌ Failed to parse: {self.stats['failed_parses']}")
        logger.info(f"📚 Total sections extracted: {self.stats['total_sections']}")
        logger.info(f"🔧 Total elements processed: {self.stats['total_elements_extracted']}")
        logger.info(f"🎯 Unstructured parsing used: {self.stats['unstructured_used']} times")
        
        success_rate = (self.stats['successful_parses'] / max(self.stats['total_processed'], 1)) * 100
        avg_sections = self.stats['total_sections'] / max(self.stats['successful_parses'], 1)
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        logger.info(f"📊 Average sections per document: {avg_sections:.1f}")
        
        # Check output
        en_files = len(list((self.output_dir / "EN").glob("*.json")))
        bm_files = len(list((self.output_dir / "BM").glob("*.json")))
        logger.info(f"💾 Output files: {en_files} EN, {bm_files} BM")
        logger.info("🎯 Parsing method: Unstructured only (maximum accuracy)")
        
        # Quality recommendations
        if success_rate < 90:
            logger.warning("⚠️  Success rate below 90% - consider reviewing failed documents")
        
        if self.stats['total_sections'] < self.stats['successful_parses'] * 5:
            logger.warning("⚠️  Low section count - may indicate parsing quality issues")
    
    def generate_final_chunks(self, documents: Dict[str, List[MalaysianActDocument]], max_tokens: int = 2000) -> Dict[str, Any]:
        """Generate final embedding-ready chunks from all documents"""
        all_docs = documents['EN'] + documents['BM']
        
        if not all_docs:
            logger.warning("⚠️  No documents available for chunk generation")
            return {}
        
        return self.generate_embeddings_ready_chunks(all_docs, "legal_chunks_complete.json")


def main():
    """Main function to run the PDF parsing pipeline"""
    import argparse
    
    parser_cmd = argparse.ArgumentParser(description="Parse Malaysian Legal PDFs for AI/RAG systems")
    parser_cmd.add_argument("--source", default="./malaysian_acts", help="Source directory containing EN and BM folders")
    parser_cmd.add_argument("--output", default="./parsed", help="Output directory for parsed documents")
    parser_cmd.add_argument("--generate-chunks", action="store_true", help="Generate embedding-ready chunks")
    parser_cmd.add_argument("--max-tokens", type=int, default=2000, help="Maximum tokens per chunk")
    parser_cmd.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser_cmd.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🇲🇾 MALAYSIAN LEGAL PDF PARSER")
    print("=" * 50)
    print("High-accuracy parsing using unstructured library")
    print("Optimized for Malaysian Principal Acts")
    print()
    
    # Check unstructured availability
    if not UNSTRUCTURED_AVAILABLE:
        print("❌ ERROR: unstructured library is required for this parser")
        print("📦 Install with: pip install unstructured[pdf]")
        print("🔧 Additional dependencies may be needed:")
        print("   - poppler-utils (for PDF processing)")
        print("   - tesseract (for OCR capabilities)")
        return
    
    # Check if source folders exist
    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"❌ Source directory '{args.source}' not found")
        print("📋 Expected structure:")
        print("   malaysian_acts/")
        print("   ├── EN/     (English PDFs)")
        print("   └── BM/     (Bahasa Malaysia PDFs)")
        return
    
    print("🎯 Processing mode: Unstructured only (maximum accuracy)")
    print("⏱️  Note: High-accuracy parsing may take longer but provides better results")
    print()
    
    # Initialize parser
    parser = MalaysianLegalPDFParser(source_dir=args.source, output_dir=args.output)
    
    # Process all documents
    results = parser.process_all_acts()
    
    # Generate chunks if requested
    if args.generate_chunks:
        print("\n🔄 Generating embedding-ready chunks...")
        chunk_data = parser.generate_final_chunks(results, max_tokens=args.max_tokens)
        
        if chunk_data:
            print(f"📦 Generated {len(chunk_data['chunks'])} embedding-ready chunks")
            print(f"💾 Chunks saved to: {args.output}/legal_chunks_complete.json")
    
    # Final summary
    total_docs = len(results['EN']) + len(results['BM'])
    print(f"\n🎉 Processing complete! {total_docs} documents parsed with maximum accuracy")
    print(f"📁 Parsed documents saved to: {args.output}/")
    
    if not args.generate_chunks:
        print("� Tip: Use --generate-chunks to create embedding-ready chunks for vector databases")
    
    return results

if __name__ == "__main__":
    main()
