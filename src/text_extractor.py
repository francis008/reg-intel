# src/text_extractor.py - Extract text from legal documents

import os
import PyPDF2
import docx
from pathlib import Path
import re

class LegalTextExtractor:
    """Extract and clean text from various legal document formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_from_pdf(self, filepath):
        """Extract text from PDF files"""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error extracting from PDF {filepath}: {e}")
            return ""
    
    def extract_from_docx(self, filepath):
        """Extract text from Word documents"""
        try:
            doc = docx.Document(filepath)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Error extracting from DOCX {filepath}: {e}")
            return ""
    
    def extract_from_txt(self, filepath):
        """Extract text from plain text files"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error extracting from TXT {filepath}: {e}")
            return ""
    
    def extract_text(self, filepath):
        """Extract text from any supported document format"""
        file_extension = Path(filepath).suffix.lower()
        
        if file_extension == '.pdf':
            return self.extract_from_pdf(filepath)
        elif file_extension == '.docx':
            return self.extract_from_docx(filepath)
        elif file_extension == '.txt':
            return self.extract_from_txt(filepath)
        else:
            print(f"Unsupported file format: {file_extension}")
            return ""
    
    def clean_legal_text(self, text):
        """Clean and preprocess legal text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common pattern: "Page X of Y")
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = re.sub(r'[^\w\s.,;:()"\'-]', '', text)  # Remove weird characters
        
        # Standardize legal citations (basic cleanup)
        # This can be expanded with more sophisticated citation parsing
        text = re.sub(r'\s+v\s+', ' v. ', text)  # Standardize "versus"
        
        return text.strip()
    
    def extract_legal_sections(self, text):
        """Extract common legal document sections"""
        sections = {}
        
        # Common legal section headers
        section_patterns = {
            'definitions': r'(?i)(definitions?|interpretation)',
            'whereas': r'(?i)(whereas|recitals?)',
            'obligations': r'(?i)(obligations?|duties|responsibilities)',
            'remedies': r'(?i)(remedies|penalties|enforcement)',
            'termination': r'(?i)(termination|expiry|dissolution)'
        }
        
        for section_name, pattern in section_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                # Extract some context around the match
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 500)
                context = text[start:end]
                
                if section_name not in sections:
                    sections[section_name] = []
                sections[section_name].append(context)
        
        return sections
    
    def create_training_examples(self, filepath, text):
        """Create training examples from legal text"""
        if not text:
            return []
        
        # Clean the text
        clean_text = self.clean_legal_text(text)
        
        # Extract sections
        sections = self.extract_legal_sections(clean_text)
        
        # Create different types of training examples
        training_examples = []
        
        # 1. Document summarization examples
        if len(clean_text) > 1000:  # Only for substantial documents
            summary_example = {
                'type': 'summarization',
                'input': f"Summarize this legal document:\n\n{clean_text[:2000]}...",
                'output': f"Summary needed for {os.path.basename(filepath)}",
                'metadata': {'source': filepath, 'doc_length': len(clean_text)}
            }
            training_examples.append(summary_example)
        
        # 2. Section extraction examples
        for section_type, section_texts in sections.items():
            for section_text in section_texts:
                section_example = {
                    'type': 'section_extraction',
                    'input': f"Extract the {section_type} section from this text:\n\n{section_text}",
                    'output': section_text,
                    'metadata': {'source': filepath, 'section_type': section_type}
                }
                training_examples.append(section_example)
        
        # 3. Question-answering examples (basic)
        qa_example = {
            'type': 'qa',
            'input': f"Based on this legal document, what are the key provisions?\n\nDocument: {clean_text[:1500]}...",
            'output': "Key provisions analysis needed",
            'metadata': {'source': filepath}
        }
        training_examples.append(qa_example)
        
        return training_examples

if __name__ == "__main__":
    # Test the extractor
    extractor = LegalTextExtractor()
    
    # Test with a sample file (you'll need to have a PDF in your docs folder)
    docs_dir = Path(__file__).parent.parent / "docs"
    
    for file in docs_dir.glob("*.pdf"):
        print(f"Extracting text from: {file.name}")
        text = extractor.extract_text(str(file))
        
        if text:
            print(f"Extracted {len(text)} characters")
            print("First 200 characters:")
            print(text[:200])
            print("\n" + "="*50 + "\n")
            
            # Create training examples
            examples = extractor.create_training_examples(str(file), text)
            print(f"Created {len(examples)} training examples")
        else:
            print("No text extracted")
