#!/usr/bin/env python3
"""
Malaysian Legal Document Chunker and Embedder
=============================================

Processes parsed Malaysian legal documents to create semantic chunks and embeddings.

Features:
- Intelligent legal document chunking by sections/paragraphs
- Open-source embedding generation (privacy-friendly)
- Malaysian legal metadata preservation
- Bilingual support (EN/BM)
- Optimized chunk sizes for legal search
- Batch processing for efficiency
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import pickle
import numpy as np
from tqdm import tqdm
import hashlib

# Chunking and embedding libraries
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available, using basic chunking")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("❌ sentence-transformers not available - embeddings disabled")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️ tiktoken not available, using character-based chunking")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LegalChunk:
    """Represents a chunk of Malaysian legal text with metadata"""
    chunk_id: str
    act_number: str
    act_title: str
    language: str
    section_id: str
    section_heading: str
    chunk_text: str
    chunk_index: int  # Position within the section
    token_count: int
    page_number: Optional[int] = None
    section_number: Optional[str] = None
    citation: str = ""  # For legal referencing
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Generate citation
        if not self.citation:
            self.citation = self.generate_citation()
    
    def generate_citation(self) -> str:
        """Generate legal citation for this chunk"""
        citation_parts = [f"Act {self.act_number}"]
        
        if self.section_number:
            citation_parts.append(f"Section {self.section_number}")
        elif self.section_heading:
            citation_parts.append(f'"{self.section_heading}"')
        
        return ", ".join(citation_parts)

@dataclass
class LegalEmbedding:
    """Represents a chunk with its embedding vector"""
    chunk: LegalChunk
    embedding: np.ndarray
    embedding_model: str
    embedding_timestamp: str

class MalaysianLegalChunker:
    """
    Advanced chunker for Malaysian legal documents
    
    Handles:
    - Section-based chunking (preserves legal structure)
    - Token-aware splitting for optimal embedding size
    - Legal metadata preservation
    - Citation generation
    """
    
    def __init__(self, 
                 chunk_size: int = 800, 
                 chunk_overlap: int = 100,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Initialize text splitter
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=self._count_tokens,
                separators=[
                    "\n\n",  # Paragraph breaks
                    "\n",    # Line breaks
                    ". ",    # Sentence breaks
                    ", ",    # Clause breaks
                    " ",     # Word breaks
                    ""       # Character breaks
                ]
            )
        else:
            self.text_splitter = None
        
        # Token counter
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            except:
                self.encoding = None
        else:
            self.encoding = None
        
        logger.info(f"Initialized chunker: {chunk_size} tokens, {chunk_overlap} overlap")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def chunk_by_sections(self, document: Dict[str, Any]) -> List[LegalChunk]:
        """
        Chunk document by preserving legal sections first, then splitting large sections
        """
        chunks = []
        
        act_number = document.get('act_number', 'Unknown')
        act_title = document.get('act_title', 'Unknown Act')
        language = document.get('language', 'EN')
        sections = document.get('sections', [])
        
        for section in sections:
            section_chunks = self._chunk_single_section(
                section, act_number, act_title, language
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _chunk_single_section(self, section: Dict[str, Any], 
                            act_number: str, act_title: str, language: str) -> List[LegalChunk]:
        """Chunk a single legal section"""
        chunks = []
        
        section_id = section.get('section_id', 'unknown')
        section_heading = section.get('heading', '')
        section_content = section.get('content', '')
        section_number = section.get('section_number')
        page_number = section.get('page_number')
        
        # Skip very short sections
        if len(section_content.strip()) < self.min_chunk_size:
            return chunks
        
        # If section is small enough, keep as single chunk
        token_count = self._count_tokens(section_content)
        
        if token_count <= self.chunk_size:
            chunk = LegalChunk(
                chunk_id=f"{act_number}_{section_id}_1",
                act_number=act_number,
                act_title=act_title,
                language=language,
                section_id=section_id,
                section_heading=section_heading,
                chunk_text=section_content,
                chunk_index=1,
                token_count=token_count,
                page_number=page_number,
                section_number=section_number,
                metadata={
                    'original_section': section,
                    'is_complete_section': True
                }
            )
            chunks.append(chunk)
            
        else:
            # Split large section into smaller chunks
            if self.text_splitter:
                text_chunks = self.text_splitter.split_text(section_content)
            else:
                # Basic fallback splitting
                text_chunks = self._basic_split(section_content)
            
            for i, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunk = LegalChunk(
                        chunk_id=f"{act_number}_{section_id}_{i+1}",
                        act_number=act_number,
                        act_title=act_title,
                        language=language,
                        section_id=section_id,
                        section_heading=section_heading,
                        chunk_text=chunk_text,
                        chunk_index=i + 1,
                        token_count=self._count_tokens(chunk_text),
                        page_number=page_number,
                        section_number=section_number,
                        metadata={
                            'original_section': section,
                            'is_complete_section': False,
                            'total_chunks_in_section': len(text_chunks)
                        }
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _basic_split(self, text: str) -> List[str]:
        """Basic text splitting fallback"""
        # Split by double newlines first (paragraphs)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if self._count_tokens(current_chunk + paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Single paragraph is too long, split by sentences
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if self._count_tokens(current_chunk + sentence) > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += sentence + ". "
            else:
                current_chunk += paragraph + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

class MalaysianLegalEmbedder:
    """
    Privacy-friendly embedding generator for Malaysian legal documents
    
    Uses open-source models that run locally:
    - BAAI/bge-base-en-v1.5 (general purpose, good for legal)
    - sentence-transformers/all-MiniLM-L6-v2 (lightweight)
    - thenlper/gte-base (legal-friendly)
    """
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.model = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ sentence-transformers not available")
            return
        
        try:
            logger.info(f"📥 Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            # Fallback to smaller model
            try:
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                logger.info(f"🔄 Trying fallback model: {fallback_model}")
                self.model = SentenceTransformer(fallback_model)
                self.model_name = fallback_model
                logger.info(f"✅ Fallback model loaded")
            except Exception as e2:
                logger.error(f"❌ Fallback model also failed: {e2}")
    
    def embed_chunks(self, chunks: List[LegalChunk], batch_size: int = 32) -> List[LegalEmbedding]:
        """Generate embeddings for legal chunks in batches"""
        if not self.model:
            logger.error("❌ No embedding model available")
            return []
        
        embeddings = []
        
        logger.info(f"🔄 Generating embeddings for {len(chunks)} chunks")
        
        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [chunk.chunk_text for chunk in batch_chunks]
            
            try:
                # Generate embeddings
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Create embedding objects
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    legal_embedding = LegalEmbedding(
                        chunk=chunk,
                        embedding=embedding,
                        embedding_model=self.model_name,
                        embedding_timestamp=str(Path().cwd())  # Simple timestamp
                    )
                    embeddings.append(legal_embedding)
                    
            except Exception as e:
                logger.error(f"❌ Batch embedding failed: {e}")
                continue
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings

class MalaysianLegalProcessor:
    """
    Complete processing pipeline for Malaysian legal documents
    
    Orchestrates:
    - Loading parsed documents
    - Chunking by legal sections
    - Embedding generation
    - Output preparation for vector database
    """
    
    def __init__(self, 
                 parsed_dir: str = "./parsed",
                 output_dir: str = "./embeddings",
                 embedding_model: str = "BAAI/bge-base-en-v1.5"):
        
        self.parsed_dir = Path(parsed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.chunker = MalaysianLegalChunker()
        self.embedder = MalaysianLegalEmbedder(embedding_model)
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'en_chunks': 0,
            'bm_chunks': 0,
            'failed_documents': 0
        }
        
        logger.info(f"Initialized Malaysian Legal Processor")
        logger.info(f"Input: {self.parsed_dir}")
        logger.info(f"Output: {self.output_dir}")
    
    def load_parsed_documents(self) -> List[Dict[str, Any]]:
        """Load all parsed documents from JSON files"""
        documents = []
        
        for lang_dir in ['EN', 'BM']:
            lang_path = self.parsed_dir / lang_dir
            if not lang_path.exists():
                logger.warning(f"⚠️ Language directory not found: {lang_path}")
                continue
            
            json_files = list(lang_path.glob("*.json"))
            logger.info(f"📁 Found {len(json_files)} {lang_dir} documents")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        document = json.load(f)
                        document['source_file'] = str(json_file)
                        documents.append(document)
                except Exception as e:
                    logger.error(f"❌ Failed to load {json_file}: {e}")
                    self.stats['failed_documents'] += 1
        
        logger.info(f"📚 Loaded {len(documents)} total documents")
        return documents
    
    def process_all_documents(self) -> List[LegalEmbedding]:
        """Process all documents through chunking and embedding pipeline"""
        logger.info("🇲🇾 Starting Malaysian Legal Document Processing")
        logger.info("=" * 60)
        
        # Load documents
        documents = self.load_parsed_documents()
        if not documents:
            logger.error("❌ No documents to process")
            return []
        
        all_embeddings = []
        
        # Process each document
        for document in tqdm(documents, desc="Processing documents"):
            try:
                # Chunk document
                chunks = self.chunker.chunk_by_sections(document)
                
                if chunks:
                    # Generate embeddings
                    embeddings = self.embedder.embed_chunks(chunks)
                    all_embeddings.extend(embeddings)
                    
                    # Update statistics
                    self.stats['total_chunks'] += len(chunks)
                    self.stats['total_embeddings'] += len(embeddings)
                    
                    lang = document.get('language', 'EN')
                    if lang == 'EN':
                        self.stats['en_chunks'] += len(chunks)
                    else:
                        self.stats['bm_chunks'] += len(chunks)
                
                self.stats['documents_processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process document: {e}")
                self.stats['failed_documents'] += 1
        
        # Save embeddings
        if all_embeddings:
            self.save_embeddings(all_embeddings)
        
        # Print summary
        self.print_processing_summary()
        
        return all_embeddings
    
    def save_embeddings(self, embeddings: List[LegalEmbedding]):
        """Save embeddings and metadata for vector database ingestion"""
        logger.info(f"💾 Saving {len(embeddings)} embeddings...")
        
        # Prepare data for vector database
        chunks_data = []
        embedding_matrix = []
        
        for emb in embeddings:
            chunk_dict = asdict(emb.chunk)
            chunk_dict['embedding_model'] = emb.embedding_model
            chunk_dict['embedding_timestamp'] = emb.embedding_timestamp
            
            chunks_data.append(chunk_dict)
            embedding_matrix.append(emb.embedding)
        
        # Save chunks metadata as JSON
        chunks_file = self.output_dir / "legal_chunks_metadata.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save embeddings as numpy array
        embeddings_file = self.output_dir / "legal_embeddings.npy"
        np.save(embeddings_file, np.array(embedding_matrix))
        
        # Save combined data as pickle for easy loading
        combined_file = self.output_dir / "legal_embeddings_complete.pkl"
        with open(combined_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save processing info
        info_file = self.output_dir / "processing_info.json"
        with open(info_file, 'w') as f:
            json.dump({
                'total_embeddings': len(embeddings),
                'embedding_model': embeddings[0].embedding_model if embeddings else None,
                'embedding_dimension': embeddings[0].embedding.shape[0] if embeddings else None,
                'statistics': self.stats,
                'files_created': [
                    str(chunks_file.name),
                    str(embeddings_file.name),
                    str(combined_file.name)
                ]
            }, f, indent=2)
        
        logger.info(f"✅ Saved to {self.output_dir}")
        logger.info(f"   📄 Metadata: {chunks_file.name}")
        logger.info(f"   🔢 Embeddings: {embeddings_file.name}")
        logger.info(f"   📦 Complete: {combined_file.name}")
    
    def print_processing_summary(self):
        """Print processing statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 MALAYSIAN LEGAL PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📚 Documents processed: {self.stats['documents_processed']}")
        logger.info(f"❌ Failed documents: {self.stats['failed_documents']}")
        logger.info(f"📄 Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"🔢 Total embeddings: {self.stats['total_embeddings']}")
        logger.info(f"🇬🇧 English chunks: {self.stats['en_chunks']}")
        logger.info(f"🇲🇾 Bahasa Malaysia chunks: {self.stats['bm_chunks']}")
        
        if self.stats['total_chunks'] > 0:
            avg_chunks = self.stats['total_chunks'] / max(self.stats['documents_processed'], 1)
            logger.info(f"📈 Average chunks per document: {avg_chunks:.1f}")


def main():
    """Main function to run the chunking and embedding pipeline"""
    print("🇲🇾 MALAYSIAN LEGAL CHUNKER & EMBEDDER")
    print("=" * 50)
    print("Creating semantic chunks and embeddings for vector search")
    print()
    
    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers not installed")
        print("📋 Run: pip install sentence-transformers")
        return
    
    # Check if parsed documents exist
    parsed_dir = Path("./parsed")
    if not parsed_dir.exists():
        print("❌ Parsed documents directory not found")
        print("📋 Run parse_pdf_folder.py first to create parsed documents")
        return
    
    # Check for documents
    en_docs = len(list((parsed_dir / "EN").glob("*.json"))) if (parsed_dir / "EN").exists() else 0
    bm_docs = len(list((parsed_dir / "BM").glob("*.json"))) if (parsed_dir / "BM").exists() else 0
    
    if en_docs + bm_docs == 0:
        print("❌ No parsed documents found")
        print("📋 Run parse_pdf_folder.py first")
        return
    
    print(f"📚 Found {en_docs} EN + {bm_docs} BM documents to process")
    
    # Initialize and run processor
    processor = MalaysianLegalProcessor()
    embeddings = processor.process_all_documents()
    
    if embeddings:
        print(f"\n🎉 Processing complete!")
        print(f"📦 {len(embeddings)} embeddings ready for vector database")
        print(f"📁 Output saved to: ./embeddings/")
        print(f"📋 Next step: Run upload_to_vectordb.py to create searchable index")
    else:
        print("\n❌ No embeddings generated")

if __name__ == "__main__":
    main()
