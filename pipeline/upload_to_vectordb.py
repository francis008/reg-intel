#!/usr/bin/env python3
"""
Malaysian Legal Vector Database Uploader
========================================

Uploads processed Malaysian legal chunks and embeddings to Qdrant vector database.

Features:
- Qdrant integration with Malaysian legal schema
- Efficient batch uploading
- Metadata preservation for legal citations
- Search optimization for legal queries
- Bilingual support (EN/BM)
- Error handling and recovery
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid

# Vector database
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("❌ qdrant-client not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("❌ numpy not available")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MalaysianLegalVectorDB:
    """
    Vector database manager for Malaysian legal documents
    
    Features:
    - Optimized schema for legal search
    - Malaysian legal metadata indexing
    - Bilingual search support
    - Citation tracking
    - Performance optimization
    """
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6333,
                 collection_name: str = "malaysian_legal_acts"):
        
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        
        if not QDRANT_AVAILABLE:
            logger.error("❌ Qdrant client not available")
            return
        
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"✅ Connected to Qdrant at {host}:{port}")
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"📚 Available collections: {len(collections.collections)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            logger.info("💡 Make sure Qdrant is running:")
            logger.info("   Docker: docker run -p 6333:6333 qdrant/qdrant")
            logger.info("   Local: Download from https://qdrant.tech/")
    
    def create_legal_collection(self, vector_size: int = 768, distance: str = "Cosine") -> bool:
        """
        Create optimized collection for Malaysian legal documents
        """
        if not self.client:
            logger.error("❌ No Qdrant client available")
            return False
        
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if self.collection_name in existing_names:
                logger.info(f"📚 Collection '{self.collection_name}' already exists")
                
                # Get collection info
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"   📊 Points: {collection_info.points_count}")
                logger.info(f"   📏 Vector size: {collection_info.config.params.vectors.size}")
                return True
            
            # Create new collection with legal-optimized configuration
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE)
                ),
                # Optimize for legal search patterns
                optimizers_config=models.OptimizersConfig(
                    default_segment_number=2,
                    max_segment_size=20000,
                    memmap_threshold=50000,
                    indexing_threshold=20000
                ),
                # Enable payload indexing for legal metadata
                hnsw_config=models.HnswConfig(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000
                )
            )
            
            logger.info(f"✅ Created collection: {self.collection_name}")
            logger.info(f"   📏 Vector size: {vector_size}")
            logger.info(f"   📐 Distance: {distance}")
            
            # Create indexes for legal search optimization
            self._create_legal_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection: {e}")
            return False
    
    def _create_legal_indexes(self):
        """Create optimized indexes for Malaysian legal metadata"""
        try:
            # Index key legal fields for fast filtering
            legal_indexes = [
                "act_number",
                "language", 
                "section_number",
                "act_title",
                "section_id"
            ]
            
            for field in legal_indexes:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    logger.debug(f"   📇 Created index: {field}")
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"   ⚠️ Index {field}: {e}")
            
            logger.info("✅ Legal metadata indexes configured")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create some indexes: {e}")
    
    def upload_legal_embeddings(self, embeddings_data: List[Dict[str, Any]], 
                              batch_size: int = 100) -> bool:
        """
        Upload Malaysian legal embeddings with optimized batching
        """
        if not self.client:
            logger.error("❌ No Qdrant client available")
            return False
        
        if not embeddings_data:
            logger.error("❌ No embeddings data provided")
            return False
        
        logger.info(f"📤 Uploading {len(embeddings_data)} legal embeddings")
        logger.info(f"   📦 Batch size: {batch_size}")
        
        uploaded_count = 0
        failed_count = 0
        
        # Process in batches
        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]
            
            try:
                points = []
                
                for item in batch:
                    # Extract embedding and metadata
                    embedding = item.get('embedding')
                    chunk_data = item.get('chunk', {})
                    
                    if embedding is None or not chunk_data:
                        logger.warning(f"⚠️ Skipping invalid item")
                        failed_count += 1
                        continue
                    
                    # Convert numpy array to list if needed
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    
                    # Prepare legal metadata for search optimization
                    payload = {
                        # Core legal identifiers
                        "act_number": chunk_data.get('act_number', ''),
                        "act_title": chunk_data.get('act_title', ''),
                        "language": chunk_data.get('language', 'EN'),
                        "section_id": chunk_data.get('section_id', ''),
                        "section_heading": chunk_data.get('section_heading', ''),
                        "section_number": chunk_data.get('section_number', ''),
                        
                        # Chunk information
                        "chunk_text": chunk_data.get('chunk_text', ''),
                        "chunk_index": chunk_data.get('chunk_index', 1),
                        "token_count": chunk_data.get('token_count', 0),
                        
                        # Legal citation and reference
                        "citation": chunk_data.get('citation', ''),
                        "page_number": chunk_data.get('page_number'),
                        
                        # Metadata for search
                        "embedding_model": item.get('embedding_model', ''),
                        "upload_timestamp": datetime.now().isoformat(),
                        
                        # Additional metadata
                        "metadata": chunk_data.get('metadata', {})
                    }
                    
                    # Create point
                    point = PointStruct(
                        id=str(uuid.uuid4()),  # Generate unique ID
                        vector=embedding,
                        payload=payload
                    )
                    
                    points.append(point)
                
                # Upload batch
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    
                    uploaded_count += len(points)
                    logger.info(f"   ✅ Batch {i//batch_size + 1}: {len(points)} points uploaded")
                
            except Exception as e:
                logger.error(f"❌ Batch upload failed: {e}")
                failed_count += len(batch)
                continue
        
        logger.info(f"📊 Upload summary:")
        logger.info(f"   ✅ Successfully uploaded: {uploaded_count}")
        logger.info(f"   ❌ Failed: {failed_count}")
        
        # Update collection info
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"   📚 Total points in collection: {collection_info.points_count}")
        except:
            pass
        
        return uploaded_count > 0
    
    def test_legal_search(self, query_text: str = "employment contract termination", 
                         limit: int = 5) -> List[Dict[str, Any]]:
        """
        Test search functionality with a legal query
        """
        if not self.client:
            logger.error("❌ No Qdrant client available")
            return []
        
        try:
            # For testing, we'll use a simple text match
            # In production, you'd embed the query using the same model
            logger.info(f"🔍 Testing search: '{query_text}'")
            
            # Search using payload filter (since we don't have query embedding here)
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_text",
                            match=models.MatchText(text=query_text)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in search_result[0]:
                result = {
                    "id": point.id,
                    "score": getattr(point, 'score', 1.0),
                    "act_number": point.payload.get('act_number'),
                    "act_title": point.payload.get('act_title'),
                    "section_heading": point.payload.get('section_heading'),
                    "citation": point.payload.get('citation'),
                    "text_preview": point.payload.get('chunk_text', '')[:200] + "...",
                    "language": point.payload.get('language')
                }
                results.append(result)
            
            logger.info(f"   📋 Found {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"   {i}. {result['citation']} - {result['text_preview'][:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the legal collection"""
        if not self.client:
            return {"error": "No client available"}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get language distribution
            en_count = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="language", match=models.MatchValue(value="EN"))]
                )
            )
            
            bm_count = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="language", match=models.MatchValue(value="BM"))]
                )
            )
            
            return {
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": str(collection_info.config.params.vectors.distance),
                "english_chunks": en_count.count,
                "bahasa_chunks": bm_count.count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status
            }
            
        except Exception as e:
            return {"error": str(e)}

class MalaysianLegalUploader:
    """
    Complete uploader for Malaysian legal embeddings
    
    Handles:
    - Loading processed embeddings
    - Vector database setup
    - Batch uploading with error recovery
    - Validation and testing
    """
    
    def __init__(self, 
                 embeddings_dir: str = "./embeddings",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        
        self.embeddings_dir = Path(embeddings_dir)
        self.vector_db = MalaysianLegalVectorDB(qdrant_host, qdrant_port)
        
        logger.info(f"Initialized Malaysian Legal Uploader")
        logger.info(f"Embeddings: {self.embeddings_dir}")
    
    def load_embeddings_data(self) -> Optional[List[Dict[str, Any]]]:
        """Load processed embeddings and prepare for upload"""
        
        # Try to load directly from vector database first (if already uploaded)
        try:
            stats = self.vector_db.get_collection_stats()
            if stats.get('total_points', 0) > 0:
                logger.info(f"✅ Found existing vector database with {stats['total_points']} points")
                response = input("Vector database already contains data. Re-upload? (y/N): ")
                if response.lower() != 'y':
                    return None
        except:
            pass
        
        # Try to load the complete pickle file first
        pickle_file = self.embeddings_dir / "legal_embeddings_complete.pkl"
        if pickle_file.exists():
            try:
                logger.info(f"📦 Loading complete embeddings: {pickle_file}")
                with open(pickle_file, 'rb') as f:
                    embeddings_objects = pickle.load(f)
                
                # Convert to upload format
                upload_data = []
                for emb_obj in embeddings_objects:
                    item = {
                        'embedding': emb_obj.embedding,
                        'chunk': {
                            'chunk_id': emb_obj.chunk.chunk_id,
                            'act_number': emb_obj.chunk.act_number,
                            'act_title': emb_obj.chunk.act_title,
                            'language': emb_obj.chunk.language,
                            'section_id': emb_obj.chunk.section_id,
                            'section_heading': emb_obj.chunk.section_heading,
                            'chunk_text': emb_obj.chunk.chunk_text,
                            'chunk_index': emb_obj.chunk.chunk_index,
                            'token_count': emb_obj.chunk.token_count,
                            'page_number': emb_obj.chunk.page_number,
                            'section_number': emb_obj.chunk.section_number,
                            'citation': emb_obj.chunk.citation,
                            'metadata': emb_obj.chunk.metadata
                        },
                        'embedding_model': emb_obj.embedding_model,
                        'embedding_timestamp': emb_obj.embedding_timestamp
                    }
                    upload_data.append(item)
                
                logger.info(f"✅ Loaded {len(upload_data)} embeddings from pickle")
                return upload_data
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load pickle file: {e}")
        
        # Fallback: load from separate files
        metadata_file = self.embeddings_dir / "legal_chunks_metadata.json"
        embeddings_file = self.embeddings_dir / "legal_embeddings.npy"
        
        if not metadata_file.exists() or not embeddings_file.exists():
            logger.error(f"❌ Required files not found in {self.embeddings_dir}")
            logger.info("📋 Expected files:")
            logger.info(f"   📄 {metadata_file.name}")
            logger.info(f"   🔢 {embeddings_file.name}")
            return None
        
        try:
            logger.info(f"📄 Loading metadata: {metadata_file}")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                chunks_metadata = json.load(f)
            
            if not NUMPY_AVAILABLE:
                logger.error("❌ numpy not available for loading embeddings")
                return None
            
            logger.info(f"🔢 Loading embeddings: {embeddings_file}")
            embeddings_array = np.load(embeddings_file)
            
            if len(chunks_metadata) != len(embeddings_array):
                logger.error(f"❌ Mismatch: {len(chunks_metadata)} chunks vs {len(embeddings_array)} embeddings")
                return None
            
            # Combine metadata and embeddings
            upload_data = []
            for chunk_meta, embedding in zip(chunks_metadata, embeddings_array):
                item = {
                    'embedding': embedding,
                    'chunk': chunk_meta,
                    'embedding_model': chunk_meta.get('embedding_model', ''),
                    'embedding_timestamp': chunk_meta.get('embedding_timestamp', '')
                }
                upload_data.append(item)
            
            logger.info(f"✅ Loaded {len(upload_data)} embeddings from separate files")
            return upload_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            return None
    
    def upload_all_embeddings(self) -> bool:
        """Complete upload pipeline"""
        logger.info("🇲🇾 Starting Malaysian Legal Vector Database Upload")
        logger.info("=" * 60)
        
        # Load embeddings
        embeddings_data = self.load_embeddings_data()
        if not embeddings_data:
            logger.error("❌ No embeddings data to upload")
            return False
        
        # Determine vector size
        vector_size = len(embeddings_data[0]['embedding'])
        logger.info(f"📏 Vector dimension: {vector_size}")
        
        # Create collection
        if not self.vector_db.create_legal_collection(vector_size=vector_size):
            logger.error("❌ Failed to create vector collection")
            return False
        
        # Upload embeddings
        if not self.vector_db.upload_legal_embeddings(embeddings_data):
            logger.error("❌ Failed to upload embeddings")
            return False
        
        # Test search
        logger.info("\n🔍 Testing legal search functionality...")
        test_queries = [
            "employment contract termination",
            "intellectual property rights",
            "corporate governance requirements"
        ]
        
        for query in test_queries:
            results = self.vector_db.test_legal_search(query, limit=3)
            if results:
                logger.info(f"   ✅ '{query}': {len(results)} results")
            else:
                logger.info(f"   ⚠️ '{query}': No results")
        
        # Print final statistics
        stats = self.vector_db.get_collection_stats()
        logger.info(f"\n📊 FINAL COLLECTION STATISTICS:")
        logger.info(f"   📚 Total legal chunks: {stats.get('total_points', 0)}")
        logger.info(f"   🇬🇧 English chunks: {stats.get('english_chunks', 0)}")
        logger.info(f"   🇲🇾 Bahasa Malaysia chunks: {stats.get('bahasa_chunks', 0)}")
        logger.info(f"   📏 Vector dimension: {stats.get('vector_size', 0)}")
        
        return True

def main():
    """Main function to run the vector database upload"""
    print("🇲🇾 MALAYSIAN LEGAL VECTOR DATABASE UPLOADER")
    print("=" * 50)
    print("Uploading Malaysian legal embeddings to Qdrant")
    print()
    
    # Check dependencies
    if not QDRANT_AVAILABLE:
        print("❌ qdrant-client not installed")
        print("📋 Run: pip install qdrant-client")
        return
    
    if not NUMPY_AVAILABLE:
        print("❌ numpy not installed")
        print("📋 Run: pip install numpy")
        return
    
    # Check for embeddings
    embeddings_dir = Path("./embeddings")
    if not embeddings_dir.exists():
        print("❌ Embeddings directory not found")
        print("📋 Run chunk_and_embed.py first to create embeddings")
        return
    
    # Check for required files
    required_files = [
        embeddings_dir / "legal_embeddings_complete.pkl",
        embeddings_dir / "legal_chunks_metadata.json"
    ]
    
    if not any(f.exists() for f in required_files):
        print("❌ No embedding files found")
        print("📋 Run chunk_and_embed.py first")
        return
    
    print("💡 Make sure Qdrant is running:")
    print("   Docker: docker run -p 6333:6333 qdrant/qdrant")
    print("   Or install locally: https://qdrant.tech/")
    print()
    
    # Initialize and run uploader
    uploader = MalaysianLegalUploader()
    success = uploader.upload_all_embeddings()
    
    if success:
        print(f"\n🎉 Upload complete!")
        print(f"🔍 Malaysian legal search database is ready")
        print(f"📋 Next step: Create search interface with FastAPI/Streamlit")
    else:
        print(f"\n❌ Upload failed - check logs for details")

if __name__ == "__main__":
    main()
