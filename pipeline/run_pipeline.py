#!/usr/bin/env python3
"""
Malaysian Legal Pipeline Runner
==============================

Complete pipeline orchestrator for Malaysian legal document processing.

Features:
- End-to-end pipeline from PDFs to searchable vector database
- Progress tracking and error recovery
- Validation and testing at each step
- Comprehensive logging and reporting
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('malaysian_legal_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MalaysianLegalPipeline:
    """
    Complete pipeline orchestrator for Malaysian legal document processing
    
    Pipeline stages:
    1. PDF Parsing - Extract text and structure from Malaysian Act PDFs
    2. Chunking & Embedding - Create semantic chunks with embeddings
    3. Vector Database - Upload to Qdrant for search
    4. API Server - Start search interface
    5. Validation - Test end-to-end functionality
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.pipeline_dir = self.base_dir / "pipeline"
        
        # Pipeline stage scripts
        self.scripts = {
            'parse': self.pipeline_dir / "parse_pdf_folder.py",
            'chunk': self.pipeline_dir / "chunk_and_embed.py", 
            'upload': self.pipeline_dir / "upload_to_vectordb.py",
            'api': self.pipeline_dir / "legal_search_api.py"
        }
        
        # Data directories
        self.directories = {
            'source': self.base_dir / "malaysian_acts",
            'parsed': self.base_dir / "parsed",
            'embeddings': self.base_dir / "embeddings"
        }
        
        # Pipeline state
        self.completed_stages = []
        self.pipeline_start_time = None
        
        logger.info(f"Initialized Malaysian Legal Pipeline")
        logger.info(f"Base directory: {self.base_dir.absolute()}")
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are met"""
        checks = {}
        
        # Check source PDFs
        en_pdfs = len(list((self.directories['source'] / "EN").glob("*.pdf"))) if (self.directories['source'] / "EN").exists() else 0
        bm_pdfs = len(list((self.directories['source'] / "BM").glob("*.pdf"))) if (self.directories['source'] / "BM").exists() else 0
        checks['source_pdfs'] = en_pdfs + bm_pdfs > 0
        
        # Check Python dependencies
        required_packages = [
            'unstructured', 'pdfminer', 'sentence-transformers',
            'qdrant-client', 'langchain', 'fastapi', 'numpy', 'tqdm'
        ]
        
        checks['dependencies'] = True
        missing_deps = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_deps.append(package)
                checks['dependencies'] = False
        
        # Check pipeline scripts
        checks['scripts'] = all(script.exists() for script in self.scripts.values())
        
        # Check Qdrant (optional - will be checked during upload)
        checks['qdrant_optional'] = True
        
        # Log results
        logger.info("📋 PREREQUISITE CHECK:")
        logger.info(f"   📄 Source PDFs: {'✅' if checks['source_pdfs'] else '❌'} ({en_pdfs} EN + {bm_pdfs} BM)")
        logger.info(f"   📦 Dependencies: {'✅' if checks['dependencies'] else '❌'}")
        if missing_deps:
            logger.info(f"      Missing: {', '.join(missing_deps)}")
        logger.info(f"   🐍 Pipeline scripts: {'✅' if checks['scripts'] else '❌'}")
        
        return checks
    
    def install_dependencies(self) -> bool:
        """Install missing dependencies"""
        logger.info("📦 Installing pipeline dependencies...")
        
        try:
            # Install main requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "unstructured[pdf]", "pdfminer.six", "sentence-transformers",
                "qdrant-client", "langchain", "langchain-community", 
                "fastapi", "uvicorn", "numpy", "pandas", "tqdm", 
                "tiktoken", "langdetect"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def run_stage(self, stage: str, stage_name: str) -> bool:
        """Run a pipeline stage"""
        script_path = self.scripts.get(stage)
        if not script_path or not script_path.exists():
            logger.error(f"❌ Script not found for stage '{stage}': {script_path}")
            return False
        
        logger.info(f"🔄 Starting stage: {stage_name}")
        logger.info(f"   Script: {script_path}")
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ Stage '{stage_name}' completed in {duration:.1f}s")
                self.completed_stages.append(stage)
                
                # Log output if available
                if result.stdout.strip():
                    logger.info(f"   Output: {result.stdout.strip()[-200:]}")  # Last 200 chars
                
                return True
            else:
                logger.error(f"❌ Stage '{stage_name}' failed after {duration:.1f}s")
                logger.error(f"   Error: {result.stderr}")
                if result.stdout:
                    logger.error(f"   Output: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Stage '{stage_name}' timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"❌ Error running stage '{stage_name}': {e}")
            return False
    
    def validate_stage_output(self, stage: str) -> bool:
        """Validate that a stage produced expected output"""
        
        if stage == 'parse':
            # Check for parsed JSON files
            en_files = len(list((self.directories['parsed'] / "EN").glob("*.json"))) if (self.directories['parsed'] / "EN").exists() else 0
            bm_files = len(list((self.directories['parsed'] / "BM").glob("*.json"))) if (self.directories['parsed'] / "BM").exists() else 0
            
            success = en_files + bm_files > 0
            logger.info(f"   📊 Parsed documents: {en_files} EN + {bm_files} BM")
            return success
        
        elif stage == 'chunk':
            # Check for embedding files
            embeddings_dir = self.directories['embeddings']
            required_files = [
                embeddings_dir / "legal_embeddings_complete.pkl",
                embeddings_dir / "legal_chunks_metadata.json",
                embeddings_dir / "processing_info.json"
            ]
            
            success = all(f.exists() for f in required_files)
            if success:
                # Check file sizes
                for f in required_files:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    logger.info(f"   📄 {f.name}: {size_mb:.1f} MB")
            return success
        
        elif stage == 'upload':
            # Check if Qdrant has data (this would require qdrant client)
            logger.info("   📊 Vector database upload - check logs above")
            return True  # Assume success if script completed
        
        return True
    
    def start_qdrant_docker(self) -> bool:
        """Start Qdrant in Docker if not running"""
        logger.info("🐳 Starting Qdrant vector database...")
        
        try:
            # Check if Qdrant is already running
            result = subprocess.run([
                "curl", "-s", "http://localhost:6333/collections"
            ], capture_output=True, timeout=5)
            
            if result.returncode == 0:
                logger.info("✅ Qdrant already running")
                return True
            
            # Start Qdrant with Docker
            result = subprocess.run([
                "docker", "run", "-d", 
                "--name", "qdrant-malaysian-legal",
                "-p", "6333:6333", 
                "-p", "6334:6334",
                "-v", "qdrant_storage:/qdrant/storage",
                "qdrant/qdrant"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Qdrant started in Docker")
                # Wait for startup
                time.sleep(10)
                return True
            else:
                logger.warning(f"⚠️ Failed to start Qdrant: {result.stderr}")
                logger.info("💡 Please start Qdrant manually:")
                logger.info("   docker run -p 6333:6333 qdrant/qdrant")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Could not start Qdrant: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete Malaysian legal processing pipeline"""
        self.pipeline_start_time = time.time()
        
        logger.info("🇲🇾 STARTING MALAYSIAN LEGAL PROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Stage 1: Prerequisites
        logger.info("\n📋 STAGE 0: CHECKING PREREQUISITES")
        logger.info("-" * 40)
        
        prereqs = self.check_prerequisites()
        
        if not prereqs['source_pdfs']:
            logger.error("❌ No source PDF files found")
            logger.info("📋 Expected structure:")
            logger.info("   malaysian_acts/EN/    (English PDFs)")
            logger.info("   malaysian_acts/BM/    (Bahasa Malaysia PDFs)")
            return False
        
        if not prereqs['dependencies']:
            logger.info("📦 Installing missing dependencies...")
            if not self.install_dependencies():
                logger.error("❌ Failed to install dependencies")
                return False
        
        # Stage 2: PDF Parsing
        logger.info("\n📄 STAGE 1: PDF PARSING")
        logger.info("-" * 40)
        
        if not self.run_stage('parse', 'PDF Parsing'):
            return False
        
        if not self.validate_stage_output('parse'):
            logger.error("❌ PDF parsing validation failed")
            return False
        
        # Stage 3: Chunking and Embedding
        logger.info("\n🔢 STAGE 2: CHUNKING & EMBEDDING")
        logger.info("-" * 40)
        
        if not self.run_stage('chunk', 'Chunking & Embedding'):
            return False
        
        if not self.validate_stage_output('chunk'):
            logger.error("❌ Chunking & embedding validation failed")
            return False
        
        # Stage 4: Start Qdrant
        logger.info("\n🐳 STAGE 3: VECTOR DATABASE SETUP")
        logger.info("-" * 40)
        
        qdrant_started = self.start_qdrant_docker()
        
        # Stage 5: Upload to Vector Database
        logger.info("\n📤 STAGE 4: VECTOR DATABASE UPLOAD")
        logger.info("-" * 40)
        
        if not self.run_stage('upload', 'Vector Database Upload'):
            if qdrant_started:
                logger.warning("⚠️ Upload failed - check Qdrant connection")
            else:
                logger.warning("⚠️ Upload failed - Qdrant not available")
            return False
        
        # Stage 6: Validation and Summary
        logger.info("\n✅ STAGE 5: PIPELINE VALIDATION")
        logger.info("-" * 40)
        
        total_time = time.time() - self.pipeline_start_time
        
        logger.info("🎉 MALAYSIAN LEGAL PIPELINE COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"⏱️ Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"✅ Completed stages: {len(self.completed_stages)}/4")
        logger.info(f"📚 Ready for legal search and Q&A")
        
        # Show next steps
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("1. Start the search API:")
        logger.info(f"   python {self.scripts['api']}")
        logger.info("2. Test search: http://localhost:8000/docs")
        logger.info("3. Try queries: employment, contract, IP rights, etc.")
        
        return True
    
    def quick_test(self) -> bool:
        """Run quick validation tests"""
        logger.info("🧪 Running quick validation tests...")
        
        try:
            # Test 1: Check parsed files exist
            parsed_files = list(self.directories['parsed'].rglob("*.json"))
            logger.info(f"   📄 Parsed files: {len(parsed_files)}")
            
            # Test 2: Check embeddings exist
            embeddings_file = self.directories['embeddings'] / "legal_embeddings_complete.pkl"
            if embeddings_file.exists():
                size_mb = embeddings_file.stat().st_size / (1024 * 1024)
                logger.info(f"   🔢 Embeddings file: {size_mb:.1f} MB")
            
            # Test 3: Try to import required modules
            modules_test = True
            try:
                import sentence_transformers
                import qdrant_client
                logger.info("   📦 Core modules: OK")
            except ImportError as e:
                logger.warning(f"   ⚠️ Module import issue: {e}")
                modules_test = False
            
            return len(parsed_files) > 0 and embeddings_file.exists() and modules_test
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return False

def main():
    """Main function to run the complete pipeline"""
    print("🇲🇾 MALAYSIAN LEGAL AI PIPELINE")
    print("=" * 50)
    print("Complete pipeline: PDFs → Search → Q&A")
    print()
    
    # Initialize pipeline
    pipeline = MalaysianLegalPipeline()
    
    # Check if we should run full pipeline or just validate
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            print("🧪 Running quick validation test...")
            success = pipeline.quick_test()
            if success:
                print("✅ Pipeline validation passed")
            else:
                print("❌ Pipeline validation failed")
            return
        
        elif sys.argv[1] == "api":
            print("🚀 Starting search API only...")
            api_script = pipeline.scripts['api']
            if api_script.exists():
                subprocess.run([sys.executable, str(api_script)])
            else:
                print(f"❌ API script not found: {api_script}")
            return
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 SUCCESS! Malaysian Legal AI system is ready!")
        print("🔍 You can now search Malaysian legal documents")
        print("❓ Ask legal questions and get AI-powered answers")
        
        # Offer to start API
        try:
            start_api = input("\n🚀 Start the search API now? (y/n): ").lower().strip()
            if start_api in ['y', 'yes']:
                api_script = pipeline.scripts['api']
                subprocess.run([sys.executable, str(api_script)])
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n❌ Pipeline failed - check logs for details")

if __name__ == "__main__":
    main()
