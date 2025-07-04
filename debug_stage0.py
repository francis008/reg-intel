#!/usr/bin/env python3
"""
Debug Stage 0 - Malaysian Legal Pipeline Prerequisites
=====================================================

This script helps diagnose why the pipeline is stuck at Stage 0.
It performs detailed checks on all prerequisites and provides specific fixes.
"""

import os
import sys
import subprocess
from pathlib import Path
import importlib.util

def check_python_environment():
    """Check Python version and environment"""
    print("🐍 PYTHON ENVIRONMENT CHECK")
    print("-" * 40)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    print()

def check_directory_structure():
    """Check if required directories exist"""
    print("📁 DIRECTORY STRUCTURE CHECK")
    print("-" * 40)
    
    base_dir = Path(".")
    directories_to_check = [
        "malaysian_acts",
        "malaysian_acts/EN", 
        "malaysian_acts/BM",
        "parsed",
        "parsed/EN",
        "parsed/BM", 
        "embeddings",
        "pipeline"
    ]
    
    for dir_name in directories_to_check:
        dir_path = base_dir / dir_name
        exists = dir_path.exists()
        print(f"{'✅' if exists else '❌'} {dir_name}")
        
        if dir_name in ["malaysian_acts/EN", "malaysian_acts/BM"] and exists:
            pdf_count = len(list(dir_path.glob("*.pdf")))
            print(f"    📄 Contains {pdf_count} PDF files")
    print()

def check_pipeline_scripts():
    """Check if pipeline scripts exist"""
    print("🐍 PIPELINE SCRIPTS CHECK")
    print("-" * 40)
    
    pipeline_dir = Path("pipeline")
    required_scripts = [
        "parse_pdf_folder.py",
        "chunk_and_embed.py", 
        "upload_to_vectordb.py",
        "legal_search_api.py",
        "run_pipeline.py"
    ]
    
    for script in required_scripts:
        script_path = pipeline_dir / script
        exists = script_path.exists()
        print(f"{'✅' if exists else '❌'} {script}")
        
        if exists:
            size_kb = script_path.stat().st_size / 1024
            print(f"    📏 Size: {size_kb:.1f} KB")
    print()

def check_dependencies():
    """Check Python dependencies"""
    print("📦 PYTHON DEPENDENCIES CHECK")
    print("-" * 40)
    
    required_packages = [
        'unstructured',
        'pdfminer', 
        'sentence_transformers',
        'qdrant_client',
        'langchain',
        'fastapi',
        'numpy',
        'pandas',
        'tqdm',
        'tiktoken',
        'langdetect'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_import = package.replace('-', '_')
        try:
            spec = importlib.util.find_spec(package_import)
            if spec is not None:
                print(f"✅ {package}")
                try:
                    module = importlib.import_module(package_import)
                    if hasattr(module, '__version__'):
                        print(f"    📌 Version: {module.__version__}")
                except:
                    pass
            else:
                print(f"❌ {package}")
                missing_packages.append(package)
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("\n💡 TO FIX - Run this command:")
        print(f"pip install {' '.join(missing_packages)}")
    
    print()
    return len(missing_packages) == 0

def check_system_requirements():
    """Check system requirements"""
    print("🖥️ SYSTEM REQUIREMENTS CHECK")
    print("-" * 40)
    
    # Check available disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"💾 Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 2:
            print("⚠️ Warning: Less than 2GB free space available")
    except:
        print("❓ Could not check disk space")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        print(f"🧠 Available RAM: {available_gb:.1f} GB")
        
        if available_gb < 4:
            print("⚠️ Warning: Less than 4GB RAM available")
    except ImportError:
        print("❓ Could not check memory (psutil not installed)")
    
    print()

def run_minimal_import_test():
    """Test minimal imports that pipeline needs"""
    print("🧪 MINIMAL IMPORT TEST")
    print("-" * 40)
    
    critical_imports = [
        ("os", "os"),
        ("sys", "sys"), 
        ("pathlib", "from pathlib import Path"),
        ("json", "import json"),
        ("subprocess", "import subprocess")
    ]
    
    for name, import_statement in critical_imports:
        try:
            exec(import_statement)
            print(f"✅ {name}")
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print()

def suggest_fixes():
    """Provide specific fix suggestions"""
    print("🔧 SUGGESTED FIXES")
    print("-" * 40)
    
    print("1. If missing PDFs:")
    print("   - Download Malaysian Act PDFs")
    print("   - Place in malaysian_acts/EN/ and malaysian_acts/BM/")
    print()
    
    print("2. If missing dependencies:")
    print("   pip install -r requirements.txt")
    print("   # OR install individually:")
    print("   pip install unstructured[pdf] pdfminer.six sentence-transformers")
    print("   pip install qdrant-client langchain fastapi uvicorn")
    print()
    
    print("3. If missing scripts:")
    print("   - Ensure all pipeline/*.py files exist")
    print("   - Check file permissions")
    print()
    
    print("4. If environment issues:")
    print("   - Try creating fresh virtual environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # macOS/Linux")
    print("   pip install -r requirements.txt")
    print()

def main():
    """Run complete diagnostic"""
    print("🔍 MALAYSIAN LEGAL PIPELINE - STAGE 0 DIAGNOSTIC")
    print("=" * 60)
    print()
    
    check_python_environment()
    check_directory_structure()
    check_pipeline_scripts()
    dependencies_ok = check_dependencies()
    check_system_requirements()
    run_minimal_import_test()
    suggest_fixes()
    
    print("🎯 SUMMARY")
    print("-" * 40)
    if dependencies_ok:
        print("✅ All dependencies appear to be installed")
        print("🔍 Check directory structure and PDF files above")
    else:
        print("❌ Missing dependencies detected")
        print("💡 Install missing packages first, then re-run pipeline")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Fix any issues shown above")
    print("2. Re-run: python pipeline/run_pipeline.py")
    print("3. If still stuck, share this diagnostic output")

if __name__ == "__main__":
    main()
