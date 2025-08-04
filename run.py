#!/usr/bin/env python3
"""
Main entry point for the Multimodal RAG application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import torch
        import transformers
        import langchain
        import streamlit
        import pymupdf
        import PIL
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_environment():
    """Check if environment variables are set"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please create a .env file with your OPENAI_API_KEY")
        return False
    
    # Load and check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please add OPENAI_API_KEY to your .env file")
        return False
    
    print("✅ Environment variables configured")
    return True

def run_streamlit():
    """Run the Streamlit application"""
    frontend_path = Path("frontend/streamlit_app.py")
    
    if not frontend_path.exists():
        print(f"❌ Frontend file not found: {frontend_path}")
        return False
    
    print("🚀 Starting Multimodal RAG application...")
    print("📍 Application will open in your default browser")
    print("🔗 Default URL: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(frontend_path),
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🔧 Multimodal RAG Application Startup")
    print("="*40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run application
    if not run_streamlit():
        sys.exit(1)

if __name__ == "__main__":
    main()