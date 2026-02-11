#!/usr/bin/env python3
"""
Main entry point for the Content Verification API
Run with: python main.py
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the FastAPI app
from api.endpoints import app

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Content Verification API...")
    print("📝 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)