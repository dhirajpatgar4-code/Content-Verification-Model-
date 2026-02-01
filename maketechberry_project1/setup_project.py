#!/usr/bin/env python3
"""
Complete project setup script
"""

import os
import subprocess
import sys

def create_directory_structure():
    """Create the complete directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "ml_models/text_model",
        "ml_models/image_model", 
        "ml_models/multimodal",
        "dataset/Education_PhysicalEdu_Dataset",
        "dataset/text_data",
        "dataset/metadata",
        "web_app/templates",
        "web_app/static/css",
        "web_app/static/js",
        "web_app/static/images",
        "api",
        "database",
        "training",
        "inference",
        "utils",
        "uploads",
        "config",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}")
    
    print("✅ Directory structure created")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    requirements = [
        "torch==2.1.0",
        "torchvision==0.16.0",
        "transformers==4.36.2",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "flask==3.0.0",
        "sqlalchemy==2.0.23",
        "pydantic==2.5.0",
        "pillow==10.1.0",
        "scikit-learn==1.3.2",
        "pandas==2.1.3",
        "numpy==1.24.3",
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  Installed: {package}")
        except:
            print(f"  Failed to install: {package}")
    
    print("✅ Dependencies installed")

def create_environment_file():
    """Create .env file"""
    print("🔧 Creating environment file...")
    
    env_content = """# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
WEB_PORT=5000
DEBUG=True

# Database
DATABASE_URL=sqlite:///content_verification.db

# Model Paths
TEXT_MODEL_PATH=ml_models/text_model/
IMAGE_MODEL_PATH=ml_models/image_model/
MULTIMODAL_PATH=ml_models/multimodal/

# File Uploads
UPLOAD_DIR=uploads
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=jpg,jpeg,png,gif

# ML Settings
MIN_CONFIDENCE=0.3
CONFIDENCE_THRESHOLD=0.6
DOMAIN_MATCH_THRESHOLD=0.5

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Environment file created")

def create_config_files():
    """Create configuration files"""
    print("⚙️ Creating configuration files...")
    
    # Business rules config
    business_rules = {
        "restricted_categories": [
            "weapons", "drugs", "adult_content", "gambling",
            "explosives", "poisons", "counterfeit", "human_organs",
            "surveillance_devices", "live_animals"
        ],
        "confidence_thresholds": {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        },
        "domain_thresholds": {
            "strict": 0.7,
            "moderate": 0.5,
            "lenient": 0.3
        },
        "related_domains": {
            "education": ["tech", "sports", "health"],
            "sports": ["health", "education"],
            "health": ["sports", "education"],
            "tech": ["education", "electronics"],
            "fashion": ["beauty", "home"],
            "travel": ["entertainment", "home"],
            "real_estate": ["home", "finance"]
        }
    }
    
    os.makedirs("config", exist_ok=True)
    import json
    with open("config/business_rules.json", "w") as f:
        json.dump(business_rules, f, indent=2)
    
    print("✅ Configuration files created")

def main():
    """Main setup function"""
    print("="*60)
    print("🤖 CONTENT VERIFICATION SYSTEM - SETUP")
    print("="*60)
    
    create_directory_structure()
    print("-"*60)
    
    create_environment_file()
    print("-"*60)
    
    create_config_files()
    print("-"*60)
    
    print("📦 Would you like to install dependencies? (y/n)")
    choice = input("> ").lower().strip()
    
    if choice == 'y':
        install_dependencies()
    else:
        print("⚠️ Skipping dependency installation")
    
    print("="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Upload your Education_PhysicalEdu_Dataset to dataset/ folder")
    print("2. Run: python create_model_files.py")
    print("3. Run: python training/train_text_model.py")
    print("4. Run: python training/train_image_model.py")
    print("5. Start the API: python api/endpoints.py")
    print("6. Start the web app: python web_app/app.py")
    print("\nAccess the system at:")
    print("  • API: http://localhost:8000")
    print("  • Web Interface: http://localhost:5000")

if __name__ == "__main__":
    main()