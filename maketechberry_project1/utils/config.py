"""Configuration settings"""
import os
from dotenv import load_dotenv

load_dotenv()

# Model paths
TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", "ml_models/text_model/model.pkl")
IMAGE_MODEL_PATH = os.getenv("IMAGE_MODEL_PATH", "ml_models/image_model/model.pth")
FUSION_MODEL_PATH = os.getenv("FUSION_MODEL_PATH", "ml_models/multimodal/fusion_model.pkl")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./verification.db")

# API
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 8000))

# Inference thresholds
TEXT_CONFIDENCE_THRESHOLD = float(os.getenv("TEXT_CONFIDENCE_THRESHOLD", 0.7))
IMAGE_CONFIDENCE_THRESHOLD = float(os.getenv("IMAGE_CONFIDENCE_THRESHOLD", 0.7))
FUSION_CONFIDENCE_THRESHOLD = float(os.getenv("FUSION_CONFIDENCE_THRESHOLD", 0.75))

# Dataset paths
DATASET_PATH = os.getenv("DATASET_PATH", "dataset/")
