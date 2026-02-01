"""Helper functions"""
import hashlib
import json
from typing import Any, Dict

def hash_content(content: str) -> str:
    """Generate hash of content"""
    return hashlib.sha256(content.encode()).hexdigest()

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage"""
    return f"{confidence * 100:.2f}%"

def log_verification(content_type: str, verified: bool, confidence: float) -> Dict[str, Any]:
    """Create verification log entry"""
    return {
        "content_type": content_type,
        "verified": verified,
        "confidence": confidence
    }
