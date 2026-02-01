"""Input validators"""
import re
from typing import Tuple

def validate_text(text: str, min_length: int = 5, max_length: int = 10000) -> Tuple[bool, str]:
    """Validate text input"""
    if not text:
        return False, "Text cannot be empty"
    
    if len(text) < min_length:
        return False, f"Text must be at least {min_length} characters"
    
    if len(text) > max_length:
        return False, f"Text cannot exceed {max_length} characters"
    
    return True, "Valid"

def validate_image_path(image_path: str) -> Tuple[bool, str]:
    """Validate image file path"""
    import os
    
    if not image_path:
        return False, "Image path cannot be empty"
    
    if not os.path.exists(image_path):
        return False, "Image file does not exist"
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
        return False, "Invalid image format"
    
    return True, "Valid"

def validate_email(email: str) -> Tuple[bool, str]:
    """Validate email address"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True, "Valid"
    return False, "Invalid email format"
