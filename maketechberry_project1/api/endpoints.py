#!/usr/bin/env python3
"""
Content Verification API - Complete Version with Web Interface
"""

import sys
import os

# Fix Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import everything
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
import uuid
import json
from datetime import datetime
import re
import sqlite3
import io
from PIL import Image
import hashlib
import colorsys
from collections import Counter
import numpy as np
import tempfile

# Import database manager
from database.database import DatabaseManager

# ========== UNIVERSAL IMAGE CLASSIFIER IMPORT ==========
try:
    from ml_models.universal_image_classifier import get_universal_classifier
    UNIVERSAL_CLASSIFIER_AVAILABLE = True
    print("✅ Universal Image Classifier available")
except ImportError:
    UNIVERSAL_CLASSIFIER_AVAILABLE = False
    get_universal_classifier = None
    print("⚠️ Universal Image Classifier not available - using fallback")

# Initialize database manager
db_manager = DatabaseManager(os.path.join(project_root, "content_verification.db"))

# Domain-specific image characteristics
DOMAIN_IMAGE_PROFILES = {
    'education': {
        'keywords': ['classroom', 'book', 'student', 'teacher', 'school', 'laboratory', 'experiment', 
                     'desk', 'chalkboard', 'whiteboard', 'notebook', 'pen', 'pencil', 'exam'],
        'common_colors': ['blue', 'white', 'black', 'brown', 'beige'],
        'common_aspect_ratios': [1.33, 1.5],
        'min_size': (400, 300),
        'max_size': (1920, 1080)
    },
    'sports': {
        'keywords': ['sports', 'fitness', 'gym', 'equipment', 'athlete', 'game', 'match',
                     'ball', 'field', 'court', 'track', 'jersey', 'shoes', 'medal'],
        'common_colors': ['green', 'red', 'blue', 'white', 'orange'],
        'common_aspect_ratios': [1.33, 1.78, 1.85],
        'min_size': (600, 400),
        'max_size': (3840, 2160)
    },
    'food': {
        'keywords': ['food', 'restaurant', 'meal', 'dish', 'cooking', 'chef', 'kitchen',
                     'plate', 'fork', 'spoon', 'knife', 'table', 'menu', 'ingredient'],
        'common_colors': ['brown', 'red', 'yellow', 'green', 'orange'],
        'common_aspect_ratios': [1.33, 1.5, 1.78],
        'min_size': (500, 500),
        'max_size': (2000, 2000)
    },
    'tech': {
        'keywords': ['computer', 'laptop', 'mobile', 'software', 'code', 'device', 'electronic',
                     'keyboard', 'screen', 'monitor', 'processor', 'circuit', 'chip'],
        'common_colors': ['black', 'gray', 'silver', 'blue', 'white'],
        'common_aspect_ratios': [1.33, 1.6, 1.78],
        'min_size': (400, 400),
        'max_size': (2560, 1440)
    },
    'health': {
        'keywords': ['health', 'wellness', 'fitness', 'exercise', 'yoga', 'meditation', 'nutrition',
                     'doctor', 'hospital', 'medicine', 'pill', 'vitamin', 'clinic', 'therapy'],
        'common_colors': ['white', 'blue', 'green', 'red', 'gray'],
        'common_aspect_ratios': [1.33, 1.5, 1.78],
        'min_size': (400, 300),
        'max_size': (1920, 1080)
    }
}

# Initialize database with business profiles
def init_database():
    """Initialize database with default business profiles"""
    try:
        existing_profiles = [
            db_manager.get_business_profile("EDU001"),
            db_manager.get_business_profile("SPORTS001"),
            db_manager.get_business_profile("MARKET001")
        ]
        
        if not any(existing_profiles):
            print("📦 Initializing database with business profiles...")
            
            sample_businesses = [
                {
                    'business_id': 'EDU001',
                    'business_name': 'EduTech Academy',
                    'business_type': 'education',
                    'domain': 'education',
                    'allowed_domains': ['education', 'tech', 'sports', 'science', 'books'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling']
                },
                {
                    'business_id': 'SPORTS001',
                    'business_name': 'Sports Gear Hub',
                    'business_type': 'sports',
                    'domain': 'sports',
                    'allowed_domains': ['sports', 'health', 'fitness', 'outdoors', 'travel'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling']
                },
                {
                    'business_id': 'MARKET001',
                    'business_name': 'MultiShop Marketplace',
                    'business_type': 'marketplace',
                    'domain': None,
                    'allowed_domains': ['education', 'sports', 'health', 'tech', 'food', 
                                       'fashion', 'travel', 'entertainment', 'automotive',
                                       'real_estate', 'finance', 'beauty', 'home', 'electronics'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling']
                },
                {
                    'business_id': 'FOOD001',
                    'business_name': 'Foodie Delights',
                    'business_type': 'food',
                    'domain': 'food',
                    'allowed_domains': ['food', 'restaurant', 'cooking', 'health', 'travel'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling']
                },
                {
                    'business_id': 'TECH001',
                    'business_name': 'Tech Solutions Inc',
                    'business_type': 'tech',
                    'domain': 'tech',
                    'allowed_domains': ['tech', 'software', 'education', 'electronics', 'business'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling']
                }
            ]
            
            for biz in sample_businesses:
                db_manager.save_business_profile(biz)
            
            print("✅ Database initialized with business profiles!")
    except Exception as e:
        print(f"⚠️ Error initializing database: {e}")

init_database()

# ========== BUSINESS VALIDATION FUNCTIONS ==========
def is_valid_business_id(business_id: str) -> bool:
    """Check if a business ID is registered in the database"""
    if not business_id:
        return False
    profile = db_manager.get_business_profile(business_id)
    return profile is not None

def get_business_domain(business_id: str) -> Optional[str]:
    """Get the primary domain for a business"""
    profile = db_manager.get_business_profile(business_id)
    if not profile:
        return None
    return profile.get('domain')

def get_allowed_domains(business_id: str) -> List[str]:
    """Get allowed domains for a business"""
    profile = db_manager.get_business_profile(business_id)
    if not profile:
        return []
    
    allowed_domains = profile.get('allowed_domains', [])
    domain = profile.get('domain')
    
    domains = []
    if domain:
        domains.append(domain)
    if allowed_domains:
        domains.extend(allowed_domains)
    
    return list(set(domains))

# ========== UNIVERSAL IMAGE VERIFICATION ==========
def verify_image_universal(image_data: bytes, business_id: str = None, filename: str = "") -> Dict[str, Any]:
    """
    Universal image verification for ANY domain
    """
    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        
        # Get user's allowed domains
        allowed_domains = []
        if business_id and business_id != "none":
            allowed_domains = get_allowed_domains(business_id)
        
        # Get universal classifier
        classifier = get_universal_classifier() if UNIVERSAL_CLASSIFIER_AVAILABLE else None
        
        if classifier:
            # Use universal classifier
            verification = classifier.verify_content(tmp_path, allowed_domains, filename)
            
            # Get enhanced analysis for logging
            enhanced_analysis = analyze_image_enhanced(image_data, filename)
            
            result = {
                **verification,
                'image_analysis': enhanced_analysis,
                'verification_method': 'universal_classifier',
                'filename': filename,
                'business_id': business_id
            }
        else:
            # Fallback to existing method
            enhanced_analysis = analyze_image_enhanced(image_data, filename)
            
            # Use existing detection (but with universal domains)
            allowed_domains_list = allowed_domains if allowed_domains else list(DOMAIN_IMAGE_PROFILES.keys())
            detection = detect_best_domain_for_image(enhanced_analysis, filename, allowed_domains_list)
            
            prediction = {
                "category": detection['category'],
                "confidence": detection['confidence'],
                "is_restricted": detection['is_restricted'],
                "source": detection.get('source', 'image_analysis')
            }
            
            decision = make_decision(prediction, business_id)
            
            result = {
                'is_allowed': decision['is_allowed'],
                'decision': decision['decision'],
                'reason': decision['reason'],
                'detected_domain': prediction['category'],
                'confidence': prediction['confidence'],
                'severity': decision['severity'],
                'requires_review': decision['requires_review'],
                'image_analysis': enhanced_analysis,
                'verification_method': 'legacy_fallback',
                'filename': filename,
                'business_id': business_id
            }
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return result
        
    except Exception as e:
        print(f"❌ Universal verification error: {e}")
        return {
            'is_allowed': False,
            'decision': 'error',
            'reason': f'Verification error: {str(e)}',
            'detected_domain': 'unknown',
            'confidence': 0.0,
            'severity': 'high',
            'requires_review': True,
            'error': str(e)
        }

# ========== TEXT CLASSIFIER ==========
def detect_content_type(text, business_id=None):
    """Keyword-based content detection with improved accuracy"""
    text_lower = text.lower()
    text_words = text_lower.split()
    
    # RESTRICTED CONTENT (HIGHEST PRIORITY)
    weapons_keywords = [
        'gun', 'weapon', 'firearm', 'rifle', 'pistol', 'ammo', 'ammunition',
        'bullet', 'shoot', 'kill', 'knife', 'sword', 'bomb', 'explosive', 'military'
    ]
    weapons_count = sum(1 for word in text_words if any(kw in word for kw in weapons_keywords))
    if weapons_count >= 1:
        confidence = min(0.80 + (weapons_count * 0.05), 0.99)
        return {
            "category": "weapons",
            "confidence": confidence,
            "is_restricted": True,
            "keywords_found": ["weapons"]
        }
    
    drugs_keywords = [
        'drug', 'cocaine', 'heroin', 'marijuana', 'opioid', 'prescription',
        'pill', 'tablet', 'narcotic', 'addict', 'cannabis', 'meth'
    ]
    drugs_count = sum(1 for word in text_words if any(kw in word for kw in drugs_keywords))
    if drugs_count >= 1:
        confidence = min(0.80 + (drugs_count * 0.05), 0.99)
        return {
            "category": "drugs", 
            "confidence": confidence,
            "is_restricted": True,
            "keywords_found": ["drugs"]
        }
    
    adult_keywords = [
        'porn', 'xxx', 'adult', 'nude', 'explicit', 'nsfw', 'erotic',
        'hentai', 'sex', 'naked'
    ]
    adult_count = sum(1 for word in text_words if any(kw in word for kw in adult_keywords))
    if adult_count >= 1:
        confidence = min(0.80 + (adult_count * 0.05), 0.99)
        return {
            "category": "adult_content",
            "confidence": confidence,
            "is_restricted": True,
            "keywords_found": ["adult_content"]
        }
    
    gambling_keywords = [
        'casino', 'gambling', 'betting', 'poker', 'lottery', 'bet',
        'wager', 'slot', 'blackjack', 'roulette'
    ]
    gambling_count = sum(1 for word in text_words if any(kw in word for kw in gambling_keywords))
    if gambling_count >= 1:
        confidence = min(0.80 + (gambling_count * 0.05), 0.99)
        return {
            "category": "gambling",
            "confidence": confidence,
            "is_restricted": True,
            "keywords_found": ["gambling"]
        }
    
    # Domain-specific keywords with improved matching
    domain_keywords = {
        'education': [
            'classroom', 'learning', 'student', 'teacher', 'school', 'education',
            'course', 'lesson', 'academic', 'university', 'college', 'training',
            'study', 'exam', 'test', 'homework', 'lecture', 'instructor', 'textbook',
            'curriculum', 'lab', 'experiment', 'research', 'thesis', 'dissertation'
        ],
        'sports': [
            'sports', 'football', 'basketball', 'cricket', 'gym', 'fitness',
            'exercise', 'training', 'yoga', 'workout', 'coach', 'team', 'game',
            'match', 'tournament', 'competition', 'athlete', 'player', 'sport',
            'tennis', 'soccer', 'volleyball', 'badminton', 'cycling', 'swimming'
        ],
        'food': [
            'food', 'restaurant', 'recipe', 'cooking', 'meal', 'pizza', 'burger',
            'pasta', 'rice', 'chef', 'kitchen', 'dining', 'dinner', 'lunch',
            'breakfast', 'snack', 'cuisine', 'dish', 'cook', 'bake',
            'ingredient', 'flavor', 'taste', 'cafe', 'bakery', 'dessert'
        ],
        'tech': [
            'tech', 'computer', 'software', 'programming', 'code', 'python',
            'java', 'app', 'website', 'laptop', 'mobile', 'phone', 'device',
            'electronic', 'digital', 'program', 'application', 'hardware',
            'network', 'database', 'algorithm', 'data', 'server', 'cloud'
        ],
        'health': [
            'health', 'wellness', 'fitness', 'exercise', 'yoga', 'meditation',
            'nutrition', 'diet', 'hospital', 'doctor', 'medicine', 'treatment',
            'therapy', 'medical', 'clinic', 'pharmacy', 'nurse', 'patient',
            'disease', 'illness', 'symptom', 'vitamin', 'supplement', 'wellbeing'
        ]
    }
    
    scores = {}
    keywords_matched = {}
    
    for domain, keywords in domain_keywords.items():
        match_count = 0
        matched_kw = []
        for word in text_words:
            for keyword in keywords:
                if keyword in word:
                    match_count += 1
                    if keyword not in matched_kw:
                        matched_kw.append(keyword)
        
        if match_count > 0:
            # Confidence formula: higher with more matches
            confidence = min(0.5 + (match_count * 0.08), 0.95)
            scores[domain] = confidence
            keywords_matched[domain] = matched_kw[:5]  # Top 5 keywords
    
    if scores:
        best_domain = max(scores, key=scores.get)
        return {
            "category": best_domain,
            "confidence": scores[best_domain],
            "is_restricted": False,
            "keywords_found": keywords_matched.get(best_domain, [])
        }
    
    return {
        "category": "unknown",
        "confidence": 0.2,
        "is_restricted": False,
        "keywords_found": []
    }

# ========== ENHANCED IMAGE ANALYSIS ==========
def rgb_to_color_name(rgb):
    """Convert RGB to color name with improved accuracy"""
    if isinstance(rgb, int):
        # Grayscale image
        if rgb < 85:
            return "black"
        elif rgb < 170:
            return "gray"
        else:
            return "white"
    
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    h_deg = h * 360
    
    # Brightness/Value based classification first
    if v < 0.15:
        return "black"
    elif v > 0.85 and s < 0.15:
        return "white"
    elif s < 0.15:
        return "gray"
    elif s < 0.3:
        return "grayish"
    
    # Hue-based classification
    if h_deg < 15 or h_deg >= 345:
        return "red"
    elif h_deg < 45:
        return "orange"
    elif h_deg < 65:
        return "yellow"
    elif h_deg < 155:
        return "green"
    elif h_deg < 190:
        return "cyan"
    elif h_deg < 260:
        return "blue"
    elif h_deg < 290:
        return "purple"
    else:
        return "pink"

def get_dominant_colors(image, num_colors=5):
    """Get dominant colors from image with improved extraction"""
    try:
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Reduce image for faster processing
        img_small = image.resize((50, 50))
        pixels = list(img_small.getdata())
        
        # Count pixel frequencies
        color_counts = Counter(pixels)
        most_common = color_counts.most_common(num_colors)
        
        colors = []
        for color, count in most_common:
            color_name = rgb_to_color_name(color)
            # Avoid duplicate color names
            if color_name not in colors:
                colors.append(color_name)
        
        # Ensure we have at least one color
        if not colors:
            colors = ["unknown"]
        
        return colors
    except Exception as e:
        return ["unknown"]

def analyze_image_enhanced(image_data: bytes, filename: str = "") -> Dict[str, Any]:
    """Enhanced image analysis with brightness, contrast, and saturation features"""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 0
        
        dominant_colors = get_dominant_colors(image)
        
        # Calculate brightness, contrast, and saturation
        pixels_array = np.array(image.convert('L'))  # Grayscale for brightness
        brightness = float(np.mean(pixels_array))
        brightness_std = float(np.std(pixels_array))
        
        # Calculate saturation from RGB
        rgb_array = np.array(image, dtype=np.float32) / 255.0
        max_val = np.max(rgb_array, axis=2)
        min_val = np.min(rgb_array, axis=2)
        saturation = np.mean(np.where(max_val > 0, (max_val - min_val) / max_val, 0))
        
        results = {
            "image_format": image.format,
            "image_size": (width, height),
            "aspect_ratio": round(aspect_ratio, 2),
            "file_size_kb": round(len(image_data) / 1024, 1),
            "dominant_colors": dominant_colors,
            "image_hash": hashlib.md5(image_data).hexdigest(),
            "resolution": f"{width}x{height}",
            "is_landscape": width > height,
            "is_portrait": height > width,
            "is_square": abs(width - height) < 10,
            "pixel_count": width * height,
            "brightness": round(brightness, 2),
            "contrast": round(brightness_std, 2),
            "saturation": round(float(saturation), 2)
        }
        
        # Determine image type
        if image.format in ['PDF', 'TIFF']:
            results["likely_type"] = "document"
        elif width <= 128 and height <= 128:
            results["likely_type"] = "icon"
        elif width <= 400 and height <= 400:
            results["likely_type"] = "thumbnail"
        elif "screenshot" in filename.lower() or width == 1920 or width == 1366 or width == 1024:
            results["likely_type"] = "screenshot"
        elif width > 2000 or height > 2000:
            results["likely_type"] = "high_res_photo"
        else:
            results["likely_type"] = "standard_image"
        
        # Add brightness category
        if brightness < 100:
            results["brightness_category"] = "dark"
        elif brightness < 155:
            results["brightness_category"] = "dim"
        else:
            results["brightness_category"] = "bright"
        
        return results
        
    except Exception as e:
        return {
            "error": f"Image analysis failed: {str(e)}",
            "image_hash": hashlib.md5(image_data).hexdigest() if image_data else ""
        }

def score_image_for_domain(image_analysis: Dict[str, Any], domain: str, filename: str = "") -> Dict[str, Any]:
    """Score how well an image matches a specific domain with improved heuristics"""
    if domain not in DOMAIN_IMAGE_PROFILES:
        return {
            "score": 0,
            "confidence": 0,
            "reasons": ["Domain not recognized"],
            "matches": []
        }
    
    domain_profile = DOMAIN_IMAGE_PROFILES[domain]
    score = 0.0
    max_score = 12.0
    reasons = []
    matches = []
    
    # 1. Check filename keywords (30% weight)
    filename_lower = filename.lower()
    keyword_matches = []
    for keyword in domain_profile['keywords']:
        if keyword in filename_lower:
            keyword_matches.append(keyword)
            score += 0.5
    
    if keyword_matches:
        matches.extend([f"filename: {kw}" for kw in keyword_matches])
        reasons.append(f"Filename contains {len(keyword_matches)} domain keywords")
    
    # 2. Check aspect ratio (15% weight)
    aspect_ratio = image_analysis.get('aspect_ratio', 0)
    if aspect_ratio:
        closest_ratio = min(domain_profile['common_aspect_ratios'], 
                           key=lambda x: abs(x - aspect_ratio))
        ratio_diff = abs(closest_ratio - aspect_ratio)
        if ratio_diff < 0.15:
            score += 1.8
            matches.append(f"aspect_ratio: {aspect_ratio:.2f}")
            reasons.append("Aspect ratio matches domain profile")
        elif ratio_diff < 0.3:
            score += 0.9
    
    # 3. Check dominant colors (25% weight)
    dominant_colors = image_analysis.get('dominant_colors', [])
    if dominant_colors:
        color_matches = set(dominant_colors) & set(domain_profile['common_colors'])
        if color_matches:
            color_score = len(color_matches) * 0.5
            score += min(color_score, 3.0)
            matches.extend([f"color: {color}" for color in color_matches])
            reasons.append(f"Colors match domain ({len(color_matches)} matches)")
    
    # 4. Check image size (15% weight)
    image_size = image_analysis.get('image_size', (0, 0))
    if image_size != (0, 0):
        min_size = domain_profile.get('min_size', (100, 100))
        max_size = domain_profile.get('max_size', (5000, 5000))
        
        if (image_size[0] >= min_size[0] and image_size[1] >= min_size[1] and
            image_size[0] <= max_size[0] and image_size[1] <= max_size[1]):
            score += 1.8
            matches.append(f"size: {image_size[0]}x{image_size[1]}")
            reasons.append("Image size within domain range")
    
    # 5. Check brightness (15% weight for certain domains)
    brightness = image_analysis.get('brightness', 127)
    brightness_cat = image_analysis.get('brightness_category', 'dim')
    
    # Educational and tech content tends to be brighter (good lighting, whiteboards, screens)
    if domain in ['education', 'tech', 'health']:
        if brightness > 130:  # Bright image
            score += 1.5
            reasons.append("Image brightness matches domain (well-lit)")
    
    # Sports and outdoor content can vary
    elif domain == 'sports':
        if brightness > 100:
            score += 0.8
    
    # 6. Saturation bonus (minor weight)
    saturation = image_analysis.get('saturation', 0.5)
    if domain in ['food', 'entertainment']:
        if saturation > 0.4:
            score += 0.8
            reasons.append("Image saturation suggests vibrant content (typical for domain)")
    
    confidence = min(score / max_score, 1.0)
    
    return {
        "score": round(score, 2),
        "confidence": round(confidence, 2),
        "reasons": reasons,
        "matches": matches,
        "max_score": max_score
    }

def detect_best_domain_for_image(image_analysis: Dict[str, Any], filename: str = "", 
                                allowed_domains: List[str] = None) -> Dict[str, Any]:
    """Find the best matching domain for an image with improved detection"""
    if 'error' in image_analysis:
        return {
            "category": "unknown",
            "confidence": 0.1,
            "is_restricted": False,
            "source": "analysis_error"
        }
    
    domains_to_check = allowed_domains if allowed_domains else list(DOMAIN_IMAGE_PROFILES.keys())
    
    best_score = 0
    best_domain = "unknown"
    best_analysis = None
    
    for domain in domains_to_check:
        if domain in DOMAIN_IMAGE_PROFILES:
            domain_score = score_image_for_domain(image_analysis, domain, filename)
            if domain_score['confidence'] > best_score:
                best_score = domain_score['confidence']
                best_domain = domain
                best_analysis = domain_score
    
    # Lowered threshold from 0.3 to 0.2 for better detection
    if best_score > 0.2:
        return {
            "category": best_domain,
            "confidence": best_score,
            "is_restricted": False,
            "source": "image_analysis",
            "domain_score": best_analysis
        }
    
    # Fallback: Use image type heuristics
    likely_type = image_analysis.get('likely_type', '')
    brightness_cat = image_analysis.get('brightness_category', '')
    is_landscape = image_analysis.get('is_landscape', False)
    is_portrait = image_analysis.get('is_portrait', False)
    
    # Documents and screenshots are typically educational
    if likely_type in ['document', 'screenshot']:
        return {
            "category": "education",
            "confidence": 0.55,
            "is_restricted": False,
            "source": "image_type_heuristic",
            "reasons": [f"Image appears to be a {likely_type}, typical of educational content"]
        }
    
    # High-res photos are typically entertainment or food
    if likely_type == 'high_res_photo':
        return {
            "category": "food" if best_score == 0 else best_domain,
            "confidence": 0.5,
            "is_restricted": False,
            "source": "image_type_heuristic",
            "reasons": ["High-resolution photo suggests professional content"]
        }
    
    # Standard images with bright conditions - likely education/tech
    if brightness_cat == 'bright' and likely_type == 'standard_image':
        return {
            "category": "education",
            "confidence": 0.45,
            "is_restricted": False,
            "source": "brightness_heuristic",
            "reasons": ["Well-lit image suggests educational or professional content"]
        }
    
    return {
        "category": "unknown",
        "confidence": 0.25,
        "is_restricted": False,
        "source": "no_match",
        "reasons": ["No strong domain match found, using fallback detection"]
    }

# ========== DECISION MAKING ==========
def make_decision(prediction, business_id=None):
    """Apply business logic rules"""
    category = prediction['category']
    confidence = prediction['confidence']
    is_restricted = prediction['is_restricted']
    
    # ALWAYS BLOCK RESTRICTED CONTENT
    if is_restricted:
        return {
            "is_allowed": False,
            "decision": "blocked",
            "reason": f"Content contains restricted material: {category}",
            "severity": "high",
            "requires_review": False
        }
    
    # Check if business ID is valid
    if business_id and not is_valid_business_id(business_id):
        return {
            "is_allowed": False,
            "decision": "blocked",
            "reason": f"Invalid business ID: '{business_id}'. Business is not registered.",
            "severity": "high",
            "requires_review": False
        }
    
    # Get business domain for comparison
    business_domain = None
    allowed_domains = []
    if business_id:
        business_domain = get_business_domain(business_id)
        allowed_domains = get_allowed_domains(business_id)
    
    # For marketplace (MARKET001), be more lenient
    if business_id == "MARKET001":
        if confidence >= 0.4:
            return {
                "is_allowed": True,
                "decision": "approved",
                "reason": f"Content approved for marketplace (confidence: {confidence:.2f})",
                "severity": "low",
                "requires_review": False
            }
        else:
            return {
                "is_allowed": False,
                "decision": "needs_review",
                "reason": f"Low confidence for marketplace (confidence: {confidence:.2f})",
                "severity": "medium",
                "requires_review": True
            }
    
    # For businesses with specific domains
    if business_domain:
        # Check if content matches business domain or allowed domains
        if category == business_domain or category in allowed_domains:
            if confidence >= 0.6:
                return {
                    "is_allowed": True,
                    "decision": "approved",
                    "reason": f"Content matches business domain '{business_domain}' (confidence: {confidence:.2f})",
                    "severity": "low",
                    "requires_review": False
                }
            elif confidence >= 0.4:
                return {
                    "is_allowed": True,
                    "decision": "approved",
                    "reason": f"Content matches business domain with moderate confidence (confidence: {confidence:.2f})",
                    "severity": "low",
                    "requires_review": False
                }
            else:
                return {
                    "is_allowed": False,
                    "decision": "needs_review",
                    "reason": f"Domain matches but confidence is low (confidence: {confidence:.2f})",
                    "severity": "medium",
                    "requires_review": True
                }
        else:
            # Content doesn't match business domain
            if confidence >= 0.7:
                # High confidence but wrong domain
                return {
                    "is_allowed": False,
                    "decision": "blocked",
                    "reason": f"Content category '{category}' does not match business domain '{business_domain}'",
                    "severity": "high",
                    "requires_review": False
                }
            else:
                # Low confidence and wrong domain
                return {
                    "is_allowed": False,
                    "decision": "blocked",
                    "reason": f"Content does not match business domain '{business_domain}' (detected: {category})",
                    "severity": "high",
                    "requires_review": False
                }
    
    # For unknown business or no business ID
    if confidence >= 0.7:
        return {
            "is_allowed": True,
            "decision": "approved",
            "reason": f"Content verified with high confidence ({confidence:.2f})",
            "severity": "low",
            "requires_review": False
        }
    elif confidence >= 0.5:
        return {
            "is_allowed": False,
            "decision": "needs_review",
            "reason": f"Medium confidence ({confidence:.2f}) - requires review",
            "severity": "medium",
            "requires_review": True
        }
    else:
        return {
            "is_allowed": False,
            "decision": "blocked",
            "reason": f"Low confidence and no business domain match ({confidence:.2f})",
            "severity": "high",
            "requires_review": False
        }

# ========== DATABASE FUNCTIONS ==========
def get_database():
    """Get database connection"""
    try:
        conn = sqlite3.connect('content_verification.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"Database error: {e}")
        return None

def save_verification(verification_data):
    """Save verification to database"""
    try:
        conn = get_database()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verification_id TEXT UNIQUE,
                business_id TEXT,
                content_type TEXT,
                title TEXT,
                description TEXT,
                image_path TEXT,
                predicted_category TEXT,
                confidence_score REAL,
                is_restricted BOOLEAN,
                ml_model_used TEXT,
                prediction_details TEXT,
                is_allowed BOOLEAN,
                decision TEXT,
                decision_reason TEXT,
                severity TEXT,
                requires_human_review BOOLEAN,
                expected_domain TEXT,
                domain_match BOOLEAN,
                domain_verification_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                verification_time_ms INTEGER,
                content_length INTEGER,
                image_hash TEXT
            )
        ''')
        
        cursor.execute('''
            INSERT INTO content_verifications 
            (verification_id, business_id, content_type, title, description, image_path,
             predicted_category, confidence_score, is_restricted, ml_model_used, prediction_details,
             is_allowed, decision, decision_reason, severity, requires_human_review,
             expected_domain, domain_match, domain_verification_score, verification_time_ms, content_length,
             image_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            verification_data['verification_id'],
            verification_data['business_id'],
            verification_data['content_type'],
            verification_data.get('title'),
            verification_data.get('description'),
            verification_data.get('image_path'),
            verification_data['prediction']['category'],
            verification_data['prediction']['confidence'],
            verification_data['prediction'].get('is_restricted', False),
            verification_data['prediction'].get('model_used', 'keyword_detector'),
            json.dumps(verification_data['prediction']),
            verification_data['decision']['is_allowed'],
            verification_data['decision']['decision'],
            verification_data['decision']['reason'],
            verification_data['decision']['severity'],
            verification_data['decision']['requires_review'],
            verification_data.get('expected_domain'),
            verification_data.get('domain_match', False),
            verification_data.get('domain_verification_score', 0.0),
            verification_data.get('verification_time_ms', 0),
            verification_data.get('content_length', 0),
            verification_data.get('image_hash', '')
        ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error saving verification: {e}")
        return False

def get_recent_verifications_db(limit=10):
    """Get recent verifications from database"""
    try:
        conn = get_database()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT verification_id, business_id, content_type, title, 
                   predicted_category, confidence_score, decision, created_at
            FROM content_verifications 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        verifications = []
        for row in results:
            verifications.append({
                'verification_id': row['verification_id'],
                'business_id': row['business_id'],
                'content_type': row['content_type'],
                'title': row['title'],
                'predicted_category': row['predicted_category'],
                'confidence_score': float(row['confidence_score']),
                'decision': row['decision'],
                'created_at': row['created_at']
            })
        
        return verifications
        
    except Exception as e:
        print(f"Error getting verifications: {e}")
        return []

def get_analytics_db(business_id=None, days=30):
    """Get analytics from database"""
    try:
        conn = get_database()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        
        query = '''
            SELECT decision, COUNT(*) as count, AVG(confidence_score) as avg_confidence
            FROM content_verifications
            WHERE created_at >= datetime('now', ?)
        '''
        params = [f'-{days} days']
        
        if business_id:
            query += " AND business_id = ?"
            params.append(business_id)
        
        query += " GROUP BY decision"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        cursor.execute('''
            SELECT predicted_category, COUNT(*) as count
            FROM content_verifications
            WHERE created_at >= datetime('now', ?)
            GROUP BY predicted_category
        ''', [f'-{days} days'])
        category_results = cursor.fetchall()
        
        conn.close()
        
        total = 0
        approved = 0
        blocked = 0
        review = 0
        avg_confidence = 0
        
        for row in results:
            count = row['count']
            total += count
            
            if row['decision'] == 'approved':
                approved = count
            elif row['decision'] == 'blocked':
                blocked = count
            elif row['decision'] == 'needs_review':
                review = count
            
            if row['avg_confidence']:
                avg_confidence += row['avg_confidence'] * count
        
        avg_confidence = avg_confidence / total if total > 0 else 0
        
        category_dist = {}
        for row in category_results:
            category_dist[row['predicted_category']] = row['count']
        
        return {
            'total_verifications': total,
            'approved_count': approved,
            'blocked_count': blocked,
            'review_count': review,
            'approval_rate': (approved / total * 100) if total > 0 else 0,
            'average_confidence': float(avg_confidence) if total > 0 else 0,
            'category_distribution': category_dist
        }
        
    except Exception as e:
        print(f"Error getting analytics: {e}")
        return {}

# ========== FASTAPI APP ==========
app = FastAPI(
    title="Content Verification API",
    description="Enhanced image verification with domain matching",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Create upload directory
UPLOAD_DIR = os.path.join(project_root, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
static_dir = os.path.join(project_root, "web_app", "static")
templates_dir = os.path.join(project_root, "web_app", "templates")

# Create directories if they don't exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

# Mount static files
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates = Jinja2Templates(directory=templates_dir)

# Pydantic models
class TextContent(BaseModel):
    text: str
    title: Optional[str] = None
    business_id: Optional[str] = None

class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    category: str
    confidence: float
    is_restricted: bool = False
    model_used: Optional[str] = None
    source: Optional[str] = None
    domain_score: Optional[Dict[str, Any]] = None

class DecisionResult(BaseModel):
    is_allowed: bool
    decision: str
    reason: str
    severity: str
    requires_review: bool

class VerificationResponse(BaseModel):
    content_type: str
    prediction: PredictionResult
    decision: DecisionResult
    timestamp: str
    verification_id: Optional[str] = None
    image_analysis: Optional[Dict[str, Any]] = None

# ========== WEB PAGES ==========
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request):
    """Verification page"""
    try:
        # Create a simple verify.html template if it doesn't exist
        verify_template_path = os.path.join(templates_dir, "verify.html")
        if not os.path.exists(verify_template_path):
            # Create a simple verify.html template
            verify_html = """<!DOCTYPE html>
<html>
<head>
    <title>Verify Content</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 10px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .subtitle {
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .required::after {
            content: " *";
            color: #e74c3c;
        }
        
        select, input[type="text"], textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        select:focus, input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #3498db;
        }
        
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        
        input[type="file"] {
            padding: 10px;
            background: #f8f9fa;
            border: 2px dashed #ddd;
            width: 100%;
        }
        
        .business-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-top: 5px;
            font-size: 14px;
            color: #666;
        }
        
        .submit-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
            font-weight: 600;
            width: 100%;
        }
        
        .submit-btn:hover {
            background: #2980b9;
        }
        
        .nav-links {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }
        
        .nav-links a {
            color: #3498db;
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 3px;
            transition: background 0.3s;
        }
        
        .nav-links a:hover {
            background: #f0f0f0;
        }
        
        .policy-info {
            background: #e8f4fc;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-top: 30px;
            border-radius: 0 5px 5px 0;
        }
        
        .policy-info h3 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .policy-info ul {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        
        .policy-info li {
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Content Verification System</h1>
        <p class="subtitle">Verify text or image content against business domain policies</p>
        
        <form action="/verify" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="business_id" class="required">Business ID</label>
                <select id="business_id" name="business_id" required>
                    <option value="">Select a Business</option>
                    <option value="EDU001">EDU001 - EduTech Academy (Education)</option>
                    <option value="SPORTS001">SPORTS001 - Sports Gear Hub (Sports)</option>
                    <option value="FOOD001">FOOD001 - Foodie Delights (Food)</option>
                    <option value="TECH001">TECH001 - Tech Solutions Inc (Technology)</option>
                    <option value="MARKET001">MARKET001 - MultiShop Marketplace (All)</option>
                    <option value="none">No Business ID (Generic Check)</option>
                </select>
                <div class="business-info">
                    Select "none" for generic verification without business restrictions
                </div>
            </div>
            
            <div class="form-group">
                <label for="content_type" class="required">Content Type</label>
                <select id="content_type" name="content_type" required onchange="toggleFields()">
                    <option value="text">Text Only</option>
                    <option value="image">Image Only</option>
                    <option value="mixed">Text + Image</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="title">Title (Optional)</label>
                <input type="text" id="title" name="title" placeholder="Enter content title">
            </div>
            
            <div class="form-group" id="description_group">
                <label for="description" class="required">Description</label>
                <textarea id="description" name="description" placeholder="Enter content description"></textarea>
            </div>
            
            <div class="form-group" id="image_group" style="display: none;">
                <label for="image" class="required">Image File</label>
                <input type="file" id="image" name="image" accept="image/*,.jpg,.jpeg,.png,.gif,.webp">
                <div class="business-info">
                    Supported formats: JPG, PNG, GIF, WebP (Max 10MB)
                </div>
            </div>
            
            <button type="submit" class="submit-btn">Verify Content</button>
        </form>
        
        <div class="policy-info">
            <h3>Business Domain Policies:</h3>
            <ul>
                <li><strong>EDU001</strong>: Education content only (education, tech, sports, science, books)</li>
                <li><strong>SPORTS001</strong>: Sports content only (sports, health, fitness, outdoors, travel)</li>
                <li><strong>FOOD001</strong>: Food content only (food, restaurant, cooking, health, travel)</li>
                <li><strong>TECH001</strong>: Technology content only (tech, software, education, electronics, business)</li>
                <li><strong>MARKET001</strong>: All non-restricted content (weapons, drugs, adult content, gambling are blocked)</li>
            </ul>
            <p><strong>Restricted Categories:</strong> weapons, drugs, adult content, gambling (always blocked)</p>
        </div>
        
        <div class="nav-links">
            <a href="/">Home</a>
            <a href="/dashboard">Dashboard</a>
            <a href="/docs">API Docs</a>
        </div>
    </div>
    
    <script>
        function toggleFields() {
            const contentType = document.getElementById('content_type').value;
            const descGroup = document.getElementById('description_group');
            const imgGroup = document.getElementById('image_group');
            const descLabel = descGroup.querySelector('label');
            const imgLabel = imgGroup.querySelector('label');
            
            if (contentType === 'text') {
                descGroup.style.display = 'block';
                imgGroup.style.display = 'none';
                descLabel.classList.add('required');
                imgLabel.classList.remove('required');
                document.getElementById('description').required = true;
                document.getElementById('image').required = false;
            } else if (contentType === 'image') {
                descGroup.style.display = 'none';
                imgGroup.style.display = 'block';
                descLabel.classList.remove('required');
                imgLabel.classList.add('required');
                document.getElementById('description').required = false;
                document.getElementById('image').required = true;
            } else if (contentType === 'mixed') {
                descGroup.style.display = 'block';
                imgGroup.style.display = 'block';
                descLabel.classList.add('required');
                imgLabel.classList.remove('required');
                document.getElementById('description').required = true;
                document.getElementById('image').required = false;
            }
        }
        
        // Initialize on page load
        window.onload = toggleFields;
    </script>
</body>
</html>"""
            
            with open(verify_template_path, "w") as f:
                f.write(verify_html)
        
        return templates.TemplateResponse("verify.html", {"request": request})
    except Exception as e:
        return HTMLResponse(f"""
        <html>
        <head><title>Verify Content</title></head>
        <body>
            <h1>Content Verification System</h1>
            <p>Verify text or image content against business domain policies</p>
            <form action="/verify" method="post" enctype="multipart/form-data">
                <div>
                    <label for="business_id">Business ID *</label>
                    <select id="business_id" name="business_id" required>
                        <option value="">Select a Business</option>
                        <option value="EDU001">EDU001 - EduTech Academy</option>
                        <option value="SPORTS001">SPORTS001 - Sports Gear Hub</option>
                        <option value="FOOD001">FOOD001 - Foodie Delights</option>
                        <option value="TECH001">TECH001 - Tech Solutions Inc</option>
                        <option value="MARKET001">MARKET001 - MultiShop Marketplace</option>
                        <option value="none">No Business (Generic)</option>
                    </select>
                </div>
                <div>
                    <label for="content_type">Content Type *</label>
                    <select id="content_type" name="content_type" required>
                        <option value="text">Text Only</option>
                        <option value="image">Image Only</option>
                        <option value="mixed">Text + Image</option>
                    </select>
                </div>
                <div>
                    <label for="title">Title (Optional)</label>
                    <input type="text" id="title" name="title">
                </div>
                <div>
                    <label for="description">Description</label>
                    <textarea id="description" name="description" rows="4"></textarea>
                </div>
                <div>
                    <label for="image">Image File</label>
                    <input type="file" id="image" name="image">
                </div>
                <button type="submit">Verify Content</button>
            </form>
            <p><a href="/">Home</a></p>
        </body>
        </html>
        """)

@app.post("/verify", response_class=HTMLResponse)
async def verify_content_form(
    request: Request,
    business_id: str = Form(...),
    content_type: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """Handle form submission from web interface"""
    try:
        verification_id = f"WEB_{uuid.uuid4().hex[:8].upper()}"
        
        # Validate business ID
        if business_id and business_id != "none" and not is_valid_business_id(business_id):
            error_msg = f"Business ID '{business_id}' is not registered. Registered IDs: EDU001, SPORTS001, FOOD001, TECH001, MARKET001"
            if templates:
                return templates.TemplateResponse("error.html", {
                    "request": request,
                    "error_message": error_msg
                }, status_code=400)
            return HTMLResponse(f"<h1>Error</h1><p>{error_msg}</p><a href='/verify'>Go back</a>", status_code=400)
        
        business_for_prediction = business_id if business_id != "none" else None
        
        if content_type == "text":
            if not description:
                error_msg = "Description is required for text content verification"
                if templates:
                    return templates.TemplateResponse("error.html", {
                        "request": request,
                        "error_message": error_msg
                    }, status_code=400)
                return HTMLResponse(f"<h1>Error</h1><p>{error_msg}</p><a href='/verify'>Go back</a>", status_code=400)
            
            detection = detect_content_type(description, business_for_prediction)
            
            prediction = {
                "category": detection['category'],
                "confidence": detection['confidence'],
                "is_restricted": detection['is_restricted'],
                "model_used": "keyword_detector",
                "source": "text_analysis"
            }
            
            decision = make_decision(prediction, business_for_prediction)
            
            verification_data = {
                'verification_id': verification_id,
                'business_id': business_id if business_id != "none" else "unknown",
                'content_type': 'text',
                'title': title,
                'description': description,
                'image_path': None,
                'prediction': prediction,
                'decision': decision,
                'verification_time_ms': 100,
                'content_length': len(description)
            }
            save_verification(verification_data)
            
            result = {
                'content_type': 'text',
                'prediction': prediction,
                'decision': decision,
                'timestamp': datetime.now().isoformat(),
                'verification_id': verification_id
            }
            
            # Create results template if it doesn't exist
            results_template_path = os.path.join(templates_dir, "results.html")
            if not os.path.exists(results_template_path):
                results_html = """<!DOCTYPE html>
<html>
<head>
    <title>Verification Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .result-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .approved { border-left: 5px solid #28a745; }
        .blocked { border-left: 5px solid #dc3545; }
        .review { border-left: 5px solid #ffc107; }
        .verification-id { font-size: 14px; color: #666; background: #f8f9fa; padding: 5px 10px; border-radius: 4px; }
        .details { margin-top: 20px; }
        .details h3 { margin-top: 0; }
        .details table { width: 100%; border-collapse: collapse; }
        .details td, .details th { padding: 8px; border-bottom: 1px solid #ddd; text-align: left; }
        .nav-links { margin-top: 20px; }
        .nav-links a { color: #007bff; text-decoration: none; margin-right: 15px; }
        .nav-links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Content Verification Results</h1>
        <div class="verification-id">Verification ID: {{ verification_id }}</div>
        
        <div class="result-card {% if result.decision.decision == 'approved' %}approved{% elif result.decision.decision == 'blocked' %}blocked{% else %}review{% endif %}">
            <h2>Decision: {{ result.decision.decision|upper }}</h2>
            <p><strong>Content Type:</strong> {{ result.content_type }}</p>
            <p><strong>Category:</strong> {{ result.prediction.category }}</p>
            <p><strong>Confidence:</strong> {{ (result.prediction.confidence * 100)|round(1) }}%</p>
            <p><strong>Reason:</strong> {{ result.decision.reason }}</p>
            <p><strong>Severity:</strong> {{ result.decision.severity }}</p>
            <p><strong>Requires Review:</strong> {{ result.decision.requires_review }}</p>
        </div>
        
        <div class="details">
            <h3>Detailed Analysis</h3>
            <table>
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Verification ID</td><td>{{ result.verification_id }}</td></tr>
                <tr><td>Timestamp</td><td>{{ result.timestamp }}</td></tr>
                <tr><td>Content Type</td><td>{{ result.content_type }}</td></tr>
                <tr><td>Predicted Category</td><td>{{ result.prediction.category }}</td></tr>
                <tr><td>Confidence Score</td><td>{{ (result.prediction.confidence * 100)|round(1) }}%</td></tr>
                <tr><td>Model Used</td><td>{{ result.prediction.model_used }}</td></tr>
                <tr><td>Source</td><td>{{ result.prediction.source }}</td></tr>
                <tr><td>Restricted Content</td><td>{{ result.prediction.is_restricted }}</td></tr>
                <tr><td>Final Decision</td><td>{{ result.decision.decision }}</td></tr>
                <tr><td>Decision Reason</td><td>{{ result.decision.reason }}</td></tr>
                <tr><td>Severity Level</td><td>{{ result.decision.severity }}</td></tr>
                <tr><td>Requires Human Review</td><td>{{ result.decision.requires_review }}</td></tr>
            </table>
        </div>
        
        <div class="nav-links">
            <a href="/verify">Verify Another</a>
            <a href="/dashboard">Dashboard</a>
            <a href="/">Home</a>
        </div>
    </div>
</body>
</html>"""
                with open(results_template_path, "w") as f:
                    f.write(results_html)
            
            return templates.TemplateResponse("results.html", {
                "request": request,
                "result": result,
                "verification_id": verification_id
            })
            
        elif content_type == "image":
            if not image:
                error_html = """
                <html>
                <head><title>Error</title></head>
                <body>
                    <h1>Verification Error</h1>
                    <p>Error: Image file required for image verification</p>
                    <a href="/verify">Go back</a>
                </body>
                </html>
                """
                return HTMLResponse(error_html)
            
            image_data = await image.read()
            
            # Use universal verification for images
            verification_result = verify_image_universal(
                image_data,
                business_for_prediction,
                image.filename
            )
            
            file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
            filename = f"{verification_id}.{file_extension}"
            image_path = os.path.join(UPLOAD_DIR, filename)
            
            with open(image_path, "wb") as f:
                f.write(image_data)
            
            # Save verification data
            verification_data = {
                'verification_id': verification_id,
                'business_id': business_id if business_id != "none" else "unknown",
                'content_type': 'image',
                'title': title,
                'description': description or '',
                'image_path': image_path,
                'prediction': {
                    'category': verification_result['detected_domain'],
                    'confidence': verification_result['confidence'],
                    'is_restricted': verification_result.get('is_restricted', False),
                    'model_used': verification_result['verification_method']
                },
                'decision': {
                    'is_allowed': verification_result['is_allowed'],
                    'decision': verification_result['decision'],
                    'reason': verification_result['reason'],
                    'severity': verification_result['severity'],
                    'requires_review': verification_result['requires_review']
                },
                'verification_time_ms': 200,
                'content_length': len(description or ""),
                'image_hash': verification_result.get('image_analysis', {}).get('image_hash', '')
            }
            save_verification(verification_data)
            
            result = {
                'content_type': 'image',
                'prediction': {
                    'category': verification_result['detected_domain'],
                    'confidence': verification_result['confidence'],
                    'is_restricted': verification_result.get('is_restricted', False),
                    'model_used': verification_result['verification_method'],
                    'source': 'universal_verification'
                },
                'decision': {
                    'is_allowed': verification_result['is_allowed'],
                    'decision': verification_result['decision'],
                    'reason': verification_result['reason'],
                    'severity': verification_result['severity'],
                    'requires_review': verification_result['requires_review']
                },
                'timestamp': datetime.now().isoformat(),
                'verification_id': verification_id,
                'image_analysis': verification_result.get('image_analysis', {})
            }
            
            return templates.TemplateResponse("results.html", {
                "request": request,
                "result": result,
                "verification_id": verification_id
            })
        
        elif content_type == "mixed":
            if not description:
                error_html = """
                <html>
                <head><title>Error</title></head>
                <body>
                    <h1>Verification Error</h1>
                    <p>Error: Description required for mixed content</p>
                    <a href="/verify">Go back</a>
                </body>
                </html>
                """
                return HTMLResponse(error_html)
            
            image_analysis = {}
            image_path = None
            image_hash = ""
            
            if image:
                image_data = await image.read()
                
                file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
                filename = f"{verification_id}.{file_extension}"
                image_path = os.path.join(UPLOAD_DIR, filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_data)
                
                image_analysis = analyze_image_enhanced(image_data, image.filename)
                image_hash = image_analysis.get('image_hash', '')
            
            text_detection = detect_content_type(description, business_for_prediction)
            
            if image and 'error' not in image_analysis:
                allowed_domains = []
                if business_for_prediction:
                    allowed_domains = get_allowed_domains(business_for_prediction)
                
                image_detection = detect_best_domain_for_image(
                    image_analysis,
                    image.filename,
                    allowed_domains
                )
                
                if text_detection['category'] == image_detection['category']:
                    combined_confidence = (text_detection['confidence'] + image_detection['confidence']) / 2
                    combined_confidence = min(combined_confidence * 1.1, 0.95)
                    final_category = text_detection['category']
                else:
                    if text_detection['confidence'] >= image_detection['confidence']:
                        final_category = text_detection['category']
                        combined_confidence = text_detection['confidence']
                    else:
                        final_category = image_detection['category']
                        combined_confidence = image_detection['confidence']
                
                is_restricted = text_detection['is_restricted'] or image_detection['is_restricted']
            else:
                final_category = text_detection['category']
                combined_confidence = text_detection['confidence']
                is_restricted = text_detection['is_restricted']
            
            prediction = {
                "category": final_category,
                "confidence": combined_confidence,
                "is_restricted": is_restricted,
                "model_used": "multimodal" if image else "keyword_detector",
                "source": "combined_analysis" if image else "text_analysis"
            }
            
            decision = make_decision(prediction, business_for_prediction)
            
            verification_data = {
                'verification_id': verification_id,
                'business_id': business_id if business_id != "none" else "unknown",
                'content_type': 'mixed',
                'title': title,
                'description': description,
                'image_path': image_path,
                'prediction': prediction,
                'decision': decision,
                'verification_time_ms': 300,
                'content_length': len(description),
                'image_hash': image_hash
            }
            save_verification(verification_data)
            
            result = {
                'content_type': 'mixed',
                'prediction': prediction,
                'decision': decision,
                'timestamp': datetime.now().isoformat(),
                'verification_id': verification_id
            }
            
            return templates.TemplateResponse("results.html", {
                "request": request,
                "result": result,
                "verification_id": verification_id
            })
        
        # Default fallback
        return HTMLResponse(f"""
        <html>
        <head><title>Verification Complete</title></head>
        <body>
            <h1>Verification Complete</h1>
            <p>Verification ID: {verification_id}</p>
            <p>Business ID: {business_id}</p>
            <p>Content Type: {content_type}</p>
            <p><a href="/verify">Verify Another</a></p>
        </body>
        </html>
        """)
        
    except Exception as e:
        error_html = f"""
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Verification Error</h1>
            <p>{str(e)}</p>
            <a href="/verify">Go back</a>
        </body>
        </html>
        """
        return HTMLResponse(error_html, status_code=500)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard page"""
    analytics = get_analytics_db(days=7)
    recent = get_recent_verifications_db(limit=10)
    
    # Create dashboard template if it doesn't exist
    dashboard_template_path = os.path.join(templates_dir, "dashboard.html")
    if not os.path.exists(dashboard_template_path):
        dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <title>Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .stat-card h3 { margin-top: 0; color: #666; }
        .stat-value { font-size: 32px; font-weight: bold; margin: 10px 0; }
        .stat-value.approved { color: #28a745; }
        .stat-value.blocked { color: #dc3545; }
        .stat-value.review { color: #ffc107; }
        .recent-table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .recent-table th, .recent-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .recent-table th { background: #f8f9fa; font-weight: 600; }
        .recent-table tr:hover { background: #f5f5f5; }
        .status-badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .status-approved { background: #d4edda; color: #155724; }
        .status-blocked { background: #f8d7da; color: #721c24; }
        .status-review { background: #fff3cd; color: #856404; }
        .nav-links { margin-top: 20px; }
        .nav-links a { color: #007bff; text-decoration: none; margin-right: 15px; }
        .nav-links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Content Verification Dashboard</h1>
        <p>Overview of content verification activity</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Verifications</h3>
                <div class="stat-value">{{ analytics.total_verifications }}</div>
                <p>Last 7 days</p>
            </div>
            <div class="stat-card">
                <h3>Approved</h3>
                <div class="stat-value approved">{{ analytics.approved_count }}</div>
                <p>{{ analytics.approval_rate|round(1) }}% approval rate</p>
            </div>
            <div class="stat-card">
                <h3>Blocked</h3>
                <div class="stat-value blocked">{{ analytics.blocked_count }}</div>
                <p>Restricted content</p>
            </div>
            <div class="stat-card">
                <h3>Needs Review</h3>
                <div class="stat-value review">{{ analytics.review_count }}</div>
                <p>Manual review required</p>
            </div>
        </div>
        
        <h2>Recent Verifications</h2>
        <table class="recent-table">
            <thead>
                <tr>
                    <th>Verification ID</th>
                    <th>Business ID</th>
                    <th>Content Type</th>
                    <th>Category</th>
                    <th>Confidence</th>
                    <th>Decision</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
                {% for item in recent_verifications %}
                <tr>
                    <td>{{ item.verification_id }}</td>
                    <td>{{ item.business_id }}</td>
                    <td>{{ item.content_type }}</td>
                    <td>{{ item.predicted_category }}</td>
                    <td>{{ (item.confidence_score * 100)|round(1) }}%</td>
                    <td>
                        <span class="status-badge status-{{ item.decision }}">
                            {{ item.decision|upper }}
                        </span>
                    </td>
                    <td>{{ item.created_at }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <div class="nav-links">
            <a href="/verify">Verify Content</a>
            <a href="/analytics">Analytics</a>
            <a href="/">Home</a>
        </div>
    </div>
</body>
</html>"""
        with open(dashboard_template_path, "w") as f:
            f.write(dashboard_html)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "analytics": analytics,
        "recent_verifications": recent
    })

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics page"""
    analytics = get_analytics_db(days=30)
    
    # Create analytics template if it doesn't exist
    analytics_template_path = os.path.join(templates_dir, "analytics.html")
    if not os.path.exists(analytics_template_path):
        analytics_html = """<!DOCTYPE html>
<html>
<head>
    <title>Analytics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .summary-card h3 { margin-top: 0; color: #666; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-name { color: #666; }
        .metric-value { font-weight: bold; }
        .category-list { list-style: none; padding: 0; }
        .category-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
        .category-name { flex: 1; }
        .category-count { color: #666; }
        .nav-links { margin-top: 20px; }
        .nav-links a { color: #007bff; text-decoration: none; margin-right: 15px; }
        .nav-links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Content Verification Analytics</h1>
        <p>Detailed statistics for the last 30 days</p>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Verification Summary</h3>
                <div class="metric">
                    <span class="metric-name">Total Verifications</span>
                    <span class="metric-value">{{ analytics.total_verifications }}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Approved</span>
                    <span class="metric-value">{{ analytics.approved_count }}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Blocked</span>
                    <span class="metric-value">{{ analytics.blocked_count }}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Needs Review</span>
                    <span class="metric-value">{{ analytics.review_count }}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Approval Rate</span>
                    <span class="metric-value">{{ analytics.approval_rate|round(1) }}%</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Avg Confidence</span>
                    <span class="metric-value">{{ (analytics.average_confidence * 100)|round(1) }}%</span>
                </div>
            </div>
            
            <div class="summary-card">
                <h3>Category Distribution</h3>
                <ul class="category-list">
                    {% for category, count in analytics.category_distribution.items() %}
                    <li class="category-item">
                        <span class="category-name">{{ category }}</span>
                        <span class="category-count">{{ count }}</span>
                    </li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        <div class="nav-links">
            <a href="/dashboard">Dashboard</a>
            <a href="/verify">Verify Content</a>
            <a href="/">Home</a>
        </div>
    </div>
</body>
</html>"""
        with open(analytics_template_path, "w") as f:
            f.write(analytics_html)
    
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "analytics": analytics
    })

# ========== API ENDPOINTS ==========
@app.post("/api/verify/image", response_model=VerificationResponse)
async def verify_image_api(
    image: UploadFile = File(...),
    business_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """Universal image verification endpoint"""
    try:
        if business_id and business_id != "none" and not is_valid_business_id(business_id):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid business ID",
                    "message": f"Business ID '{business_id}' is not registered.",
                    "registered_businesses": ["EDU001", "SPORTS001", "FOOD001", "TECH001", "MARKET001"]
                }
            )
        
        start_time = datetime.now()
        
        image_data = await image.read()
        
        # Use universal verification
        verification_result = verify_image_universal(
            image_data, 
            business_id if business_id != "none" else None,
            image.filename
        )
        
        # Generate verification ID
        verification_id = f"IMG_{uuid.uuid4().hex[:8].upper()}"
        
        # Prepare response
        response_data = {
            "content_type": "image",
            "prediction": {
                "category": verification_result['detected_domain'],
                "confidence": verification_result['confidence'],
                "is_restricted": verification_result.get('is_restricted', False),
                "model_used": verification_result['verification_method'],
                "source": "universal_verification"
            },
            "decision": {
                "is_allowed": verification_result['is_allowed'],
                "decision": verification_result['decision'],
                "reason": verification_result['reason'],
                "severity": verification_result['severity'],
                "requires_review": verification_result['requires_review']
            },
            "timestamp": datetime.now().isoformat(),
            "verification_id": verification_id,
            "image_analysis": verification_result.get('image_analysis', {})
        }
        
        # Save to database
        verification_data = {
            'verification_id': verification_id,
            'business_id': business_id or 'unknown',
            'content_type': 'image',
            'title': title,
            'description': description or '',
            'image_path': f"uploads/{verification_id}.jpg",
            'predicted_category': verification_result['detected_domain'],
            'confidence_score': verification_result['confidence'],
            'is_restricted': verification_result.get('is_restricted', False),
            'is_allowed': verification_result['is_allowed'],
            'decision': verification_result['decision'],
            'decision_reason': verification_result['reason'],
            'severity': verification_result['severity'],
            'requires_human_review': verification_result['requires_review']
        }
        save_verification(verification_data)
        
        return VerificationResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image verification error: {str(e)}")

@app.post("/api/verify/text", response_model=VerificationResponse)
async def verify_text_api(content: TextContent):
    """API endpoint for text verification"""
    try:
        if content.business_id and not is_valid_business_id(content.business_id):
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Invalid business ID",
                    "message": f"Business ID '{content.business_id}' is not registered in the system.",
                    "registered_businesses": ["EDU001", "SPORTS001", "FOOD001", "TECH001", "MARKET001"]
                }
            )
        
        start_time = datetime.now()
        
        detection = detect_content_type(content.text, content.business_id)
        
        prediction = {
            "category": detection['category'],
            "confidence": detection['confidence'],
            "is_restricted": detection['is_restricted'],
            "model_used": "keyword_detector",
            "source": "text_analysis"
        }
        
        decision = make_decision(prediction, content.business_id)
        
        verification_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        verification_id = f"API_{uuid.uuid4().hex[:8].upper()}"
        verification_data = {
            'verification_id': verification_id,
            'business_id': content.business_id or 'unknown',
            'content_type': 'text',
            'title': content.title,
            'description': content.text,
            'image_path': None,
            'prediction': prediction,
            'decision': decision,
            'verification_time_ms': verification_time_ms,
            'content_length': len(content.text)
        }
        save_verification(verification_data)
        
        response = VerificationResponse(
            content_type="text",
            prediction=PredictionResult(**prediction),
            decision=DecisionResult(**decision),
            timestamp=datetime.now().isoformat(),
            verification_id=verification_id
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/business/validate/{business_id}")
async def validate_business_api(business_id: str):
    """Validate if a business ID is registered"""
    try:
        is_valid = is_valid_business_id(business_id)
        profile = db_manager.get_business_profile(business_id) if is_valid else None
        
        return {
            "business_id": business_id,
            "is_valid": is_valid,
            "business_profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recent-verifications")
async def get_recent_verifications(limit: int = 10):
    """Get recent verifications"""
    try:
        conn = get_database()
        if not conn:
            return {"verifications": []}
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM content_verifications 
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                'verification_id': row['verification_id'],
                'business_id': row['business_id'],
                'predicted_category': row['predicted_category'],
                'confidence': float(row['confidence_score']),
                'decision': row['decision'],
                'decision_reason': row['decision_reason'],
                'created_at': row['created_at']
            })
        
        return {"verifications": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def get_analytics(business_id: Optional[str] = None):
    """Get analytics data"""
    try:
        conn = get_database()
        if not conn:
            return {
                "total_verifications": 0,
                "approved_count": 0,
                "blocked_count": 0,
                "review_count": 0,
                "approval_rate": 0.0
            }
        
        cursor = conn.cursor()
        
        if business_id:
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN decision='approved' THEN 1 ELSE 0 END) as approved,
                       SUM(CASE WHEN decision='blocked' THEN 1 ELSE 0 END) as blocked,
                       SUM(CASE WHEN decision='needs_review' THEN 1 ELSE 0 END) as review
                FROM content_verifications
                WHERE business_id = ?
            ''', (business_id,))
        else:
            cursor.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN decision='approved' THEN 1 ELSE 0 END) as approved,
                       SUM(CASE WHEN decision='blocked' THEN 1 ELSE 0 END) as blocked,
                       SUM(CASE WHEN decision='needs_review' THEN 1 ELSE 0 END) as review
                FROM content_verifications
            ''')
        
        row = cursor.fetchone()
        conn.close()
        
        total = row['total'] or 0
        approved = row['approved'] or 0
        blocked = row['blocked'] or 0
        review = row['review'] or 0
        
        approval_rate = (approved / total * 100) if total > 0 else 0.0
        
        return {
            "total_verifications": total,
            "approved_count": approved,
            "blocked_count": blocked,
            "review_count": review,
            "approval_rate": round(approval_rate, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_database()
        db_status = "connected" if conn else "disconnected"
        if conn:
            conn.close()
        
        classifier_status = "available" if UNIVERSAL_CLASSIFIER_AVAILABLE else "unavailable"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": db_status,
                "universal_classifier": classifier_status,
                "api": "running",
                "web_interface": "enabled",
                "image_analysis": "enhanced"
            },
            "registered_businesses": ["EDU001", "SPORTS001", "FOOD001", "TECH001", "MARKET001"]
        }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting Enhanced Content Verification API...")
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("📈 Analytics: http://localhost:8000/analytics")
    print("\n🔍 Enhanced Image Analysis Features:")
    print("   - Universal Image Classifier for ANY domain")
    print("   - Domain-specific color matching")
    print("   - Aspect ratio analysis")
    print("   - Filename keyword detection")
    print("   - Image type classification")
    print("\n🏢 Business Domain Enforcement:")
    print("   - EDU001 (Education): Only allows education, tech, sports, science, books")
    print("   - SPORTS001 (Sports): Only allows sports, health, fitness, outdoors, travel")
    print("   - FOOD001 (Food): Only allows food, restaurant, cooking, health, travel")
    print("   - TECH001 (Tech): Only allows tech, software, education, electronics, business")
    print("   - MARKET001 (Marketplace): Allows all non-restricted content")
    
    # Pass application as import string to enable `reload=True` without warnings.
    # Use the package-style import path so uvicorn can reload the module correctly.
    uvicorn.run("api.endpoints:app", host="0.0.0.0", port=8000, reload=True)