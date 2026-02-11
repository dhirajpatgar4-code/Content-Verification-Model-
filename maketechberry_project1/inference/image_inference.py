import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pickle
import os
import sys
import numpy as np
from typing import Dict, Optional

sys.path.append('..')

class ImageClassifier(nn.Module):
    """CNN-based image classifier"""
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ImageInference:
    """Complete image inference engine"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.categories = []
        self.idx_to_category = {}
        self.loaded = False
        # Keyword mappings for filename/semantic detection
        self.category_keywords = {
            "food": ["food", "meal", "recipe", "dish", "cuisine", "cooking", "restaurant", "ingredient", "cake", "pizza"],
            "tech": ["tech", "phone", "laptop", "code", "software", "computer", "app", "gadget", "circuit", "digital"],
            "education": ["school", "education", "classroom", "student", "teacher", "study", "course", "university", "learning", "lecture"],
            "health": ["health", "doctor", "hospital", "medicine", "fitness", "gym", "exercise", "wellness", "care", "medical"],
            "finance": ["finance", "money", "bank", "investment", "stock", "graph", "chart", "business", "income", "wealth"],
            "fashion": ["fashion", "dress", "clothing", "style", "designer", "outfit", "wear", "apparel", "shoe", "accessory"],
            "electronics": ["electronic", "phone", "screen", "monitor", "gadget", "device", "charger", "cable", "camera", "headphone"],
            "automotive": ["car", "vehicle", "automotive", "truck", "bike", "motorcycle", "engine", "mechanic", "road", "traffic"],
            "real_estate": ["house", "property", "building", "apartment", "real estate", "home", "land", "construction", "estate", "residential"],
            "entertainment": ["movie", "film", "music", "game", "entertainment", "show", "actor", "concert", "video", "performance"],
            "travel": ["travel", "flight", "hotel", "airport", "destination", "trip", "tour", "tourism", "vacation", "luggage"],
            "beauty": ["beauty", "cosmetic", "makeup", "skincare", "hair", "salon", "spa", "nail", "fragrance", "treatment"],
            "home": ["home", "furniture", "interior", "decor", "kitchen", "living room", "bedroom", "design", "appliance", "household"],
            "sports": ["sport", "game", "player", "team", "match", "competition", "athletic", "training", "championship", "ball"],
            "weapons": ["weapon", "gun", "bomb", "explosive", "firearm", "ammunition", "knife", "violence", "armed", "combat"],
            "drugs": ["drug", "narcotic", "cocaine", "heroin", "substance", "illegal", "abuse", "addiction", "dealer", "narcotic"],
            "adult_content": ["adult", "nude", "xxx", "pornography", "explicit", "sexual", "mature", "erotic", "18+", "restricted"],
            "gambling": ["gamble", "casino", "poker", "betting", "slot", "lottery", "roulette", "wager", "jackpot", "chips"],
        }
    
    def load_model(self, model_path="ml_models/image_model/"):
        """Load trained image model"""
        try:
            # Load category mappings
            with open(os.path.join(model_path, "categories.pkl"), "rb") as f:
                mappings = pickle.load(f)
                self.categories = mappings['categories']
                self.idx_to_category = mappings['idx_to_category']
            
            # Initialize model
            self.model = ImageClassifier(len(self.categories)).to(self.device)
            
            # Load weights
            model_file = os.path.join(model_path, "model.pth")
            if os.path.exists(model_file):
                self.model.load_state_dict(
                    torch.load(model_file, map_location=self.device)
                )
            else:
                print("⚠️ Image model weights not found. Using random weights.")
            
            self.model.eval()
            
            # Load transform
            transform_file = os.path.join(model_path, "transform.pkl")
            if os.path.exists(transform_file):
                with open(transform_file, "rb") as f:
                    self.transform = pickle.load(f)
            else:
                # Default transform
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            
            self.loaded = True
            print("✅ Image model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading image model: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize with fallback values for testing"""
        self.categories = [
            "food", "tech", "education", "health", "finance", "fashion",
            "electronics", "automotive", "real_estate", "entertainment",
            "travel", "beauty", "home", "sports",
            "weapons", "drugs", "adult_content", "gambling", "unknown"
        ]
        self.idx_to_category = {i: cat for i, cat in enumerate(self.categories)}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.model = ImageClassifier(len(self.categories)).to(self.device)
        self.model.eval()
        self.loaded = True
        print("⚠️ Using fallback image model for testing")
    
    def predict(self, image_path: str) -> Dict:
        """Predict category for image"""
        if not self.loaded:
            self.load_model()
        
        try:
            # If model weights don't exist, use semantic prediction
            if not os.path.exists("ml_models/image_model/model.pth"):
                return self._predict_semantic(image_path)
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transform
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                confidence_val = confidence.item()
                
                # Get top 3 predictions
                top_conf, top_indices = torch.topk(probabilities, 3)
            
            # If confidence is too low, fall back to semantic
            if confidence_val < 0.3:
                return self._predict_semantic(image_path)
            
            # Convert to readable format
            category = self.idx_to_category[predicted_idx.item()]
            
            top_categories = []
            for i in range(3):
                idx = top_indices[0][i].item()
                top_categories.append({
                    'category': self.idx_to_category[idx],
                    'confidence': top_conf[0][i].item()
                })
            
            # Check if restricted
            is_restricted = category in ["weapons", "drugs", "adult_content", "gambling"]
            
            return {
                'category': category,
                'confidence': confidence_val,
                'is_restricted': is_restricted,
                'top_categories': top_categories,
                'model_used': 'resnet_image_classifier'
            }
            
        except Exception as e:
            print(f"⚠️ Error predicting image: {e}")
            return self._predict_semantic(image_path)
    
    def _predict_semantic(self, image_path: str) -> Dict:
        """Semantic-based image prediction from filename and basic image analysis"""
        filename = os.path.basename(image_path).lower()
        
        # Count keyword matches
        scores = {}
        for category, keywords in self.category_keywords.items():
            match_count = sum(1 for keyword in keywords if keyword in filename)
            scores[category] = match_count
        
        # Try basic image analysis (color/brightness analysis as fallback)
        try:
            image = Image.open(image_path).convert('RGB')
            # Get average color
            pixels = np.array(image)
            avg_color = pixels.mean(axis=(0, 1))
            
            # Brightness analysis
            brightness = np.mean(avg_color)
            
            # Very simplistic heuristic
            if brightness > 200:  # Bright image
                scores["beauty"] += 1
                scores["fashion"] += 1
            elif brightness < 100:  # Dark image
                scores["weapons"] += 1
                scores["adult_content"] += 1
        except:
            pass
        
        # Find best match
        if max(scores.values()) == 0:
            # No keywords matched
            category = "unknown"
            confidence = 0.4
            top_categories = [{'category': cat, 'confidence': 0.1} for cat in list(self.categories)[:3]]
        else:
            # Sort by match count
            sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            category = sorted_categories[0][0]
            match_count = sorted_categories[0][1]
            
            # Confidence based on matches
            confidence = min(0.95, 0.5 + (match_count * 0.15))
            
            # Higher confidence for restricted categories if matched
            if category in ["weapons", "drugs", "adult_content", "gambling"] and match_count > 0:
                confidence = min(0.99, confidence + 0.25)
            
            # Top 3
            top_categories = []
            for cat, count in sorted_categories[:3]:
                cat_confidence = min(0.95, 0.3 + (count * 0.15))
                top_categories.append({
                    'category': cat,
                    'confidence': cat_confidence
                })
        
        # Check if restricted
        is_restricted = category in ["weapons", "drugs", "adult_content", "gambling"]
        
        return {
            'category': category,
            'confidence': confidence,
            'is_restricted': is_restricted,
            'top_categories': top_categories,
            'model_used': 'semantic_filename_analyzer'
        }
    
    def verify(self, image_path: str, business_profile: Optional[Dict] = None, 
               expected_domain: Optional[str] = None) -> Dict:
        """Complete verification with decision"""
        from .decision_engine import DecisionEngine
        
        # Get prediction
        prediction = self.predict(image_path)
        
        # Apply business rules
        decision_engine = DecisionEngine()
        decision = decision_engine.make_decision(
            prediction={'prediction': prediction},
            business_profile=business_profile,
            content_type='image'
        )
        
        # Calculate domain verification
        domain_verification = None
        domain_to_check = expected_domain or (business_profile.get('domain') if business_profile else None)
        
        if domain_to_check:
            domain_match = decision_engine.verify_domain_match(
                predicted_category=prediction['category'],
                expected_domain=domain_to_check,
                confidence=prediction['confidence']
            )
            
            # Add domain match score to prediction
            prediction['domain_match_score'] = domain_match['adjusted_score']
            domain_verification = domain_match
        
        return {
            'prediction': prediction,
            'decision': {
                'is_allowed': decision.is_allowed,
                'decision': decision.decision,
                'reason': decision.reason,
                'severity': decision.severity,
                'requires_review': decision.requires_review
            },
            'domain_match': domain_verification['is_match'] if domain_verification else False,
            'domain_verification_score': domain_verification['adjusted_score'] if domain_verification else 0.0,
            'domain_verification': domain_verification
        }