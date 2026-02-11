# ml_models/universal_image_classifier.py
"""
Universal Image Classifier for Content Verification
Can detect ANY domain including restricted content
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define ALL possible domains (expandable)
ALL_DOMAINS = [
    # Restricted domains (ALWAYS BLOCKED)
    'weapons', 'drugs', 'adult_content', 'gambling',
    
    # Common commercial domains
    'education', 'sports', 'food', 'tech', 'health',
    'fashion', 'travel', 'entertainment', 'automotive',
    'real_estate', 'finance', 'beauty', 'home', 'electronics',
    
    # Additional domains
    'pets', 'art', 'music', 'books', 'gaming',
    'outdoors', 'fitness', 'business', 'science',
    
    # Fallback
    'unknown'
]

# Map domains to super-categories for training
DOMAIN_GROUPS = {
    'education': ['education', 'school', 'learning', 'classroom'],
    'sports': ['sports', 'fitness', 'exercise', 'athletics'],
    'food': ['food', 'restaurant', 'cooking', 'recipe'],
    'tech': ['tech', 'electronics', 'computer', 'software'],
    'health': ['health', 'medical', 'wellness', 'fitness'],
    'restricted': ['weapons', 'drugs', 'adult_content', 'gambling']
}

class UniversalImageClassifier:
    """Universal classifier for ANY domain"""
    
    def __init__(self, model_path="ml_models/universal_model/"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.domains = ALL_DOMAINS
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(self.domains)}
        self.idx_to_domain = {idx: domain for idx, domain in enumerate(self.domains)}
        
        # Restricted domains (always blocked)
        self.restricted_domains = ['weapons', 'drugs', 'adult_content', 'gambling']
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize zero-shot classifier for unlabelled images
        self.zero_shot_model = None
        
        # Try to load trained model
        self.model = None
        self.load_or_initialize_model()
        
        print(f"✅ Universal Image Classifier initialized")
        print(f"📊 Domains: {len(self.domains)} total ({len(self.restricted_domains)} restricted)")
    
    def load_or_initialize_model(self):
        """Load trained model or initialize for training"""
        try:
            if os.path.exists(os.path.join(self.model_path, "model.pth")):
                self.load_model()
                print(f"✅ Loaded trained model from {self.model_path}")
            else:
                print(f"⚠️ No trained model found. Using hybrid detection (keyword + zero-shot)")
                self.initialize_zero_shot()
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}. Using fallback methods.")
            self.initialize_zero_shot()
    
    def initialize_zero_shot(self):
        """Initialize zero-shot classifier for unlabelled images"""
        try:
            # Try to import CLIP for zero-shot classification
            import clip
            self.zero_shot_model, self.zero_shot_preprocess = clip.load("ViT-B/32", device=self.device)
            print("✅ CLIP zero-shot model loaded for unlabelled images")
        except ImportError:
            print("⚠️ CLIP not installed. Using keyword-based fallback only.")
            self.zero_shot_model = None
    
    def create_model(self, num_classes):
        """Create a universal classifier model"""
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except:
            model = models.resnet18(pretrained=True)
        
        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        return model
    
    def load_model(self):
        """Load trained universal model"""
        checkpoint = torch.load(
            os.path.join(self.model_path, "model.pth"),
            map_location=self.device,
            weights_only=False
        )
        
        self.model = self.create_model(len(self.domains))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs(self.model_path, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'domains': self.domains,
            'domain_to_idx': self.domain_to_idx,
            'idx_to_domain': self.idx_to_domain,
            'restricted_domains': self.restricted_domains
        }, os.path.join(self.model_path, "model.pth"))
        
        print(f"💾 Universal model saved to {self.model_path}")
    
    def predict_with_model(self, image_tensor, top_k=5):
        """Predict using trained model"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            predictions.append({
                'domain': self.idx_to_domain[idx],
                'confidence': top_probs[0][i].item(),
                'is_restricted': self.idx_to_domain[idx] in self.restricted_domains
            })
        
        return predictions
    
    def predict_with_clip(self, image_path, candidate_domains=None):
        """Zero-shot prediction using CLIP"""
        if self.zero_shot_model is None:
            return None
        
        try:
            import clip
            
            # Use provided domains or all non-restricted domains
            if candidate_domains is None:
                candidate_domains = [d for d in self.domains if d not in self.restricted_domains]
            
            # Prepare text prompts
            text_inputs = torch.cat([
                clip.tokenize(f"a photo of {domain}") for domain in candidate_domains
            ]).to(self.device)
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.zero_shot_preprocess(image).unsqueeze(0).to(self.device)
            
            # Calculate features
            with torch.no_grad():
                image_features = self.zero_shot_model.encode_image(image_input)
                text_features = self.zero_shot_model.encode_text(text_inputs)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(min(5, len(candidate_domains)))
            
            predictions = []
            for value, index in zip(values, indices):
                domain = candidate_domains[index.item()]
                predictions.append({
                    'domain': domain,
                    'confidence': value.item(),
                    'is_restricted': domain in self.restricted_domains,
                    'source': 'clip_zero_shot'
                })
            
            return predictions
            
        except Exception as e:
            print(f"⚠️ CLIP prediction failed: {e}")
            return None
    
    def keyword_based_detection(self, image_path, filename=""):
        """Fallback keyword-based detection"""
        # Extract keywords from filename
        filename_lower = filename.lower() if filename else ""
        path_lower = str(image_path).lower()
        
        # Domain keywords mapping
        domain_keywords = {
            'education': ['classroom', 'school', 'student', 'teacher', 'book', 'learning', 'education'],
            'sports': ['sports', 'football', 'basketball', 'game', 'player', 'stadium', 'athlete'],
            'food': ['food', 'restaurant', 'meal', 'dish', 'cooking', 'chef', 'kitchen'],
            'tech': ['computer', 'laptop', 'phone', 'electronic', 'software', 'code', 'tech'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'fitness', 'exercise', 'wellness'],
            'weapons': ['gun', 'weapon', 'rifle', 'knife', 'ammo', 'bullet', 'military'],
            'drugs': ['drug', 'cocaine', 'heroin', 'pill', 'tablet', 'narcotic', 'medicine'],
            'adult_content': ['porn', 'adult', 'xxx', 'nsfw', 'explicit', 'nude', 'sexy'],
            'gambling': ['casino', 'gambling', 'poker', 'betting', 'slot', 'lottery', 'bet'],
            'fashion': ['fashion', 'clothes', 'dress', 'shirt', 'shoes', 'bag', 'accessory'],
            'travel': ['travel', 'tourist', 'hotel', 'beach', 'mountain', 'airplane', 'vacation'],
            'automotive': ['car', 'vehicle', 'automotive', 'motor', 'engine', 'tire', 'auto']
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in filename_lower or keyword in path_lower:
                    score += 1
            
            if score > 0:
                confidence = min(0.3 + (score * 0.1), 0.8)  # Lower confidence for keywords
                scores[domain] = confidence
        
        if scores:
            best_domain = max(scores, key=scores.get)
            return [{
                'domain': best_domain,
                'confidence': scores[best_domain],
                'is_restricted': best_domain in self.restricted_domains,
                'source': 'keyword_detection'
            }]
        
        return None
    
    def predict(self, image_path, filename="", top_k=5, allowed_domains=None):
        """
        Universal prediction using multiple methods
        Priority: Trained Model > CLIP Zero-shot > Keywords > Unknown
        """
        results = {
            'predictions': [],
            'primary_domain': 'unknown',
            'primary_confidence': 0.0,
            'is_restricted': False,
            'detection_method': 'unknown',
            'all_domains': []
        }
        
        try:
            # Method 1: Trained model (if available)
            if self.model is not None:
                # Load and transform image
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                model_predictions = self.predict_with_model(image_tensor, top_k)
                if model_predictions:
                    results['predictions'] = model_predictions
                    results['primary_domain'] = model_predictions[0]['domain']
                    results['primary_confidence'] = model_predictions[0]['confidence']
                    results['is_restricted'] = model_predictions[0]['is_restricted']
                    results['detection_method'] = 'trained_model'
                    results['all_domains'] = [p['domain'] for p in model_predictions]
                    return results
            
            # Method 2: CLIP zero-shot
            if self.zero_shot_model is not None:
                # If allowed_domains provided, only check those
                candidate_domains = allowed_domains if allowed_domains else None
                clip_predictions = self.predict_with_clip(image_path, candidate_domains)
                
                if clip_predictions:
                    results['predictions'] = clip_predictions
                    results['primary_domain'] = clip_predictions[0]['domain']
                    results['primary_confidence'] = clip_predictions[0]['confidence']
                    results['is_restricted'] = clip_predictions[0]['is_restricted']
                    results['detection_method'] = 'clip_zero_shot'
                    results['all_domains'] = [p['domain'] for p in clip_predictions]
                    return results
            
            # Method 3: Keyword-based fallback
            keyword_predictions = self.keyword_based_detection(image_path, filename)
            if keyword_predictions:
                results['predictions'] = keyword_predictions
                results['primary_domain'] = keyword_predictions[0]['domain']
                results['primary_confidence'] = keyword_predictions[0]['confidence']
                results['is_restricted'] = keyword_predictions[0]['is_restricted']
                results['detection_method'] = 'keyword_detection'
                results['all_domains'] = [p['domain'] for p in keyword_predictions]
                return results
            
            # Method 4: Default to unknown
            results['predictions'] = [{
                'domain': 'unknown',
                'confidence': 0.1,
                'is_restricted': False,
                'source': 'unknown'
            }]
            results['primary_domain'] = 'unknown'
            results['primary_confidence'] = 0.1
            results['detection_method'] = 'unknown'
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            results['error'] = str(e)
            results['primary_domain'] = 'unknown'
            results['primary_confidence'] = 0.0
        
        return results
    
    def verify_content(self, image_path, user_domains, filename=""):
        """
        Complete verification for an image
        Returns: (is_allowed, reason, confidence, detected_domain)
        """
        # Step 1: Detect domain
        detection = self.predict(image_path, filename, allowed_domains=user_domains)
        
        detected_domain = detection['primary_domain']
        confidence = detection['primary_confidence']
        is_restricted = detection['is_restricted']
        
        # Step 2: Check if restricted
        if is_restricted:
            return {
                'is_allowed': False,
                'decision': 'blocked',
                'reason': f'Restricted content detected: {detected_domain}',
                'detected_domain': detected_domain,
                'confidence': confidence,
                'severity': 'high',
                'requires_review': False
            }
        
        # Step 3: Check if domain matches user's allowed domains
        if detected_domain == 'unknown':
            return {
                'is_allowed': False,
                'decision': 'needs_review',
                'reason': 'Cannot determine content domain',
                'detected_domain': 'unknown',
                'confidence': confidence,
                'severity': 'medium',
                'requires_review': True
            }
        
        if detected_domain in user_domains:
            if confidence > 0.7:
                return {
                    'is_allowed': True,
                    'decision': 'approved',
                    'reason': f'Content matches allowed domain: {detected_domain}',
                    'detected_domain': detected_domain,
                    'confidence': confidence,
                    'severity': 'low',
                    'requires_review': False
                }
            else:
                return {
                    'is_allowed': True,
                    'decision': 'approved_with_review',
                    'reason': f'Content likely matches {detected_domain} (confidence: {confidence:.1%})',
                    'detected_domain': detected_domain,
                    'confidence': confidence,
                    'severity': 'low',
                    'requires_review': True
                }
        else:
            return {
                'is_allowed': False,
                'decision': 'blocked',
                'reason': f'Content domain "{detected_domain}" not allowed for user',
                'detected_domain': detected_domain,
                'confidence': confidence,
                'severity': 'high',
                'requires_review': False
            }

# Global instance
_universal_classifier = None

def get_universal_classifier():
    """Get singleton classifier instance"""
    global _universal_classifier
    if _universal_classifier is None:
        try:
            _universal_classifier = UniversalImageClassifier()
        except Exception as e:
            print(f"⚠️ Failed to create universal classifier: {e}")
            _universal_classifier = None
    return _universal_classifier