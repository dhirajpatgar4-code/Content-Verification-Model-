import torch
import torch.nn as nn
import pickle
import os
import sys
sys.path.append('..')

from transformers import BertTokenizer, BertModel
from typing import Dict, Optional, List

class TextClassifier(nn.Module):
    """BERT-based text classifier"""
    def __init__(self, num_classes):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class TextInference:
    """Complete text inference engine"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.categories = []
        self.idx_to_category = {}
        self.loaded = False
        # Keyword mappings for fallback semantic detection
        self.category_keywords = {
            "food": ["recipe", "cook", "meal", "taste", "dish", "food", "eat", "restaurant", "cuisine", "ingredient"],
            "tech": ["software", "computer", "code", "algorithm", "programming", "app", "development", "tech", "digital", "gadget"],
            "education": ["learn", "school", "teacher", "student", "course", "university", "class", "study", "lecture", "education"],
            "health": ["doctor", "medicine", "health", "disease", "hospital", "patient", "treatment", "fitness", "exercise", "wellness"],
            "finance": ["money", "bank", "investment", "stock", "finance", "loan", "credit", "portfolio", "trading", "budget"],
            "fashion": ["dress", "fashion", "clothing", "style", "designer", "wear", "outfit", "trend", "apparel", "brand"],
            "electronics": ["phone", "camera", "monitor", "laptop", "tablet", "device", "gadget", "screen", "electronic", "circuit"],
            "automotive": ["car", "vehicle", "driver", "truck", "engine", "mechanic", "automotive", "bike", "motor", "transmission"],
            "real_estate": ["property", "house", "apartment", "rent", "building", "real estate", "land", "mortgage", "lease", "dwelling"],
            "entertainment": ["movie", "film", "music", "game", "entertainment", "show", "video", "actor", "concert", "performance"],
            "travel": ["trip", "flight", "hotel", "destination", "tourism", "journey", "tour", "travel", "airport", "vacation"],
            "beauty": ["cosmetic", "makeup", "skincare", "beauty", "hair", "salon", "nail", "fragrance", "treatment", "spa"],
            "home": ["furniture", "decor", "home", "appliance", "interior", "kitchen", "design", "room", "household", "living"],
            "sports": ["sport", "game", "player", "team", "match", "championship", "athletic", "competition", "exercise", "training"],
            "weapons": ["gun", "weapon", "bomb", "explosive", "firearm", "ammunition", "arsenal", "knife", "sword", "violence"],
            "drugs": ["drug", "narcotic", "cocaine", "heroin", "marijuana", "illegal", "substance", "addiction", "dealer", "abuse"],
            "adult_content": ["adult", "nude", "sexual", "sex", "explicit", "pornography", "xxx", "mature", "intimate", "erotic"],
            "gambling": ["gamble", "casino", "poker", "betting", "wager", "lottery", "slot", "gambling", "blackjack", "roulette"],
            "unknown": []
        }
    
    def load_model(self, model_path="ml_models/text_model/"):
        """Load trained text model"""
        try:
            # Load category mappings
            with open(os.path.join(model_path, "categories.pkl"), "rb") as f:
                mappings = pickle.load(f)
                self.categories = mappings['categories']
                self.idx_to_category = mappings['idx_to_category']
            
            # Initialize model
            self.model = TextClassifier(len(self.categories)).to(self.device)
            
            # Load weights
            model_file = os.path.join(model_path, "model.pth")
            if os.path.exists(model_file):
                self.model.load_state_dict(
                    torch.load(model_file, map_location=self.device)
                )
            else:
                print("⚠️ Model weights not found. Using pre-trained BERT only.")
            
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            
            self.loaded = True
            print("✅ Text model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading text model: {e}")
            # Initialize with fallback
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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TextClassifier(len(self.categories)).to(self.device)
        self.model.eval()
        self.loaded = True
        print("⚠️ Using fallback text model for testing")
    
    def predict(self, text: str, title: str = "") -> Dict:
        """Predict category for text"""
        if not self.loaded:
            self.load_model()
        
        # Combine title and text
        full_text = f"{title}: {text}" if title else text
        full_text_lower = full_text.lower()
        
        # If using fallback model, use semantic-based prediction
        if not os.path.exists("ml_models/text_model/model.pth"):
            return self._predict_semantic(full_text_lower)
        
        # Try trained model
        try:
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                confidence_val = confidence.item()
                
                # Get top 3 predictions
                top_conf, top_indices = torch.topk(probabilities, 3)
            
            # If confidence is too low, fall back to semantic
            if confidence_val < 0.3:
                return self._predict_semantic(full_text_lower)
            
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
                'model_used': 'bert_text_classifier'
            }
        except Exception as e:
            # Fall back to semantic if model fails
            print(f"⚠️ Model error, using semantic fallback: {e}")
            return self._predict_semantic(full_text_lower)
    
    def _predict_semantic(self, text: str) -> Dict:
        """Semantic-based prediction when model is not available"""
        text_lower = text.lower()
        
        # Count keyword matches for each category
        scores = {}
        for category, keywords in self.category_keywords.items():
            if category == "unknown":
                continue
            match_count = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = match_count
        
        # Find best matches
        if max(scores.values()) == 0:
            # No keywords matched
            category = "unknown"
            confidence = 0.5
            top_categories = [{'category': cat, 'confidence': 0.1} for cat in list(self.categories)[:3]]
        else:
            # Sort by match count
            sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Primary prediction
            category = sorted_categories[0][0]
            # Confidence based on keyword density
            match_count = sorted_categories[0][1]
            word_count = len(text_lower.split())
            confidence = min(0.95, (match_count / max(1, word_count / 10)))
            
            # Restrict categories always get high confidence if matched
            if category in ["weapons", "drugs", "adult_content", "gambling"] and match_count > 0:
                confidence = min(0.99, confidence + 0.3)
            
            # Get top 3
            top_categories = []
            for cat, count in sorted_categories[:3]:
                cat_confidence = (count / max(1, word_count / 10)) * 0.9
                top_categories.append({
                    'category': cat,
                    'confidence': min(0.95, cat_confidence)
                })
        
        # Check if restricted
        is_restricted = category in ["weapons", "drugs", "adult_content", "gambling"]
        
        return {
            'category': category,
            'confidence': confidence,
            'is_restricted': is_restricted,
            'top_categories': top_categories,
            'model_used': 'semantic_keyword_detector'
        }
    
    def verify(self, text: str, title: str = "", business_profile: Optional[Dict] = None) -> Dict:
        """Complete verification with decision"""
        from .decision_engine import DecisionEngine
        
        # Get prediction
        prediction = self.predict(text, title)
        
        # Apply business rules
        decision_engine = DecisionEngine()
        decision = decision_engine.make_decision(
            prediction={'prediction': prediction},
            business_profile=business_profile,
            content_type='text'
        )
        
        # Calculate domain verification
        domain_verification = None
        if business_profile and business_profile.get('domain'):
            domain_match = decision_engine.verify_domain_match(
                predicted_category=prediction['category'],
                expected_domain=business_profile['domain'],
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