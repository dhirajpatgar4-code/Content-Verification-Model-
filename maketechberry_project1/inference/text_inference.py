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
            
            # Get top 3 predictions
            top_conf, top_indices = torch.topk(probabilities, 3)
        
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
            'confidence': confidence.item(),
            'is_restricted': is_restricted,
            'top_categories': top_categories,
            'model_used': 'bert_text_classifier'
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