import sys
sys.path.append('..')

from typing import Dict, Optional
import numpy as np

from .text_inference import TextInference
from .image_inference import ImageInference
from .decision_engine import DecisionEngine

class MultimodalInference:
    """Complete multimodal inference engine"""
    
    def __init__(self):
        self.text_inference = TextInference()
        self.image_inference = ImageInference()
        self.decision_engine = DecisionEngine()
        
        # Fusion weights
        self.text_weight = 0.7
        self.image_weight = 0.3
        
        # Load models
        self.text_inference.load_model()
        self.image_inference.load_model()
    
    def fuse_predictions(self, text_pred: Dict, image_pred: Dict, expected_domain: Optional[str] = None) -> Dict:
        """Fuse text and image predictions"""
        
        # If no image prediction, use text
        if not image_pred or image_pred['category'] == 'unknown':
            text_pred['fusion_method'] = 'text_only'
            return text_pred
        
        # If no text prediction, use image
        if not text_pred or text_pred['category'] == 'unknown':
            image_pred['fusion_method'] = 'image_only'
            return image_pred
        
        # Both predictions available - fuse them
        text_category = text_pred['category']
        text_confidence = text_pred['confidence']
        image_category = image_pred['category']
        image_confidence = image_pred['confidence']
        
        # Case 1: Same category
        if text_category == image_category:
            # Boost confidence
            fused_confidence = (text_confidence * 0.6 + image_confidence * 0.4) * 1.1
            fused_confidence = min(fused_confidence, 1.0)
            
            return {
                'category': text_category,
                'confidence': fused_confidence,
                'is_restricted': text_pred['is_restricted'] or image_pred['is_restricted'],
                'top_categories': text_pred['top_categories'],  # Use text's top categories
                'model_used': 'multimodal_fusion',
                'fusion_method': 'consensus',
                'text_confidence': text_confidence,
                'image_confidence': image_confidence
            }
        
        # Case 2: Different categories - weighted fusion
        text_score = text_confidence * self.text_weight
        image_score = image_confidence * self.image_weight
        
        if text_score >= image_score:
            # Text wins
            adjusted_confidence = text_confidence * 0.9  # Penalize for disagreement
            return {
                'category': text_category,
                'confidence': adjusted_confidence,
                'is_restricted': text_pred['is_restricted'] or image_pred['is_restricted'],
                'top_categories': text_pred['top_categories'],
                'model_used': 'multimodal_fusion',
                'fusion_method': 'text_weighted',
                'text_confidence': text_confidence,
                'image_confidence': image_confidence,
                'alternative_category': image_category
            }
        else:
            # Image wins
            adjusted_confidence = image_confidence * 0.9  # Penalize for disagreement
            return {
                'category': image_category,
                'confidence': adjusted_confidence,
                'is_restricted': text_pred['is_restricted'] or image_pred['is_restricted'],
                'top_categories': image_pred['top_categories'],
                'model_used': 'multimodal_fusion',
                'fusion_method': 'image_weighted',
                'text_confidence': text_confidence,
                'image_confidence': image_confidence,
                'alternative_category': text_category
            }
    
    def verify(self, text: str, title: str = "", image_path: Optional[str] = None, 
               business_profile: Optional[Dict] = None) -> Dict:
        """Complete multimodal verification"""
        
        # Get text prediction
        text_pred = self.text_inference.predict(text, title)
        
        # Get image prediction if image provided
        image_pred = None
        if image_path:
            image_pred = self.image_inference.predict(image_path)
        else:
            # Create dummy image prediction
            image_pred = {
                'category': 'unknown',
                'confidence': 0.0,
                'is_restricted': False,
                'top_categories': [],
                'model_used': 'none'
            }
        
        # Fuse predictions
        expected_domain = business_profile.get('domain') if business_profile else None
        fused_pred = self.fuse_predictions(text_pred, image_pred, expected_domain)
        
        # Apply business rules
        decision = self.decision_engine.make_decision(
            prediction={'prediction': fused_pred},
            business_profile=business_profile,
            content_type='mixed'
        )
        
        # Calculate domain verification
        domain_verification = None
        if business_profile and business_profile.get('domain'):
            domain_match = self.decision_engine.verify_domain_match(
                predicted_category=fused_pred['category'],
                expected_domain=business_profile['domain'],
                confidence=fused_pred['confidence']
            )
            
            # Add domain match score to prediction
            fused_pred['domain_match_score'] = domain_match['adjusted_score']
            domain_verification = domain_match
        
        return {
            'prediction': fused_pred,
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