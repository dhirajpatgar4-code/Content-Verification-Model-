from typing import Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class Decision:
    is_allowed: bool
    decision: str  # "approved", "blocked", "needs_review"
    reason: str
    severity: str  # "high", "medium", "low"
    requires_review: bool

class DecisionEngine:
    """Complete decision engine following your business logic"""
    
    def __init__(self, config_path="config/business_rules.json"):
        self.restricted_categories = [
            "weapons", "drugs", "adult_content", "gambling",
            "explosives", "poisons", "counterfeit", "human_organs",
            "surveillance_devices", "live_animals"
        ]
        
        # Confidence thresholds
        self.thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Domain verification thresholds
        self.domain_thresholds = {
            "strict": 0.7,    # Single-domain businesses
            "moderate": 0.5,  # Related domains
            "lenient": 0.3    # Marketplace
        }
        
        # Related domains mapping
        self.related_domains = {
            "education": ["tech", "sports", "health"],
            "sports": ["health", "education"],
            "health": ["sports", "education"],
            "tech": ["education", "electronics"],
            "fashion": ["beauty", "home"],
            "travel": ["entertainment", "home"],
            "real_estate": ["home", "finance"]
        }
        
        # Try to load config
        self._load_config(config_path)
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        try:
            import os
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.restricted_categories = config.get('restricted_categories', self.restricted_categories)
                    self.thresholds = config.get('thresholds', self.thresholds)
        except:
            pass  # Use defaults if config fails to load
    
    def make_decision(self, prediction: Dict, business_profile: Optional[Dict] = None, 
                     content_type: str = "text") -> Decision:
        """Make final decision based on ML prediction and business rules"""
        
        pred_data = prediction.get('prediction', {})
        category = pred_data.get('category', 'unknown')
        confidence = pred_data.get('confidence', 0.0)
        is_restricted = pred_data.get('is_restricted', False)
        
        # STEP 1: Check for restricted content (ALWAYS BLOCK)
        if is_restricted or category in self.restricted_categories:
            return Decision(
                is_allowed=False,
                decision="blocked",
                reason=f"Content contains restricted material: {category}",
                severity="high",
                requires_review=False
            )
        
        # STEP 2: Check confidence
        if confidence < self.thresholds["low"]:
            return Decision(
                is_allowed=False,
                decision="needs_review",
                reason=f"Low confidence prediction ({confidence:.2f})",
                severity="medium",
                requires_review=True
            )
        
        # STEP 3: Apply business domain rules
        if business_profile:
            return self._apply_business_rules(category, confidence, business_profile, content_type)
        
        # STEP 4: Default decision (no business profile)
        if confidence >= self.thresholds["high"]:
            return Decision(
                is_allowed=True,
                decision="approved",
                reason=f"Content verified with high confidence ({confidence:.2f})",
                severity="low",
                requires_review=False
            )
        else:
            return Decision(
                is_allowed=False,
                decision="needs_review",
                reason=f"Medium confidence ({confidence:.2f}) - requires review",
                severity="medium",
                requires_review=True
            )
    
    def _apply_business_rules(self, category: str, confidence: float, 
                            business_profile: Dict, content_type: str) -> Decision:
        """Apply your specific business logic"""
        
        business_type = business_profile.get('business_type', 'single_domain')
        business_domain = business_profile.get('domain', 'unknown')
        allowed_domains = business_profile.get('allowed_domains', [])
        
        # CASE 1: SINGLE-DOMAIN BUSINESS
        if business_type == 'single_domain':
            # Check if category matches business domain or related domains
            if category == business_domain:
                # Exact match
                if confidence >= self.thresholds["medium"]:
                    return Decision(
                        is_allowed=True,
                        decision="approved",
                        reason=f"Content exactly matches business domain '{business_domain}'",
                        severity="low",
                        requires_review=False
                    )
                else:
                    return Decision(
                        is_allowed=False,
                        decision="needs_review",
                        reason=f"Content matches domain but confidence is low ({confidence:.2f})",
                        severity="medium",
                        requires_review=True
                    )
            
            # Check related domains
            elif business_domain in self.related_domains and category in self.related_domains[business_domain]:
                # Related domain match
                if confidence >= self.thresholds["medium"]:
                    return Decision(
                        is_allowed=True,
                        decision="approved",
                        reason=f"Content is related to business domain '{business_domain}'",
                        severity="low",
                        requires_review=False
                    )
                else:
                    return Decision(
                        is_allowed=False,
                        decision="needs_review",
                        reason=f"Content is related but confidence is low ({confidence:.2f})",
                        severity="medium",
                        requires_review=True
                    )
            else:
                # No match
                return Decision(
                    is_allowed=False,
                    decision="blocked",
                    reason=f"Content category '{category}' does not match business domain '{business_domain}'",
                    severity="high",
                    requires_review=False
                )
        
        # CASE 2: MARKETPLACE BUSINESS
        else:
            # Marketplaces check what they ARE NOT allowed to post (restricted content already handled)
            if allowed_domains and category not in allowed_domains:
                return Decision(
                    is_allowed=False,
                    decision="blocked",
                    reason=f"Content category '{category}' is not in allowed domains",
                    severity="high",
                    requires_review=False
                )
            
            # Check confidence
            if confidence >= self.thresholds["medium"]:
                return Decision(
                    is_allowed=True,
                    decision="approved",
                    reason="Marketplace content approved",
                    severity="low",
                    requires_review=False
                )
            else:
                return Decision(
                    is_allowed=False,
                    decision="needs_review",
                    reason=f"Low confidence ({confidence:.2f}) - requires review",
                    severity="medium",
                    requires_review=True
                )
    
    def verify_domain_match(self, predicted_category: str, expected_domain: str, 
                           confidence: float = 1.0) -> Dict:
        """Verify if predicted category matches expected domain"""
        
        # Calculate similarity
        similarity = self._calculate_domain_similarity(predicted_category, expected_domain)
        
        # Adjust with confidence
        adjusted_similarity = similarity * confidence
        
        # Determine if it's a match based on business type
        is_match = adjusted_similarity >= self.domain_thresholds["moderate"]
        
        return {
            "expected_domain": expected_domain,
            "predicted_category": predicted_category,
            "similarity_score": similarity,
            "adjusted_score": adjusted_similarity,
            "is_match": is_match,
            "threshold_used": self.domain_thresholds["moderate"],
            "confidence_weight": confidence
        }
    
    def _calculate_domain_similarity(self, category1: str, category2: str) -> float:
        """Calculate similarity between two domains"""
        if category1 == category2:
            return 1.0
        
        # Check if directly related
        for domain, related in self.related_domains.items():
            if category1 == domain and category2 in related:
                return 0.8
            if category2 == domain and category1 in related:
                return 0.8
        
        # Check if in same related group
        for related_group in self.related_domains.values():
            if category1 in related_group and category2 in related_group:
                return 0.7
        
        return 0.0