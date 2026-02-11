#!/usr/bin/env python3
"""
Simple keyword-based classifier for immediate testing
"""

import re
from typing import Dict, List
import numpy as np

class SimpleTextClassifier:
    """Keyword-based text classifier that actually works"""
    
    def __init__(self):
        # Define keywords for each category
        self.keyword_map = {
            # Education keywords
            "education": [
                "classroom", "learning", "student", "teacher", "school", 
                "education", "study", "course", "lesson", "academic",
                "university", "college", "training", "workshop", "seminar",
                "physics", "chemistry", "biology", "math", "science",
                "lab", "laboratory", "experiment", "research", "project"
            ],
            
            # Sports keywords
            "sports": [
                "sports", "football", "basketball", "cricket", "athletics",
                "fitness", "exercise", "training", "workout", "gym",
                "yoga", "meditation", "stretching", "warmup", "physical",
                "coach", "team", "match", "game", "tournament"
            ],
            
            # Health keywords
            "health": [
                "health", "wellness", "fitness", "exercise", "yoga",
                "meditation", "nutrition", "diet", "wellbeing", "therapy",
                "doctor", "hospital", "medical", "medicine", "treatment"
            ],
            
            # Food keywords
            "food": [
                "food", "restaurant", "recipe", "cooking", "meal",
                "pizza", "burger", "pasta", "rice", "bread",
                "chef", "kitchen", "dining", "eat", "drink"
            ],
            
            # Tech keywords
            "tech": [
                "tech", "computer", "software", "programming", "code",
                "python", "java", "javascript", "app", "website",
                "mobile", "phone", "laptop", "electronic", "device"
            ],
            
            # Restricted categories (HIGH PRIORITY)
            "weapons": [
                "gun", "weapon", "firearm", "rifle", "pistol",
                "ammunition", "bullet", "shoot", "kill", "violence",
                "bomb", "explosive", "knife", "sword", "army"
            ],
            
            "drugs": [
                "drug", "cocaine", "heroin", "marijuana", "opioid",
                "prescription", "medicine", "pill", "tablet", "inject",
                "illegal", "narcotic", "addict", "overdose", "pharmacy"
            ],
            
            "adult_content": [
                "porn", "xxx", "adult", "sex", "nude",
                "explicit", "nsfw", "erotic", "xxx", "hentai"
            ],
            
            "gambling": [
                "casino", "gambling", "betting", "poker", "lottery",
                "bet", "wager", "slot", "roulette", "blackjack"
            ]
        }
        
        # All categories
        self.categories = list(self.keyword_map.keys()) + ["unknown"]
        
        # Priority: Restricted categories checked first
        self.restricted_categories = ["weapons", "drugs", "adult_content", "gambling"]
        
        # Business domains
        self.business_domains = {
            "EDU001": "education",
            "SPORTS001": "sports",
            "HEALTH001": "health",
            "FOOD001": "food",
            "TECH001": "tech",
            "MARKET001": "marketplace"
        }
    
    def predict(self, text: str, title: str = "") -> Dict:
        """Predict category with confidence"""
        full_text = f"{title} {text}".lower()
        
        scores = {}
        
        # Calculate scores for each category
        for category, keywords in self.keyword_map.items():
            score = 0
            for keyword in keywords:
                # Count occurrences with weight
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', full_text))
                score += count * (3 if category in self.restricted_categories else 1)
            
            if score > 0:
                scores[category] = score
        
        # Normalize scores to confidence
        if scores:
            total_score = sum(scores.values())
            
            # Get top category
            top_category = max(scores, key=scores.get)
            confidence = scores[top_category] / (total_score + 5)  # Normalize
            
            # Boost confidence if high score
            if scores[top_category] >= 3:
                confidence = min(confidence * 1.5, 0.95)
            
            # Get top 3
            sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_categories = []
            
            for cat, score in sorted_categories:
                cat_confidence = score / (total_score + 5)
                top_categories.append({
                    "category": cat,
                    "confidence": min(cat_confidence, 0.95)
                })
            
            # Check if restricted
            is_restricted = top_category in self.restricted_categories
            
            return {
                "category": top_category,
                "confidence": min(confidence, 0.95),
                "is_restricted": is_restricted,
                "top_categories": top_categories,
                "model_used": "keyword_classifier"
            }
        else:
            # No keywords found
            return {
                "category": "unknown",
                "confidence": 0.1,
                "is_restricted": False,
                "top_categories": [{"category": "unknown", "confidence": 0.1}],
                "model_used": "keyword_classifier"
            }
    
    def verify(self, text: str, title: str = "", business_id: str = None) -> Dict:
        """Complete verification with decision logic"""
        from .decision_engine import DecisionEngine
        
        # Get prediction
        prediction = self.predict(text, title)
        
        # Get business profile
        business_profile = None
        if business_id and business_id in self.business_domains:
            if business_id == "MARKET001":
                business_profile = {
                    "business_type": "marketplace",
                    "domain": None,
                    "allowed_domains": ["education", "sports", "health", "tech", "food", "fashion"]
                }
            else:
                business_profile = {
                    "business_type": "single_domain",
                    "domain": self.business_domains[business_id]
                }
        
        # Apply business rules
        decision_engine = DecisionEngine()
        decision = decision_engine.make_decision(
            prediction={"prediction": prediction},
            business_profile=business_profile,
            content_type="text"
        )
        
        # Calculate domain verification
        domain_verification = None
        if business_profile and business_profile.get('domain'):
            domain_match = decision_engine.verify_domain_match(
                predicted_category=prediction['category'],
                expected_domain=business_profile['domain'],
                confidence=prediction['confidence']
            )
            
            prediction['domain_match_score'] = domain_match['adjusted_score']
            domain_verification = domain_match
        
        return {
            "prediction": prediction,
            "decision": {
                "is_allowed": decision.is_allowed,
                "decision": decision.decision,
                "reason": decision.reason,
                "severity": decision.severity,
                "requires_review": decision.requires_review
            },
            "domain_match": domain_verification['is_match'] if domain_verification else False,
            "domain_verification_score": domain_verification['adjusted_score'] if domain_verification else 0.0,
            "domain_verification": domain_verification
        }