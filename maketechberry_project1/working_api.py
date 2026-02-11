#!/usr/bin/env python3
"""
WORKING Content Verification API - With proper keyword detection
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime
import re

app = FastAPI(title="Content Verification API", version="1.0.0")

class TextContent(BaseModel):
    text: str
    title: Optional[str] = None
    business_id: Optional[str] = None

class VerificationResponse(BaseModel):
    content_type: str
    prediction: dict
    decision: dict
    timestamp: str
    verification_id: str

def detect_content_type(text, business_id=None):
    """Keyword-based content detection that actually works"""
    text_lower = text.lower()
    
    # Business domain mapping
    business_domains = {
        "EDU001": "education",
        "SPORTS001": "sports", 
        "FOOD001": "food",
        "TECH001": "tech",
        "MARKET001": "marketplace"
    }
    
    # RESTRICTED CONTENT (HIGHEST PRIORITY)
    weapons_patterns = [
        r'\bgun\b', r'\bweapon\b', r'\bfirearm\b', r'\brifle\b', r'\bpistol\b',
        r'\bammo\b', r'\bammunition\b', r'\bbullet\b', r'\bshoot\b', r'\bkill\b'
    ]
    for pattern in weapons_patterns:
        if re.search(pattern, text_lower):
            return {
                "category": "weapons",
                "confidence": 0.95,
                "is_restricted": True,
                "keywords_found": ["weapons"]
            }
    
    drugs_patterns = [
        r'\bdrug\b', r'\bcocaine\b', r'\bheroin\b', r'\bmarijuana\b',
        r'\bopioid\b', r'\bprescription\b'
    ]
    for pattern in drugs_patterns:
        if re.search(pattern, text_lower):
            return {
                "category": "drugs", 
                "confidence": 0.92,
                "is_restricted": True,
                "keywords_found": ["drugs"]
            }
    
    gambling_patterns = [r'\bcasino\b', r'\bgambling\b', r'\bbetting\b', r'\bpoker\b']
    for pattern in gambling_patterns:
        if re.search(pattern, text_lower):
            return {
                "category": "gambling",
                "confidence": 0.90,
                "is_restricted": True,
                "keywords_found": ["gambling"]
            }
    
    # EDUCATION CONTENT
    education_patterns = [
        r'\bclassroom\b', r'\blearning\b', r'\bstudent\b', r'\bteacher\b',
        r'\bschool\b', r'\beducation\b', r'\bphysics\b', r'\bchemistry\b',
        r'\bbiology\b', r'\bmath\b', r'\bscience\b', r'\blab\b',
        r'\blaboratory\b', r'\bexperiment\b'
    ]
    education_count = 0
    for pattern in education_patterns:
        if re.search(pattern, text_lower):
            education_count += 1
    
    if education_count >= 2:
        return {
            "category": "education",
            "confidence": min(0.7 + (education_count * 0.05), 0.95),
            "is_restricted": False,
            "keywords_found": education_count
        }
    elif education_count == 1:
        return {
            "category": "education",
            "confidence": 0.65,
            "is_restricted": False,
            "keywords_found": 1
        }
    
    # SPORTS CONTENT
    sports_patterns = [
        r'\bsports\b', r'\bfootball\b', r'\bbasketball\b', r'\bcricket\b',
        r'\bgym\b', r'\bfitness\b', r'\bexercise\b', r'\btraining\b',
        r'\byoga\b', r'\bmeditation\b'
    ]
    sports_count = 0
    for pattern in sports_patterns:
        if re.search(pattern, text_lower):
            sports_count += 1
    
    if sports_count >= 2:
        return {
            "category": "sports",
            "confidence": min(0.75 + (sports_count * 0.04), 0.95),
            "is_restricted": False,
            "keywords_found": sports_count
        }
    elif sports_count == 1:
        return {
            "category": "sports", 
            "confidence": 0.68,
            "is_restricted": False,
            "keywords_found": 1
        }
    
    # FOOD CONTENT
    food_patterns = [
        r'\bfood\b', r'\brestaurant\b', r'\brecipe\b', r'\bcooking\b',
        r'\bmeal\b', r'\bpizza\b', r'\bburger\b', r'\bpasta\b'
    ]
    food_count = 0
    for pattern in food_patterns:
        if re.search(pattern, text_lower):
            food_count += 1
    
    if food_count >= 2:
        return {
            "category": "food",
            "confidence": min(0.72 + (food_count * 0.04), 0.95),
            "is_restricted": False,
            "keywords_found": food_count
        }
    
    # TECH CONTENT
    tech_patterns = [
        r'\btech\b', r'\bcomputer\b', r'\bsoftware\b', r'\bprogramming\b',
        r'\bcode\b', r'\bpython\b', r'\bjava\b', r'\bapp\b', r'\bwebsite\b'
    ]
    tech_count = 0
    for pattern in tech_patterns:
        if re.search(pattern, text_lower):
            tech_count += 1
    
    if tech_count >= 2:
        return {
            "category": "tech",
            "confidence": min(0.78 + (tech_count * 0.04), 0.95),
            "is_restricted": False,
            "keywords_found": tech_count
        }
    
    # DEFAULT
    return {
        "category": "unknown",
        "confidence": 0.3,
        "is_restricted": False,
        "keywords_found": 0
    }

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
    
    # Check confidence
    if confidence < 0.4:
        return {
            "is_allowed": False,
            "decision": "needs_review",
            "reason": f"Low confidence prediction ({confidence:.2f})",
            "severity": "medium",
            "requires_review": True
        }
    
    # Apply business domain rules
    if business_id:
        # Business domain mapping
        business_rules = {
            "EDU001": {
                "type": "single_domain",
                "domain": "education",
                "related": ["sports", "health", "tech"]  # Education can include sports (PE), health, tech
            },
            "SPORTS001": {
                "type": "single_domain", 
                "domain": "sports",
                "related": ["health", "education"]  # Sports can include health, education (PE theory)
            },
            "FOOD001": {
                "type": "single_domain",
                "domain": "food",
                "related": []
            },
            "TECH001": {
                "type": "single_domain",
                "domain": "tech",
                "related": ["education"]  # Tech can include education (edtech)
            },
            "MARKET001": {
                "type": "marketplace",
                "domain": None,
                "allowed": ["education", "sports", "health", "tech", "food", "fashion"]
            }
        }
        
        if business_id in business_rules:
            rules = business_rules[business_id]
            
            if rules["type"] == "single_domain":
                # Single-domain business
                if category == rules["domain"] or category in rules["related"]:
                    if confidence >= 0.6:
                        return {
                            "is_allowed": True,
                            "decision": "approved",
                            "reason": f"Content matches or is related to business domain '{rules['domain']}'",
                            "severity": "low",
                            "requires_review": False
                        }
                    else:
                        return {
                            "is_allowed": False,
                            "decision": "needs_review",
                            "reason": f"Domain matches but confidence is low ({confidence:.2f})",
                            "severity": "medium", 
                            "requires_review": True
                        }
                else:
                    return {
                        "is_allowed": False,
                        "decision": "blocked",
                        "reason": f"Content category '{category}' does not match business domain '{rules['domain']}'",
                        "severity": "high",
                        "requires_review": False
                    }
            
            else:  # marketplace
                # Marketplace can post anything except restricted (already handled)
                if confidence >= 0.5:
                    return {
                        "is_allowed": True,
                        "decision": "approved",
                        "reason": "Marketplace content approved",
                        "severity": "low",
                        "requires_review": False
                    }
                else:
                    return {
                        "is_allowed": False,
                        "decision": "needs_review",
                        "reason": f"Low confidence ({confidence:.2f}) - requires review",
                        "severity": "medium",
                        "requires_review": True
                    }
    
    # Default for no business ID
    if confidence >= 0.7:
        return {
            "is_allowed": True,
            "decision": "approved",
            "reason": f"Content verified with high confidence ({confidence:.2f})",
            "severity": "low",
            "requires_review": False
        }
    else:
        return {
            "is_allowed": False,
            "decision": "needs_review",
            "reason": f"Medium confidence ({confidence:.2f}) - requires review",
            "severity": "medium",
            "requires_review": True
        }

@app.get("/")
async def root():
    return {"message": "Content Verification API", "status": "working"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/verify/text", response_model=VerificationResponse)
async def verify_text(content: TextContent):
    try:
        # Detect content
        detection = detect_content_type(content.text, content.business_id)
        
        # Create prediction object
        prediction = {
            "category": detection['category'],
            "confidence": detection['confidence'],
            "is_restricted": detection['is_restricted'],
            "top_categories": [
                {"category": detection['category'], "confidence": detection['confidence']}
            ],
            "model_used": "keyword_detector"
        }
        
        # Make decision
        decision = make_decision(prediction, content.business_id)
        
        # Create response
        response = VerificationResponse(
            content_type="text",
            prediction=prediction,
            decision=decision,
            timestamp=datetime.now().isoformat(),
            verification_id=f"VER_{uuid.uuid4().hex[:8].upper()}"
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recent-verifications")
async def get_recent():
    return {
        "verifications": [
            {
                "verification_id": "DEMO_001",
                "business_id": "EDU001",
                "title": "Sample Education Content",
                "predicted_category": "education",
                "confidence_score": 0.88,
                "decision": "approved",
                "created_at": datetime.now().isoformat()
            }
        ]
    }

@app.get("/api/analytics")
async def get_analytics():
    return {
        "total_verifications": 10,
        "approved_count": 7,
        "blocked_count": 2,
        "review_count": 1,
        "approval_rate": 70.0,
        "average_confidence": 0.75
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting WORKING Content Verification API...")
    print("✅ Keyword-based detection enabled")
    print("✅ Business logic implemented")
    print("📚 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)