from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class TextContent(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    title: Optional[str] = Field(None, max_length=200)
    business_id: Optional[str] = Field(None, max_length=50)
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or v.isspace():
            raise ValueError('Text cannot be empty')
        return v

class ImageContent(BaseModel):
    image_url: Optional[str] = None
    business_id: Optional[str] = Field(None, max_length=50)
    expected_domain: Optional[str] = Field(None, max_length=100)

class MixedContent(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    image_url: Optional[str] = None
    title: Optional[str] = Field(None, max_length=200)
    business_id: Optional[str] = Field(None, max_length=50)

class PredictionResult(BaseModel):
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    subcategory: Optional[str] = None
    is_restricted: bool = False
    top_categories: List[Dict[str, Any]] = []
    model_used: Optional[str] = None
    domain_match_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class DecisionResult(BaseModel):
    is_allowed: bool
    decision: str  # "approved", "blocked", "needs_review"
    reason: str
    severity: str  # "high", "medium", "low"
    requires_review: bool

class VerificationResponse(BaseModel):
    content_type: str
    prediction: PredictionResult
    decision: DecisionResult
    timestamp: str
    verification_id: Optional[str] = None
    domain_verification: Optional[Dict[str, Any]] = None

class BusinessProfile(BaseModel):
    business_id: str
    business_name: str
    business_type: str  # "single_domain", "marketplace"
    domain: Optional[str]
    allowed_domains: List[str] = []
    restricted_categories: List[str] = []
    
    @validator('business_type')
    def validate_business_type(cls, v):
        if v not in ['single_domain', 'marketplace']:
            raise ValueError('business_type must be either "single_domain" or "marketplace"')
        return v

class AnalyticsResponse(BaseModel):
    total_verifications: int
    approved_count: int
    blocked_count: int
    review_count: int
    approval_rate: float
    average_confidence: float
    domain_distribution: Dict[str, int] = {}
    category_distribution: Dict[str, int] = {}