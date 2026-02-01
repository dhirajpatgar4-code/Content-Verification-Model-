from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ContentVerification(Base):
    __tablename__ = 'content_verifications'
    
    id = Column(Integer, primary_key=True)
    verification_id = Column(String(100), unique=True, index=True)
    business_id = Column(String(50), nullable=False, index=True)
    
    # Content details
    content_type = Column(String(20))
    title = Column(Text)
    description = Column(Text)
    image_path = Column(String(500))
    
    # ML Predictions
    predicted_category = Column(String(100))
    confidence_score = Column(Float)
    is_restricted = Column(Boolean, default=False)
    ml_model_used = Column(String(50))
    prediction_details = Column(JSON)
    
    # Decision
    is_allowed = Column(Boolean)
    decision = Column(String(50))
    decision_reason = Column(Text)
    severity = Column(String(20))
    requires_human_review = Column(Boolean, default=False)
    
    # Domain verification
    expected_domain = Column(String(100))
    domain_match = Column(Boolean)
    domain_verification_score = Column(Float)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    verified_at = Column(DateTime, default=datetime.utcnow)
    posted_at = Column(DateTime, nullable=True)
    status = Column(String(20), default='verified')
    
    # Performance
    verification_time_ms = Column(Integer)
    content_length = Column(Integer)

class BusinessProfile(Base):
    __tablename__ = 'business_profiles'
    
    id = Column(Integer, primary_key=True)
    business_id = Column(String(50), unique=True, nullable=False, index=True)
    business_name = Column(String(255))
    business_type = Column(String(50))
    domain = Column(String(100))
    allowed_domains = Column(JSON)
    restricted_categories = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PostedContent(Base):
    __tablename__ = 'posted_content'
    
    id = Column(Integer, primary_key=True)
    verification_id = Column(String(100), unique=True)
    business_id = Column(String(50), nullable=False, index=True)
    content_id = Column(String(100))
    posted_at = Column(DateTime, default=datetime.utcnow)
    platform = Column(String(50))
    views = Column(Integer, default=0)
    engagements = Column(Integer, default=0)