from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
import os
import json
from datetime import datetime, timedelta

# Import models from schemas
from api.schemas import Base, ContentVerification, BusinessProfile, PostedContent

class DatabaseManager:
    """Complete database manager"""
    
    def __init__(self, db_path="content_verification.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
    
    def save_verification_result(self, verification_data):
        """Save verification result to database"""
        session = self.Session()
        try:
            verification = ContentVerification(
                verification_id=verification_data['verification_id'],
                business_id=verification_data['business_id'],
                content_type=verification_data['content_type'],
                title=verification_data.get('title'),
                description=verification_data.get('description'),
                image_path=verification_data.get('image_path'),
                predicted_category=verification_data['prediction']['category'],
                confidence_score=verification_data['prediction']['confidence'],
                is_restricted=verification_data['prediction'].get('is_restricted', False),
                ml_model_used=verification_data['prediction'].get('model_used', 'unknown'),
                prediction_details=json.dumps(verification_data['prediction']),
                is_allowed=verification_data['decision']['is_allowed'],
                decision=verification_data['decision']['decision'],
                decision_reason=verification_data['decision']['reason'],
                severity=verification_data['decision']['severity'],
                requires_human_review=verification_data['decision']['requires_review'],
                expected_domain=verification_data.get('expected_domain'),
                domain_match=verification_data.get('domain_match', False),
                domain_verification_score=verification_data.get('domain_verification_score', 0.0),
                verification_time_ms=verification_data.get('verification_time_ms', 0),
                content_length=len(str(verification_data.get('description', ''))),
                verified_at=datetime.utcnow()
            )
            session.add(verification)
            session.commit()
            return verification.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_business_profile(self, business_id):
        """Get business profile from database"""
        session = self.Session()
        try:
            business = session.query(BusinessProfile).filter_by(business_id=business_id).first()
            if business:
                return {
                    'business_id': business.business_id,
                    'business_name': business.business_name,
                    'business_type': business.business_type,
                    'domain': business.domain,
                    'allowed_domains': json.loads(business.allowed_domains) if business.allowed_domains else [],
                    'restricted_categories': json.loads(business.restricted_categories) if business.restricted_categories else []
                }
            return None
        finally:
            session.close()
    
    def save_business_profile(self, business_data):
        """Save or update business profile"""
        session = self.Session()
        try:
            business = session.query(BusinessProfile).filter_by(business_id=business_data['business_id']).first()
            
            if business:
                # Update existing
                business.business_name = business_data.get('business_name', business.business_name)
                business.business_type = business_data.get('business_type', business.business_type)
                business.domain = business_data.get('domain', business.domain)
                business.allowed_domains = json.dumps(business_data.get('allowed_domains', []))
                business.restricted_categories = json.dumps(business_data.get('restricted_categories', []))
                business.updated_at = datetime.utcnow()
            else:
                # Create new
                business = BusinessProfile(
                    business_id=business_data['business_id'],
                    business_name=business_data.get('business_name', ''),
                    business_type=business_data['business_type'],
                    domain=business_data.get('domain'),
                    allowed_domains=json.dumps(business_data.get('allowed_domains', [])),
                    restricted_categories=json.dumps(business_data.get('restricted_categories', [])),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(business)
            
            session.commit()
            return business.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def save_posted_content(self, verification_id, business_id, content_id, platform="website"):
        """Mark content as posted"""
        session = self.Session()
        try:
            # Update verification status
            verification = session.query(ContentVerification).filter_by(
                verification_id=verification_id
            ).first()
            if verification:
                verification.posted_at = datetime.utcnow()
                verification.status = 'posted'
            
            # Add to posted content
            posted = PostedContent(
                verification_id=verification_id,
                business_id=business_id,
                content_id=content_id,
                platform=platform,
                posted_at=datetime.utcnow()
            )
            session.add(posted)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_analytics(self, business_id=None, days=30):
        """Get analytics data"""
        session = self.Session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Build query
            query = session.query(ContentVerification).filter(
                ContentVerification.created_at >= start_date
            )
            
            if business_id:
                query = query.filter_by(business_id=business_id)
            
            verifications = query.all()
            
            # Calculate metrics
            total = len(verifications)
            approved = sum(1 for v in verifications if v.decision == 'approved')
            blocked = sum(1 for v in verifications if v.decision == 'blocked')
            review = sum(1 for v in verifications if v.decision == 'needs_review')
            
            # Calculate average confidence
            avg_confidence = 0
            if total > 0:
                avg_confidence = sum(v.confidence_score for v in verifications) / total
            
            # Get category distribution
            category_dist = {}
            for v in verifications:
                cat = v.predicted_category
                category_dist[cat] = category_dist.get(cat, 0) + 1
            
            # Get domain distribution
            domain_dist = {}
            for v in verifications:
                if v.expected_domain:
                    domain = v.expected_domain
                    domain_dist[domain] = domain_dist.get(domain, 0) + 1
            
            return {
                'total_verifications': total,
                'approved_count': approved,
                'blocked_count': blocked,
                'review_count': review,
                'approval_rate': (approved / total * 100) if total > 0 else 0,
                'average_confidence': avg_confidence,
                'category_distribution': category_dist,
                'domain_distribution': domain_dist
            }
        finally:
            session.close()
    
    def get_recent_verifications(self, limit=10):
        """Get recent verifications"""
        session = self.Session()
        try:
            verifications = session.query(ContentVerification).order_by(
                ContentVerification.created_at.desc()
            ).limit(limit).all()
            
            result = []
            for v in verifications:
                result.append({
                    'verification_id': v.verification_id,
                    'business_id': v.business_id,
                    'content_type': v.content_type,
                    'title': v.title,
                    'predicted_category': v.predicted_category,
                    'confidence_score': float(v.confidence_score),
                    'decision': v.decision,
                    'created_at': v.created_at.isoformat() if v.created_at else None,
                    'requires_review': v.requires_human_review
                })
            
            return result
        finally:
            session.close()
    
    def initialize_sample_data(self):
        """Initialize database with sample data"""
        session = self.Session()
        try:
            # Clear existing data
            session.query(ContentVerification).delete()
            session.query(BusinessProfile).delete()
            session.query(PostedContent).delete()
            
            # Add sample businesses
            sample_businesses = [
                {
                    'business_id': 'EDU001',
                    'business_name': 'EduTech Academy',
                    'business_type': 'single_domain',
                    'domain': 'education',
                    'allowed_domains': ['education'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling']
                },
                {
                    'business_id': 'SPORTS001',
                    'business_name': 'Sports Gear Hub',
                    'business_type': 'single_domain',
                    'domain': 'sports',
                    'allowed_domains': ['sports', 'health'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling']
                },
                {
                    'business_id': 'MARKET001',
                    'business_name': 'MultiShop Marketplace',
                    'business_type': 'marketplace',
                    'domain': None,
                    'allowed_domains': ['education', 'sports', 'health', 'tech', 'fashion'],
                    'restricted_categories': ['weapons', 'drugs', 'adult_content', 'gambling', 'explosives']
                }
            ]
            
            for biz in sample_businesses:
                self.save_business_profile(biz)
            
            session.commit()
            print("✅ Sample data initialized successfully!")
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error initializing sample data: {e}")
        finally:
            session.close()