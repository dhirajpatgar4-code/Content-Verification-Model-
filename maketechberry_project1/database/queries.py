from sqlalchemy import func, desc
from datetime import datetime, timedelta
from .database import DatabaseManager

def get_daily_verifications(day_count=7):
    """Get verification counts per day"""
    db = DatabaseManager()
    session = db.Session()
    
    try:
        start_date = datetime.utcnow() - timedelta(days=day_count)
        
        # Group by day
        results = session.query(
            func.date(db.ContentVerification.created_at).label('date'),
            func.count(db.ContentVerification.id).label('count')
        ).filter(
            db.ContentVerification.created_at >= start_date
        ).group_by(
            func.date(db.ContentVerification.created_at)
        ).order_by('date').all()
        
        return [{'date': r[0], 'count': r[1]} for r in results]
    finally:
        session.close()

def get_business_verification_stats(business_id):
    """Get verification statistics for a business"""
    db = DatabaseManager()
    session = db.Session()
    
    try:
        # Get counts by decision
        stats = session.query(
            db.ContentVerification.decision,
            func.count(db.ContentVerification.id).label('count')
        ).filter_by(
            business_id=business_id
        ).group_by(
            db.ContentVerification.decision
        ).all()
        
        # Get average confidence
        avg_confidence = session.query(
            func.avg(db.ContentVerification.confidence_score)
        ).filter_by(
            business_id=business_id
        ).scalar() or 0
        
        # Get domain match rate
        domain_match_rate = session.query(
            func.avg(db.ContentVerification.domain_verification_score)
        ).filter_by(
            business_id=business_id,
            expected_domain=session.query(db.BusinessProfile.domain).filter_by(
                business_id=business_id
            ).scalar_subquery()
        ).scalar() or 0
        
        return {
            'decision_stats': {s[0]: s[1] for s in stats},
            'average_confidence': float(avg_confidence),
            'domain_match_rate': float(domain_match_rate)
        }
    finally:
        session.close()

def get_category_accuracy():
    """Get accuracy per category"""
    db = DatabaseManager()
    session = db.Session()
    
    try:
        # For each category, calculate percentage of correct domain matches
        # This assumes we have ground truth data (which we don't in production)
        # This is a simplified version
        
        results = session.query(
            db.ContentVerification.predicted_category,
            func.count(db.ContentVerification.id).label('total'),
            func.avg(db.ContentVerification.domain_verification_score).label('accuracy')
        ).filter(
            db.ContentVerification.expected_domain.isnot(None)
        ).group_by(
            db.ContentVerification.predicted_category
        ).all()
        
        return [
            {
                'category': r[0],
                'total_verifications': r[1],
                'accuracy_score': float(r[2] or 0)
            }
            for r in results
        ]
    finally:
        session.close()

def search_verifications(search_term=None, business_id=None, 
                        start_date=None, end_date=None, 
                        decision=None, limit=50):
    """Search verifications with filters"""
    db = DatabaseManager()
    session = db.Session()
    
    try:
        query = session.query(db.ContentVerification)
        
        # Apply filters
        if search_term:
            query = query.filter(
                (db.ContentVerification.title.contains(search_term)) |
                (db.ContentVerification.description.contains(search_term))
            )
        
        if business_id:
            query = query.filter_by(business_id=business_id)
        
        if start_date:
            query = query.filter(db.ContentVerification.created_at >= start_date)
        
        if end_date:
            query = query.filter(db.ContentVerification.created_at <= end_date)
        
        if decision:
            query = query.filter_by(decision=decision)
        
        # Get results
        results = query.order_by(
            desc(db.ContentVerification.created_at)
        ).limit(limit).all()
        
        return [
            {
                'verification_id': v.verification_id,
                'business_id': v.business_id,
                'title': v.title,
                'predicted_category': v.predicted_category,
                'confidence_score': float(v.confidence_score),
                'decision': v.decision,
                'created_at': v.created_at.isoformat() if v.created_at else None,
                'requires_review': v.requires_human_review
            }
            for v in results
        ]
    finally:
        session.close()