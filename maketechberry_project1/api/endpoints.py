from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import sys
sys.path.append('..')

from .models import TextContent, ImageContent, MixedContent, VerificationResponse
from .schemas import ContentVerification
from database.database import DatabaseManager
from inference.text_inference import TextInference
from inference.image_inference import ImageInference
from inference.multimodal_inference import MultimodalInference
from inference.decision_engine import DecisionEngine

app = FastAPI(title="Content Verification API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="web_app/templates")

# Initialize components
db = DatabaseManager()
text_inference = TextInference()
image_inference = ImageInference()
multimodal_inference = MultimodalInference()
decision_engine = DecisionEngine()

# Create upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve home page"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/verify", response_class=HTMLResponse)
async def verify_page():
    """Serve verification page"""
    return templates.TemplateResponse("verify.html", {"request": {}})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve dashboard page"""
    # Get analytics
    analytics = db.get_analytics()
    
    # Get recent verifications
    session = db.Session()
    recent = session.query(ContentVerification).order_by(
        ContentVerification.created_at.desc()
    ).limit(10).all()
    
    return templates.TemplateResponse(
        "dashboard.html", 
        {
            "request": {},
            "analytics": analytics,
            "recent_verifications": recent
        }
    )

@app.post("/api/verify/text", response_model=VerificationResponse)
async def verify_text_api(content: TextContent):
    """API endpoint for text verification"""
    try:
        start_time = datetime.now()
        
        # Get business profile
        business_profile = None
        if content.business_id:
            business_profile = db.get_business_profile(content.business_id)
        
        # Run text inference
        result = text_inference.verify(
            text=content.text,
            title=content.title,
            business_profile=business_profile
        )
        
        # Calculate verification time
        verification_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Save to database
        verification_id = f"TEXT_{uuid.uuid4().hex[:8]}"
        verification_data = {
            'verification_id': verification_id,
            'business_id': content.business_id or 'unknown',
            'content_type': 'text',
            'title': content.title,
            'description': content.text,
            'image_path': None,
            'prediction': result['prediction'],
            'decision': result['decision'],
            'expected_domain': business_profile.get('domain') if business_profile else None,
            'domain_match': result.get('domain_match', False),
            'domain_verification_score': result.get('domain_verification_score', 0.0),
            'verification_time_ms': verification_time_ms
        }
        
        db.save_verification_result(verification_data)
        
        return VerificationResponse(
            content_type="text",
            prediction=result['prediction'],
            decision=result['decision'],
            timestamp=datetime.now().isoformat(),
            verification_id=verification_id,
            domain_verification=result.get('domain_verification')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify/image", response_model=VerificationResponse)
async def verify_image_api(
    image: UploadFile = File(...),
    business_id: Optional[str] = Form(None),
    expected_domain: Optional[str] = Form(None)
):
    """API endpoint for image verification"""
    try:
        start_time = datetime.now()
        
        # Save uploaded image
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        filename = f"{uuid.uuid4().hex}.{file_extension}"
        image_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Get business profile
        business_profile = None
        if business_id:
            business_profile = db.get_business_profile(business_id)
            if business_profile and not expected_domain:
                expected_domain = business_profile.get('domain')
        
        # Run image inference
        result = image_inference.verify(
            image_path=image_path,
            business_profile=business_profile,
            expected_domain=expected_domain
        )
        
        # Calculate verification time
        verification_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Save to database
        verification_id = f"IMG_{uuid.uuid4().hex[:8]}"
        verification_data = {
            'verification_id': verification_id,
            'business_id': business_id or 'unknown',
            'content_type': 'image',
            'title': None,
            'description': None,
            'image_path': image_path,
            'prediction': result['prediction'],
            'decision': result['decision'],
            'expected_domain': expected_domain,
            'domain_match': result.get('domain_match', False),
            'domain_verification_score': result.get('domain_verification_score', 0.0),
            'verification_time_ms': verification_time_ms
        }
        
        db.save_verification_result(verification_data)
        
        return VerificationResponse(
            content_type="image",
            prediction=result['prediction'],
            decision=result['decision'],
            timestamp=datetime.now().isoformat(),
            verification_id=verification_id,
            domain_verification=result.get('domain_verification')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify/mixed", response_model=VerificationResponse)
async def verify_mixed_api(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    business_id: Optional[str] = Form(None)
):
    """API endpoint for mixed content verification"""
    try:
        start_time = datetime.now()
        
        # Handle image upload
        image_path = None
        if image:
            file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
            filename = f"{uuid.uuid4().hex}.{file_extension}"
            image_path = os.path.join(UPLOAD_DIR, filename)
            
            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)
        
        # Get business profile
        business_profile = None
        if business_id:
            business_profile = db.get_business_profile(business_id)
        
        # Run multimodal inference
        result = multimodal_inference.verify(
            text=text,
            title=title,
            image_path=image_path,
            business_profile=business_profile
        )
        
        # Calculate verification time
        verification_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Save to database
        verification_id = f"MIX_{uuid.uuid4().hex[:8]}"
        verification_data = {
            'verification_id': verification_id,
            'business_id': business_id or 'unknown',
            'content_type': 'mixed',
            'title': title,
            'description': text,
            'image_path': image_path,
            'prediction': result['prediction'],
            'decision': result['decision'],
            'expected_domain': business_profile.get('domain') if business_profile else None,
            'domain_match': result.get('domain_match', False),
            'domain_verification_score': result.get('domain_verification_score', 0.0),
            'verification_time_ms': verification_time_ms
        }
        
        db.save_verification_result(verification_data)
        
        return VerificationResponse(
            content_type="mixed",
            prediction=result['prediction'],
            decision=result['decision'],
            timestamp=datetime.now().isoformat(),
            verification_id=verification_id,
            domain_verification=result.get('domain_verification')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recent-verifications")
async def get_recent_verifications(limit: int = 10):
    """Get recent verifications"""
    try:
        session = db.Session()
        recent = session.query(ContentVerification).order_by(
            ContentVerification.created_at.desc()
        ).limit(limit).all()
        
        result = []
        for item in recent:
            result.append({
                'verification_id': item.verification_id,
                'business_id': item.business_id,
                'title': item.title,
                'predicted_category': item.predicted_category,
                'confidence_score': float(item.confidence_score),
                'decision': item.decision,
                'created_at': item.created_at.isoformat() if item.created_at else None
            })
        
        return {"verifications": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def get_analytics(days: int = 30, business_id: Optional[str] = None):
    """Get analytics data"""
    try:
        analytics = db.get_analytics(business_id=business_id, days=days)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/content/{verification_id}/post")
async def post_content(verification_id: str, background_tasks: BackgroundTasks):
    """Mark content as posted"""
    try:
        # Get verification record
        session = db.Session()
        verification = session.query(ContentVerification).filter_by(
            verification_id=verification_id
        ).first()
        
        if not verification:
            raise HTTPException(status_code=404, detail="Verification not found")
        
        if verification.decision != 'approved':
            raise HTTPException(status_code=400, detail="Content not approved for posting")
        
        # Generate content ID
        content_id = f"POST_{uuid.uuid4().hex[:8]}"
        
        # Mark as posted in background
        def mark_as_posted():
            db.save_posted_content(
                verification_id=verification_id,
                business_id=verification.business_id,
                content_id=content_id,
                platform="api"
            )
        
        background_tasks.add_task(mark_as_posted)
        
        return {
            "success": True,
            "message": "Content marked for posting",
            "content_id": content_id,
            "verification_id": verification_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",
            "ml_models": "loaded",
            "api": "running"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)