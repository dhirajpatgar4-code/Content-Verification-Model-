from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import uuid
import json
from datetime import datetime
import sys
import base64
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import DatabaseManager
from inference.text_inference import TextInference
from inference.image_inference import ImageInference
from inference.multimodal_inference import MultimodalInference
from inference.decision_engine import DecisionEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
db = DatabaseManager()
text_inference = TextInference()
image_inference = ImageInference()
multimodal_inference = MultimodalInference()
decision_engine = DecisionEngine()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/verify', methods=['GET', 'POST'])
def verify_content():
    """Content verification page"""
    if request.method == 'POST':
        # Get form data
        business_id = request.form.get('business_id')
        content_type = request.form.get('content_type')
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        
        # Get business profile
        business_profile = db.get_business_profile(business_id)
        if not business_profile:
            return render_template('error.html', 
                                 error="Business profile not found. Please register first in Problem Statement 4.")
        
        verification_id = f"VER_{uuid.uuid4().hex[:8]}"
        
        # Handle image upload
        image_path = None
        if content_type in ['image', 'mixed'] and 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                filename = f"{verification_id}_{image_file.filename}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
        
        # Perform verification based on content type
        start_time = datetime.now()
        
        if content_type == 'text':
            result = text_inference.verify(
                text=description,
                title=title,
                business_profile=business_profile
            )
        elif content_type == 'image':
            result = image_inference.verify(
                image_path=image_path,
                business_profile=business_profile
            )
        else:  # mixed
            result = multimodal_inference.verify(
                text=description,
                title=title,
                image_path=image_path,
                business_profile=business_profile
            )
        
        # Calculate verification time
        verification_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Prepare verification data for database
        verification_data = {
            'verification_id': verification_id,
            'business_id': business_id,
            'content_type': content_type,
            'title': title,
            'description': description,
            'image_path': image_path,
            'prediction': result['prediction'],
            'decision': result['decision'],
            'expected_domain': business_profile.get('domain'),
            'domain_match': result.get('domain_match', False),
            'domain_verification_score': result.get('domain_verification_score', 0.0),
            'verification_time_ms': int(verification_time)
        }
        
        # Save to database
        db.save_verification_result(verification_data)
        
        # Render result page
        return render_template('results.html', 
                             result=result,
                             verification_id=verification_id,
                             business_profile=business_profile)
    
    return render_template('verify.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    # Get analytics
    analytics = db.get_analytics()
    
    # Get recent verifications
    session = db.Session()
    recent = session.query(db.ContentVerification).order_by(
        db.ContentVerification.created_at.desc()
    ).limit(10).all()
    
    # Prepare data for template
    recent_data = []
    for item in recent:
        recent_data.append({
            'verification_id': item.verification_id,
            'business_id': item.business_id,
            'title': item.title,
            'predicted_category': item.predicted_category,
            'confidence_score': item.confidence_score,
            'decision': item.decision,
            'created_at': item.created_at
        })
    
    return render_template('dashboard.html',
                         analytics=analytics,
                         recent_verifications=recent_data)

@app.route('/analytics')
def analytics():
    """Analytics page"""
    # Get detailed analytics
    analytics_data = db.get_analytics(days=30)
    
    # Get category distribution
    session = db.Session()
    category_counts = session.query(
        db.ContentVerification.predicted_category,
        db.func.count(db.ContentVerification.id)
    ).group_by(db.ContentVerification.predicted_category).all()
    
    # Get decision distribution
    decision_counts = session.query(
        db.ContentVerification.decision,
        db.func.count(db.ContentVerification.id)
    ).group_by(db.ContentVerification.decision).all()
    
    # Get domain verification success rate
    domain_success = session.query(
        db.ContentVerification.expected_domain,
        db.func.avg(db.ContentVerification.domain_verification_score)
    ).filter(db.ContentVerification.expected_domain.isnot(None)).group_by(
        db.ContentVerification.expected_domain
    ).all()
    
    return render_template('analytics.html',
                         analytics=analytics_data,
                         category_counts=category_counts,
                         decision_counts=decision_counts,
                         domain_success=domain_success)

@app.route('/api/recent-verifications')
def api_recent_verifications():
    """API endpoint for recent verifications"""
    session = db.Session()
    recent = session.query(db.ContentVerification).order_by(
        db.ContentVerification.created_at.desc()
    ).limit(6).all()
    
    result = []
    for item in recent:
        result.append({
            'verification_id': item.verification_id,
            'business_id': item.business_id,
            'title': item.title,
            'predicted_category': item.predicted_category,
            'confidence_score': float(item.confidence_score),
            'decision': item.decision,
            'created_at': item.created_at.isoformat()
        })
    
    return jsonify(result)

@app.route('/api/verify', methods=['POST'])
def api_verify():
    """API endpoint for programmatic verification"""
    try:
        data = request.json
        
        # Validate input
        if not data.get('business_id'):
            return jsonify({'error': 'business_id is required'}), 400
        
        if not data.get('content_type') in ['text', 'image', 'mixed']:
            return jsonify({'error': 'Invalid content_type'}), 400
        
        # Get business profile
        business_profile = db.get_business_profile(data['business_id'])
        if not business_profile:
            return jsonify({'error': 'Business not found'}), 404
        
        verification_id = f"API_{uuid.uuid4().hex[:8]}"
        
        # Handle image if provided
        image_path = None
        if data.get('content_type') in ['image', 'mixed'] and data.get('image_base64'):
            # Decode base64 image
            try:
                image_data = base64.b64decode(data['image_base64'])
                image = Image.open(BytesIO(image_data))
                filename = f"{verification_id}.jpg"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path, 'JPEG')
            except Exception as e:
                return jsonify({'error': f'Invalid image: {str(e)}'}), 400
        
        # Perform verification
        start_time = datetime.now()
        
        if data['content_type'] == 'text':
            result = text_inference.verify(
                text=data.get('text', ''),
                title=data.get('title', ''),
                business_profile=business_profile
            )
        elif data['content_type'] == 'image':
            result = image_inference.verify(
                image_path=image_path,
                business_profile=business_profile
            )
        else:  # mixed
            result = multimodal_inference.verify(
                text=data.get('text', ''),
                title=data.get('title', ''),
                image_path=image_path,
                business_profile=business_profile
            )
        
        verification_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Save to database
        verification_data = {
            'verification_id': verification_id,
            'business_id': data['business_id'],
            'content_type': data['content_type'],
            'title': data.get('title', ''),
            'description': data.get('text', ''),
            'image_path': image_path,
            'prediction': result['prediction'],
            'decision': result['decision'],
            'expected_domain': business_profile.get('domain'),
            'domain_match': result.get('domain_match', False),
            'domain_verification_score': result.get('domain_verification_score', 0.0),
            'verification_time_ms': int(verification_time)
        }
        
        db.save_verification_result(verification_data)
        
        # Return result
        return jsonify({
            'verification_id': verification_id,
            'result': result,
            'business_profile': business_profile,
            'verification_time_ms': int(verification_time)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/content/<verification_id>/post', methods=['POST'])
def api_post_content(verification_id):
    """Mark content as posted"""
    try:
        data = request.json
        content_id = data.get('content_id', f'POST_{uuid.uuid4().hex[:8]}')
        platform = data.get('platform', 'api')
        
        # Get verification record
        session = db.Session()
        verification = session.query(db.ContentVerification).filter_by(
            verification_id=verification_id
        ).first()
        
        if not verification:
            return jsonify({'error': 'Verification not found'}), 404
        
        if verification.decision != 'approved':
            return jsonify({'error': 'Content not approved for posting'}), 400
        
        # Mark as posted
        db.save_posted_content(
            verification_id=verification_id,
            business_id=verification.business_id,
            content_id=content_id,
            platform=platform
        )
        
        return jsonify({
            'success': True,
            'message': 'Content posted successfully',
            'content_id': content_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load ML models
    print("🔄 Loading ML models...")
    text_inference.load_model()
    image_inference.load_model()
    print("✅ Models loaded successfully!")
    
    # Start web server
    app.run(debug=True, host='0.0.0.0', port=5000)