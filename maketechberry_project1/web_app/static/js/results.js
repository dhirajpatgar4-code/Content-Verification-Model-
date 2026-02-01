// Results page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get result data from page
    const resultDataElement = document.getElementById('resultData');
    if (resultDataElement) {
        const resultData = JSON.parse(resultDataElement.textContent);
        displayResults(resultData);
    }
    
    // Setup copy buttons
    setupCopyButtons();
});

function displayResults(data) {
    // Update decision display
    const decision = data.decision;
    const resultDiv = document.getElementById('verificationResult');
    
    // Set result styling based on decision
    resultDiv.className = 'verification-result ';
    if (decision.decision === 'approved') {
        resultDiv.classList.add('approved');
        resultDiv.innerHTML = `
            <div class="text-center">
                <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                <h3 class="text-success">Content Approved!</h3>
                <p class="mb-0">${decision.reason}</p>
            </div>
        `;
    } else if (decision.decision === 'blocked') {
        resultDiv.classList.add('blocked');
        resultDiv.innerHTML = `
            <div class="text-center">
                <i class="fas fa-times-circle fa-3x text-danger mb-3"></i>
                <h3 class="text-danger">Content Blocked</h3>
                <p class="mb-0">${decision.reason}</p>
            </div>
        `;
    } else {
        resultDiv.classList.add('review');
        resultDiv.innerHTML = `
            <div class="text-center">
                <i class="fas fa-exclamation-triangle fa-3x text-warning mb-3"></i>
                <h3 class="text-warning">Review Required</h3>
                <p class="mb-0">${decision.reason}</p>
            </div>
        `;
    }
    
    // Update prediction details
    const prediction = data.prediction;
    document.getElementById('predictedCategory').textContent = prediction.category;
    document.getElementById('confidenceScore').textContent = (prediction.confidence * 100).toFixed(1) + '%';
    document.getElementById('modelUsed').textContent = prediction.model_used || 'Unknown';
    document.getElementById('isRestricted').textContent = prediction.is_restricted ? 'Yes' : 'No';
    
    // Update confidence bar
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = (prediction.confidence * 100) + '%';
    confidenceBar.textContent = (prediction.confidence * 100).toFixed(1) + '%';
    
    // Update confidence bar color
    if (prediction.confidence >= 0.8) {
        confidenceBar.classList.add('bg-success');
    } else if (prediction.confidence >= 0.5) {
        confidenceBar.classList.add('bg-warning');
    } else {
        confidenceBar.classList.add('bg-danger');
    }
    
    // Update top categories
    const topCategoriesList = document.getElementById('topCategories');
    topCategoriesList.innerHTML = '';
    
    if (prediction.top_categories && prediction.top_categories.length > 0) {
        prediction.top_categories.forEach((cat, index) => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.innerHTML = `
                ${cat.category}
                <span class="badge bg-primary rounded-pill">${(cat.confidence * 100).toFixed(1)}%</span>
            `;
            topCategoriesList.appendChild(li);
        });
    }
    
    // Update domain verification if available
    if (data.domain_verification) {
        const domainVerificationDiv = document.getElementById('domainVerification');
        domainVerificationDiv.style.display = 'block';
        
        const domainMatch = data.domain_verification.is_match;
        const domainScore = data.domain_verification.adjusted_score;
        
        document.getElementById('expectedDomain').textContent = data.domain_verification.expected_domain;
        document.getElementById('domainMatchResult').textContent = domainMatch ? 'Match' : 'Mismatch';
        document.getElementById('domainMatchResult').className = domainMatch ? 'text-success' : 'text-danger';
        document.getElementById('domainSimilarityScore').textContent = (domainScore * 100).toFixed(1) + '%';
        
        // Update domain match badge
        const domainBadge = document.getElementById('domainMatchBadge');
        domainBadge.textContent = domainMatch ? '✓ Domain Verified' : '✗ Domain Mismatch';
        domainBadge.className = domainMatch ? 'badge bg-success' : 'badge bg-danger';
    }
    
    // Update metadata
    document.getElementById('verificationId').textContent = data.verification_id || 'N/A';
    document.getElementById('verificationTime').textContent = new Date().toLocaleString();
    
    // Show/hide action buttons based on decision
    const postButton = document.getElementById('postContentBtn');
    const reviewButton = document.getElementById('requestReviewBtn');
    
    if (decision.decision === 'approved') {
        postButton.style.display = 'block';
        reviewButton.style.display = 'none';
    } else if (decision.decision === 'needs_review') {
        postButton.style.display = 'none';
        reviewButton.style.display = 'block';
    } else {
        postButton.style.display = 'none';
        reviewButton.style.display = 'none';
    }
}

function setupCopyButtons() {
    // Copy verification ID
    document.getElementById('copyVerificationId')?.addEventListener('click', function() {
        const verificationId = document.getElementById('verificationId').textContent;
        navigator.clipboard.writeText(verificationId).then(() => {
            showToast('Verification ID copied to clipboard');
        });
    });
    
    // Copy result summary
    document.getElementById('copyResultSummary')?.addEventListener('click', function() {
        const resultText = generateResultSummary();
        navigator.clipboard.writeText(resultText).then(() => {
            showToast('Result summary copied to clipboard');
        });
    });
}

function generateResultSummary() {
    const prediction = window.resultData?.prediction;
    const decision = window.resultData?.decision;
    
    if (!prediction || !decision) return '';
    
    return `Content Verification Result:
Verification ID: ${document.getElementById('verificationId').textContent}
Predicted Category: ${prediction.category}
Confidence: ${(prediction.confidence * 100).toFixed(1)}%
Decision: ${decision.decision}
Reason: ${decision.reason}
Severity: ${decision.severity}
Date: ${new Date().toLocaleString()}`;
}

function postContent() {
    const verificationId = document.getElementById('verificationId').textContent;
    
    if (!verificationId || verificationId === 'N/A') {
        showToast('No verification ID available', 'error');
        return;
    }
    
    // Show confirmation
    if (!confirm('Are you sure you want to post this content?')) {
        return;
    }
    
    // Send POST request to mark content as posted
    fetch(`/api/content/${verificationId}/post`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Content posted successfully!', 'success');
            
            // Update UI
            const postButton = document.getElementById('postContentBtn');
            postButton.innerHTML = '<i class="fas fa-check"></i> Posted';
            postButton.classList.remove('btn-primary');
            postButton.classList.add('btn-success');
            postButton.disabled = true;
            
            // Update verification status
            const statusBadge = document.getElementById('verificationStatus');
            statusBadge.textContent = 'Posted';
            statusBadge.className = 'badge bg-success';
            
        } else {
            showToast(data.error || 'Failed to post content', 'error');
        }
    })
    .catch(error => {
        console.error('Error posting content:', error);
        showToast('Error posting content', 'error');
    });
}

function requestReview() {
    const verificationId = document.getElementById('verificationId').textContent;
    
    if (!verificationId || verificationId === 'N/A') {
        showToast('No verification ID available', 'error');
        return;
    }
    
    // Send review request
    fetch(`/api/content/${verificationId}/review`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            review_reason: prompt('Please enter reason for review request:') || 'Manual review requested'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Review requested successfully!', 'success');
            
            // Update UI
            const reviewButton = document.getElementById('requestReviewBtn');
            reviewButton.innerHTML = '<i class="fas fa-check"></i> Review Requested';
            reviewButton.classList.remove('btn-warning');
            reviewButton.classList.add('btn-info');
            reviewButton.disabled = true;
            
        } else {
            showToast(data.error || 'Failed to request review', 'error');
        }
    })
    .catch(error => {
        console.error('Error requesting review:', error);
        showToast('Error requesting review', 'error');
    });
}

function downloadResult() {
    const resultText = generateResultSummary();
    const blob = new Blob([resultText], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `verification_result_${document.getElementById('verificationId').textContent}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showToast('Result downloaded', 'success');
}

function showToast(message, type = 'info') {
    // Create toast
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} border-0`;
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-${type === 'error' ? 'exclamation-circle' : type === 'success' ? 'check-circle' : 'info-circle'} me-2"></i>${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.querySelector('.toast-container').appendChild(toast);
    
    // Show toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove toast after hiding
    toast.addEventListener('hidden.bs.toast', function () {
        toast.remove();
    });
}