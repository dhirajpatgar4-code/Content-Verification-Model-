// Verification page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    setupFormHandlers();
    loadBusinessProfiles();
    setupImagePreview();
});

function setupFormHandlers() {
    const form = document.getElementById('verificationForm');
    const contentTypeSelect = document.getElementById('content_type');
    
    // Show/hide form sections based on content type
    contentTypeSelect.addEventListener('change', function() {
        const type = this.value;
        
        // Show/hide text section
        const textSection = document.getElementById('textSection');
        if (textSection) {
            textSection.style.display = (type === 'text' || type === 'mixed') ? 'block' : 'none';
        }
        
        // Show/hide image section
        const imageSection = document.getElementById('imageSection');
        if (imageSection) {
            imageSection.style.display = (type === 'image' || type === 'mixed') ? 'block' : 'none';
        }
    });
    
    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying...';
        submitBtn.disabled = true;
        
        try {
            const formData = new FormData(form);
            const contentType = formData.get('content_type');
            
            let endpoint = '/api/verify/';
            let body;
            
            if (contentType === 'text') {
                endpoint += 'text';
                body = {
                    text: formData.get('description'),
                    title: formData.get('title'),
                    business_id: formData.get('business_id')
                };
                
                // Send as JSON
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(body)
                });
                
                await handleResponse(response);
                
            } else if (contentType === 'image') {
                endpoint += 'image';
                
                // Send as FormData
                const imageResponse = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                
                await handleResponse(imageResponse);
                
            } else if (contentType === 'mixed') {
                endpoint += 'mixed';
                
                // Send as FormData
                const mixedResponse = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });
                
                await handleResponse(mixedResponse);
            }
            
        } catch (error) {
            console.error('Error:', error);
            showError('An error occurred during verification');
        } finally {
            // Reset button
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
    });
}

async function handleResponse(response) {
    if (response.ok) {
        const result = await response.json();
        
        // Store result in sessionStorage for results page
        sessionStorage.setItem('verificationResult', JSON.stringify(result));
        
        // Redirect to results page
        window.location.href = '/results';
        
    } else {
        const error = await response.json();
        showError(error.detail || 'Verification failed');
    }
}

async function loadBusinessProfiles() {
    try {
        // In a real implementation, this would fetch from an API
        // For now, use sample data
        const sampleBusinesses = [
            { id: 'EDU001', name: 'EduTech Academy', domain: 'education' },
            { id: 'SPORTS001', name: 'Sports Gear Hub', domain: 'sports' },
            { id: 'MARKET001', name: 'MultiShop Marketplace', domain: 'marketplace' }
        ];
        
        const businessSelect = document.getElementById('business_id');
        if (businessSelect) {
            // Clear existing options except the first one
            while (businessSelect.options.length > 1) {
                businessSelect.remove(1);
            }
            
            // Add sample businesses
            sampleBusinesses.forEach(business => {
                const option = document.createElement('option');
                option.value = business.id;
                option.textContent = `${business.name} (${business.id})`;
                businessSelect.appendChild(option);
            });
            
            // Add event listener to auto-fill expected domain
            businessSelect.addEventListener('change', function() {
                const selectedBusiness = sampleBusinesses.find(b => b.id === this.value);
                const domainInput = document.getElementById('expected_domain');
                if (domainInput && selectedBusiness) {
                    domainInput.value = selectedBusiness.domain;
                }
            });
        }
        
    } catch (error) {
        console.error('Error loading business profiles:', error);
    }
}

function setupImagePreview() {
    const imageInput = document.getElementById('image');
    const previewContainer = document.getElementById('imagePreview');
    
    if (imageInput && previewContainer) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            
            if (file) {
                // Validate file type
                if (!file.type.match('image.*')) {
                    showError('Please select an image file');
                    this.value = '';
                    previewContainer.innerHTML = '';
                    return;
                }
                
                // Validate file size (max 10MB)
                if (file.size > 10 * 1024 * 1024) {
                    showError('Image size must be less than 10MB');
                    this.value = '';
                    previewContainer.innerHTML = '';
                    return;
                }
                
                // Create preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewContainer.innerHTML = `
                        <div class="image-preview mt-2">
                            <img src="${e.target.result}" 
                                 class="img-thumbnail" 
                                 style="max-height: 200px; max-width: 100%;">
                            <div class="mt-1 small text-muted">
                                ${file.name} (${(file.size / 1024).toFixed(1)} KB)
                            </div>
                        </div>
                    `;
                };
                reader.readAsDataURL(file);
                
            } else {
                previewContainer.innerHTML = '';
            }
        });
    }
}

function showError(message) {
    // Create error alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-circle me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert after form title or at the top of form
    const form = document.getElementById('verificationForm');
    form.parentNode.insertBefore(alertDiv, form);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }
    }, 5000);
}

function showSuccess(message) {
    // Create success alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show';
    alertDiv.innerHTML = `
        <i class="fas fa-check-circle me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const form = document.getElementById('verificationForm');
    form.parentNode.insertBefore(alertDiv, form);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }
    }, 5000);
}

// Quick verification from home page
function quickVerify(contentType) {
    const form = document.getElementById('verificationForm');
    const contentTypeSelect = document.getElementById('content_type');
    
    if (contentTypeSelect) {
        contentTypeSelect.value = contentType;
        contentTypeSelect.dispatchEvent(new Event('change'));
    }
    
    // Scroll to form
    form.scrollIntoView({ behavior: 'smooth' });
}