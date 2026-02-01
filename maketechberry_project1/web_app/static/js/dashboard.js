// Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    loadAnalytics();
    loadRecentVerifications();
    setupCharts();
    
    // Auto-refresh every 30 seconds
    setInterval(loadAnalytics, 30000);
    setInterval(loadRecentVerifications, 30000);
});

async function loadAnalytics() {
    try {
        const response = await fetch('/api/analytics?days=7');
        const data = await response.json();
        
        updateDashboardMetrics(data);
        updateCharts(data);
        
    } catch (error) {
        console.error('Error loading analytics:', error);
        showError('Failed to load analytics data');
    }
}

async function loadRecentVerifications() {
    try {
        const response = await fetch('/api/recent-verifications?limit=10');
        const data = await response.json();
        
        updateRecentVerificationsTable(data.verifications || data);
        
    } catch (error) {
        console.error('Error loading recent verifications:', error);
    }
}

function updateDashboardMetrics(data) {
    // Update metric cards
    document.getElementById('totalVerifications').textContent = data.total_verifications || 0;
    document.getElementById('approvedCount').textContent = data.approved_count || 0;
    document.getElementById('blockedCount').textContent = data.blocked_count || 0;
    document.getElementById('reviewCount').textContent = data.review_count || 0;
    document.getElementById('approvalRate').textContent = (data.approval_rate || 0).toFixed(1) + '%';
    document.getElementById('avgConfidence').textContent = (data.average_confidence * 100 || 0).toFixed(1) + '%';
}

function updateRecentVerificationsTable(verifications) {
    const tbody = document.getElementById('recentVerificationsBody');
    tbody.innerHTML = '';
    
    verifications.forEach(item => {
        const row = document.createElement('tr');
        
        // Status badge
        let badgeClass = 'badge ';
        if (item.decision === 'approved') badgeClass += 'bg-success';
        else if (item.decision === 'blocked') badgeClass += 'bg-danger';
        else badgeClass += 'bg-warning';
        
        row.innerHTML = `
            <td>${item.verification_id || 'N/A'}</td>
            <td>${item.business_id || 'N/A'}</td>
            <td>${item.title || 'No title'}</td>
            <td>${item.predicted_category || 'unknown'}</td>
            <td>${(item.confidence_score * 100).toFixed(1)}%</td>
            <td><span class="${badgeClass}">${item.decision}</span></td>
            <td>${new Date(item.created_at).toLocaleString()}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="viewVerification('${item.verification_id}')">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

function setupCharts() {
    // Initialize chart containers
    const ctx1 = document.getElementById('decisionChart')?.getContext('2d');
    const ctx2 = document.getElementById('categoryChart')?.getContext('2d');
    
    if (ctx1) {
        window.decisionChart = new Chart(ctx1, {
            type: 'doughnut',
            data: {
                labels: ['Approved', 'Blocked', 'Needs Review'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#28a745', '#dc3545', '#ffc107']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    if (ctx2) {
        window.categoryChart = new Chart(ctx2, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Verifications',
                    data: [],
                    backgroundColor: '#007bff'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

function updateCharts(data) {
    // Update decision chart
    if (window.decisionChart) {
        window.decisionChart.data.datasets[0].data = [
            data.approved_count || 0,
            data.blocked_count || 0,
            data.review_count || 0
        ];
        window.decisionChart.update();
    }
    
    // Update category chart
    if (window.categoryChart && data.category_distribution) {
        const categories = Object.keys(data.category_distribution);
        const counts = Object.values(data.category_distribution);
        
        // Sort by count (descending) and take top 10
        const sorted = categories.map((cat, idx) => ({cat, count: counts[idx]}))
            .sort((a, b) => b.count - a.count)
            .slice(0, 10);
        
        window.categoryChart.data.labels = sorted.map(item => item.cat);
        window.categoryChart.data.datasets[0].data = sorted.map(item => item.count);
        window.categoryChart.update();
    }
}

function viewVerification(verificationId) {
    // Navigate to verification details page
    window.location.href = `/verification/${verificationId}`;
}

function showError(message) {
    // Create error toast
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-danger border-0';
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-exclamation-circle me-2"></i>${message}
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

// Export filter function
async function exportData(format = 'csv') {
    try {
        const response = await fetch('/api/analytics/export?format=' + format);
        const blob = await response.blob();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `verification_analytics.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
    } catch (error) {
        console.error('Error exporting data:', error);
        showError('Failed to export data');
    }
}