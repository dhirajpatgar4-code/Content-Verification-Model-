// Verification form script
document.getElementById('verificationForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const textInput = document.getElementById('textInput').value;
    const imageInput = document.getElementById('imageInput').files[0];
    
    if (!textInput && !imageInput) {
        alert('Please provide text or image content');
        return;
    }
    
    const formData = new FormData();
    if (textInput) formData.append('text', textInput);
    if (imageInput) formData.append('image', imageInput);
    
    try {
        const response = await fetch('/api/verify', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = '<p>Error: ' + error.message + '</p>';
    }
});

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="result-card">
            <h3>Verification Result</h3>
            <p>Status: ${data.verified ? 'Verified' : 'Not Verified'}</p>
        </div>
    `;
}
