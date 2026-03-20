// API endpoints
const API_BASE = 'http://localhost:5001';

// DOM elements
const instructionInput = document.getElementById('instruction');
const inputTextarea = document.getElementById('input');
const maxTokensInput = document.getElementById('maxTokens');
const temperatureSlider = document.getElementById('temperature');
const tempValue = document.getElementById('tempValue');
const predictBtn = document.getElementById('predictBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultDiv = document.getElementById('result');
const metadataDiv = document.getElementById('metadata');
const modelBadge = document.getElementById('modelBadge');
const exampleButtons = document.getElementById('exampleButtons');

// Temperature slider
temperatureSlider.addEventListener('input', function() {
    tempValue.textContent = this.value;
});

// Load examples on page load
async function loadExamples() {
    try {
        const response = await fetch(`${API_BASE}/examples`);
        const examples = await response.json();
        
        exampleButtons.innerHTML = examples.map(example => `
            <button class="example-btn" onclick="loadExample('${example.instruction.replace(/'/g, "\\'")}', '${example.input.replace(/'/g, "\\'")}')">
                ${example.description}
            </button>
        `).join('');
    } catch (error) {
        console.error('Failed to load examples:', error);
    }
}

// Load example
function loadExample(instruction, input) {
    instructionInput.value = instruction;
    inputTextarea.value = input;
}

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/info`);
        const data = await response.json();
        
        const badgeHtml = `
            <span class="badge">
                🤖 ${data.model} | 
                ${data.fine_tuned ? '✅ Fine-tuned by JeffreyWoo' : '⚠️ Base Model'} | 
                💻 ${data.device.toUpperCase()}
            </span>
        `;
        modelBadge.innerHTML = badgeHtml;
    } catch (error) {
        console.error('Failed to check model status:', error);
        modelBadge.innerHTML = '<span class="badge">⚠️ Model not available</span>';
    }
}

// Make prediction
async function makePrediction() {
    const instruction = instructionInput.value.trim();
    const input = inputTextarea.value.trim();
    const maxTokens = parseInt(maxTokensInput.value);
    const temperature = parseFloat(temperatureSlider.value);
    
    if (!instruction) {
        alert('Please enter an analysis instruction');
        return;
    }
    
    // Show loading
    predictBtn.disabled = true;
    loadingIndicator.style.display = 'block';
    resultDiv.innerHTML = '';
    metadataDiv.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                instruction: instruction,
                input: input,
                max_tokens: maxTokens,
                temperature: temperature
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            resultDiv.innerHTML = `<div class="placeholder" style="color: #e53e3e;">Error: ${data.error}</div>`;
        } else {
            // Format response
            const formattedResponse = data.response
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
            
            resultDiv.innerHTML = `
                <div class="result-content">
                    ${formattedResponse}
                </div>
            `;
            
            metadataDiv.innerHTML = `
                <strong>📊 Analysis Details</strong><br>
                • Inference Time: ${data.metadata.inference_time_ms} ms<br>
                • Device: ${data.metadata.device.toUpperCase()}<br>
                • Model: ${data.metadata.model}<br>
                • Developer: JeffreyWoo<br>
                • Timestamp: ${data.metadata.timestamp}<br>
                • Tokens: ${maxTokens} | Temperature: ${temperature}
            `;
            metadataDiv.style.display = 'block';
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        resultDiv.innerHTML = `<div class="placeholder" style="color: #e53e3e;">Error: ${error.message}</div>`;
    } finally {
        predictBtn.disabled = false;
        loadingIndicator.style.display = 'none';
    }
}

// Event listeners
predictBtn.addEventListener('click', makePrediction);

// Enter key support
inputTextarea.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        makePrediction();
    }
});

// Initialize
loadExamples();
checkModelStatus();

// Refresh model status every 30 seconds
setInterval(checkModelStatus, 30000);