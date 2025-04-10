// DOM Elements
const tryModelBtn = document.getElementById('tryModelBtn');
const modelModal = document.getElementById('modelModal');
const closeBtn = document.querySelector('.close-btn');
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const submitBtn = document.getElementById('submitBtn');
const resultBox = document.getElementById('resultBox');

// === Animate Feature Cards on Scroll ===
document.addEventListener('DOMContentLoaded', () => {
    const featureCards = document.querySelectorAll('.feature-card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    featureCards.forEach(card => {
        card.style.opacity = 0;
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
});

// === Modal Logic ===
tryModelBtn.addEventListener('click', () => {
    modelModal.style.display = 'block';
    document.body.style.overflow = 'hidden';
});

closeBtn.addEventListener('click', () => {
    modelModal.style.display = 'none';
    document.body.style.overflow = 'auto';
});

window.addEventListener('click', (e) => {
    if (e.target === modelModal) {
        modelModal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
});

// === Image Upload Preview ===
imageUpload.addEventListener('change', () => {
    const file = imageUpload.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded fabric">`;
        };
        reader.readAsDataURL(file);
    }
});

// === Analyze Button Click ===
submitBtn.addEventListener('click', () => {
    if (!imageUpload.files[0]) {
        const uploadLabel = document.querySelector('.upload-label');
        uploadLabel.classList.add('shake');
        setTimeout(() => uploadLabel.classList.remove('shake'), 500);
        return;
    }

    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    submitBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', imageUpload.files[0]);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            resultBox.innerHTML = `<p style="color: red;">${data.error}</p>`;
            resultBox.style.display = 'block';
            return;
        }

        resultBox.innerHTML = `
            <h3>Analysis Results</h3>
            <div class="result-item"><span class="result-label">Cloth Material:</span><span class="result-value">${data.fabric_type}</span></div>
            <div class="result-item"><span class="result-label">Defect Detected:</span><span class="result-value defect">${data.defect_type}</span></div>
            <div class="result-item"><span class="result-label">No. of Defect Detected:</span><span class="result-value defect">${data.defect_count}</span></div>
            <div class="result-item"><span class="result-label">Shape:</span><span class="result-value">${data.shape}</span></div>
            <div class="result-item"><span class="result-label">Area:</span><span class="result-value">${data.area} px²</span></div>
            <div class="result-item"><span class="result-label">Severity:</span><span class="result-value">${data.severity} ${data.emoji}</span></div>
            <div class="result-item"><span class="result-label">Original Image:</span><br><img src="${data.original_image}" height="150"></div>
            <div class="result-item"><span class="result-label">Analyzed Image:</span><br><img src="${data.analyzed_image}" height="150"></div>
        `;

        resultBox.style.display = 'block';
        resultBox.scrollIntoView({ behavior: 'smooth' });
    })
    .catch(err => {
        alert("Error analyzing image. Please try again.");
        console.error(err);
    })
    .finally(() => {
        submitBtn.innerHTML = 'Submit for Analysis';
        submitBtn.disabled = false;
    });
});

// === Shake Animation ===
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20%, 60% { transform: translateX(-5px); }
        40%, 80% { transform: translateX(5px); }
    }

    .shake {
        animation: shake 0.5s ease;
    }

    .highlight {
        background-color: rgba(215, 108, 130, 0.2);
        border-color: #B03052;
    }
`;
document.head.appendChild(style);

// === Drag and Drop Upload ===
const uploadLabel = document.querySelector('.upload-label');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, () => uploadLabel.classList.add('highlight'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadLabel.addEventListener(eventName, () => uploadLabel.classList.remove('highlight'), false);
});

uploadLabel.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) {
        imageUpload.files = files;
        const event = new Event('change');
        imageUpload.dispatchEvent(event);
    }
}

// === Hero Text Animation ===
document.querySelectorAll('.hero-content h2, .hero-content p').forEach((el, index) => {
    el.style.opacity = 0;
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    el.style.transitionDelay = `${0.2 + (index * 0.2)}s`;

    setTimeout(() => {
        el.style.opacity = 1;
        el.style.transform = 'translateY(0)';
    }, 100);
});
