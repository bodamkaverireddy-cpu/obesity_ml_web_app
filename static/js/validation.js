// Form validation utilities

function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function validatePassword(password) {
    return password.length >= 6;
}

function validateFile(file, allowedExtensions) {
    if (!file) return false;
    const extension = file.name.split('.').pop().toLowerCase();
    return allowedExtensions.includes(extension);
}

function showError(element, message) {
    element.style.borderColor = '#dc3545';
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.style.color = '#dc3545';
    errorDiv.textContent = message;
    element.parentNode.appendChild(errorDiv);
}

function clearError(element) {
    element.style.borderColor = '#ddd';
    const errorDiv = element.parentNode.querySelector('.error-message');
    if (errorDiv) {
        errorDiv.remove();
    }
}

