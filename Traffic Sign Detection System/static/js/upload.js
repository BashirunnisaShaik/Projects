/**
 * Upload Module JavaScript
 * Handles drag & drop, file selection, client-side validation,
 * image preview, and animated processing overlay.
 */

document.addEventListener('DOMContentLoaded', function () {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('image-upload-input');
  const previewCard = document.getElementById('previewCard');
  const previewImg = document.getElementById('previewImg');
  const previewName = document.getElementById('previewName');
  const previewSize = document.getElementById('previewSize');
  const previewDimensions = document.getElementById('previewDimensions');
  const clearBtn = document.getElementById('clearPreviewBtn');
  const uploadForm = document.getElementById('detectionUploadForm');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingStatusText = document.getElementById('loadingStatusText');

  if (!dropzone || !fileInput) return;

  // 1. Click to trigger file input
  dropzone.addEventListener('click', function (e) {
    if (e.target.tagName !== 'BUTTON' && !e.target.closest('button')) {
      fileInput.click();
    }
  });

  // 2. Drag & Drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', function (e) {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  // 3. Input change event
  fileInput.addEventListener('change', function () {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  // 4. File processing & preview
  function handleFile(file) {
    // Validate format
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      alert('Invalid file format. Please upload a JPG, JPEG, or PNG image.');
      fileInput.value = '';
      return;
    }

    // Validate size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size exceeds 10MB. Please select a smaller image.');
      fileInput.value = '';
      return;
    }

    // Display metadata
    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);

    // Read and render image
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImg.src = e.target.result;
      previewCard.style.display = 'block';

      // Read natural image dimensions
      const img = new Image();
      img.onload = function () {
        previewDimensions.textContent = `${img.naturalWidth} x ${img.naturalHeight} px`;
      };
      img.src = e.target.result;

      // Scroll smoothly to preview
      previewCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    };
    reader.readAsDataURL(file);
  }

  // 5. Clear preview button
  if (clearBtn) {
    clearBtn.addEventListener('click', function () {
      fileInput.value = '';
      previewCard.style.display = 'none';
      previewImg.src = '';
    });
  }

  // 6. Form submission loading overlay & duplicate submission prevention
  const startDetectBtn = document.getElementById('startDetectBtn');
  if (uploadForm) {
    uploadForm.addEventListener('submit', function (e) {
      if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select or drop an image file before analyzing.');
        e.preventDefault();
        return;
      }

      // Prevent duplicate submissions
      if (startDetectBtn) {
        startDetectBtn.disabled = true;
        startDetectBtn.innerHTML = `
          <svg class="spinner-inline" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 0.8s linear infinite; margin-right: 6px;">
            <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="12"></circle>
          </svg>
          Processing...
        `;
      }

      // Show loader
      if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
        simulateLoadingSteps();
      }
    });
  }

  // 7. Instant Test Samples click handling
  const sampleForms = document.querySelectorAll('.sample-submit-form');
  sampleForms.forEach(form => {
    form.addEventListener('submit', function () {
      const btn = form.querySelector('button');
      if (btn) btn.disabled = true;
      if (loadingOverlay) {
        loadingOverlay.style.display = 'flex';
        simulateLoadingSteps();
      }
    });
  });

  // Helper: Format bytes to human readable string
  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Simulated status message progression for academic presentation
  function simulateLoadingSteps() {
    if (!loadingStatusText) return;
    const steps = [
      'Reading uploaded image...',
      'Applying Gaussian noise reduction...',
      'Segmenting HSV color spaces (Red, Blue, Yellow)...',
      'Executing Canny edge and contour detection...',
      'Analyzing shape geometry and polygon approximation...',
      'Recognizing traffic sign classification...'
    ];
    let stepIndex = 0;
    setInterval(() => {
      if (stepIndex < steps.length) {
        loadingStatusText.textContent = steps[stepIndex];
        stepIndex++;
      }
    }, 450);
  }
});
