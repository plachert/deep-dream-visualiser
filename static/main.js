// Retrieve the input elements and canvas
const originalImageInput = document.getElementById('original-image');
const processedImageCanvas = document.getElementById('processed-image');
const processButton = document.getElementById('process-button');

// Handle the image selection
originalImageInput.addEventListener('change', () => {
    const file = originalImageInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            // Display the selected image on the canvas
            const img = new Image();
            img.src = e.target.result;
            img.onload = () => {
                processedImageCanvas.width = img.width;
                processedImageCanvas.height = img.height;
                const context = processedImageCanvas.getContext('2d');
                context.drawImage(img, 0, 0);
            };
        };
        reader.readAsDataURL(file);
    }
});

// Add event listener to process the image
processButton.addEventListener('click', () => {
    // Retrieve the processed image using AJAX or Fetch API
    const canvasData = processedImageCanvas.toDataURL('image/jpeg');
    const formData = new FormData();
    formData.append('image', canvasData);

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        // Create a new image element and display the processed image
        const img = new Image();
        img.src = URL.createObjectURL(blob);
        img.onload = () => {
            processedImageCanvas.width = img.width;
            processedImageCanvas.height = img.height;
            const context = processedImageCanvas.getContext('2d');
            context.drawImage(img, 0, 0);
        };
    })
    .catch(error => console.error(error));
});
