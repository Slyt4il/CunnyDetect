async function main() {
    const imageInput = document.getElementById('imageInput');
    const imageElement = document.getElementById('imageElement');
    const predictButton = document.getElementById('predictButton');

    predictButton.addEventListener('click', async () => {
        const selectedFile = imageInput.files[0];
        if (selectedFile) {
            imageElement.src = URL.createObjectURL(selectedFile);
            const normalizedTensor = await preprocessImage(imageElement);
            const result = await predict(normalizedTensor);

            predictionResult.textContent = await interpret(result);
        }
    });
}

window.addEventListener('DOMContentLoaded', () => {
    main();
    loadModel();
});