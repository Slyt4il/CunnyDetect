async function preprocessImage(img) {
    return new Promise((resolve) => {
        const image = new Image();
        image.onload = async () => {
            const tensor = tf.browser.fromPixels(image);
            const resizedTensor = tf.image.resizeBilinear(tensor, [256, 256]);
            const normalizedTensor = resizedTensor.div(255);
            tensor.dispose();
            resizedTensor.dispose();
            resolve(normalizedTensor);
        };
        image.src = img.src;
    });
}
