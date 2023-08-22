async function preprocessImage(img) {
    return new Promise((resolve) => {
        const image = new Image();
        image.onload = async () => {
            const tensor = tf.browser.fromPixels(image);
            resolve(tensor.expandDims(0).toFloat());
            tensor.dispose();
        };
        image.src = img.src;
    });
}
