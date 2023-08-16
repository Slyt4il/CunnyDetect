let model;

async function loadModel() {
    if (model) {
        return;
    }
    model = await tf.loadLayersModel("js_model/model.json");
}

async function predict(tensor) {
    if (!model) {
        return;
    }
    const reshapedTensor = tensor.reshape([1, 256, 256, 3]);
    const prediction = model.predict(reshapedTensor);
    const prediction_arr = prediction.arraySync();
    
    console.log
    return prediction_arr;
}

async function interpret(predictionArray) {
    if (predictionArray < 0.475) {
        return "This image does not contain cunny. 😇";
    } else if (predictionArray < 0.5){
        return "This image might not contain cunny. 🤔";
    }
    else if (predictionArray < 0.525){
        return "This image could contain cunny. 🤔";
    }
    else {
        return "This image contains cunny. 😭";
    }
}