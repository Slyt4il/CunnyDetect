async function predict(tensor) {
    if (!model) {
        return;
    }
    const prediction = model.predict(tensor);
    const prediction_arr = prediction.arraySync();
    
    console.log
    return prediction_arr;
}

async function interpret(predictionArray) {
    if (predictionArray < 0.5) {
        return "This image does not contain cunny. 😇";
    } else {
        return "This image contains cunny. 😭";
    }
}