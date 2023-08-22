let model;

async function loadModel() {
    if (model) {
        return;
    }
    model = await tf.loadGraphModel("js_model/model.json");
}