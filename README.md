# CunnyDetect

CunnyDetect is a state-of-the-art image classifier running entirely in the browser that aims to solve the age-old conundrum: *"Is this thing a loli?"*

- Can detect the presence of loli imagery with sufficient accuracy if the weather is good. (Lower accuracy if it's raining outside or if you look at it funny.)
- Lightweight. Can run in the browser.
- Help convince your peers that your waifu is, in fact, not a loli.

<br>
<p align="center">
    <img src="images_to_predict\aris.png" alt="example of a cunny image." width="162" height="226"></img>
</p>
<p align="center"> prediction: cunny 😭 </p>
<p align="center">
    <img src="images_to_predict\kafka.jpg" alt="example of a non-cunny image." width="162" height="226"></img>
</p>
<p align="center"> prediction: not_cunny 😇 </p>
<br>

## Model

| Model | Number of images | Vaidation accuracy |
| :---: | :--------------: | :----------------: |
| mobilenetv3_finetuned(3M) | 26446 | 86.7 |

## Requirements

`TensorFlow 2.10.0` is required if you want GPU support on Windows. Also cudatoolkit, cuda-nvcc, and cudnn.

If not training, CPU should be enough for prediction.


## Usage

**Training**
Place images in `training_data`
```
python model.py
```
**Predicting**
Place images in `images_to_predict`
```
python predict.py
```
**Converting to tensorflowjs model**
```
tensorflowjs_converter --input_format=keras --output_format=tfjs_graph_model models/mobilenetv3_finetuned(3M).h5 client/js_model
```

## Training data

Images are sourced from gelbooru, of which many are too cursed to redistribute.

| Class | Example tags |
| :---- | :----------- |
| cunny | `1girl` `solo` `loli` `flat-chest` `female_focus` `highres`|
| not_cunny | `1girl` `1boy` `solo` `breasts` `mature_female` `female_focus` `male_focus` `highres` |