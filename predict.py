import numpy as np
import os
from keras.models import load_model
from keras.utils import load_img, img_to_array

# Disable annoying info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model
model_to_load = 'mobilenetv3_finetuned(3M)'
model = load_model(os.path.join('models', model_to_load + '.h5'))

# Load images and predict
image_folder = 'images_to_predict'
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]

results = []
for img in image_files:
    image_path = os.path.join(image_folder, img)
    image = load_img(image_path)
    image_array = img_to_array(image)

    # Predict
    y_pred = model.predict(np.expand_dims(image_array, axis=0))

    # Interpret
    if y_pred > 0.5:
        results.append(f'{img} {y_pred} cunny 😭')
    else:
        results.append(f'{img} {y_pred} not cunny 😇')

# Print the results 
print('')      
for res in results:
    print(res)