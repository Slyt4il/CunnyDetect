import numpy as np
import os
from keras.models import load_model
from keras.utils import load_img, img_to_array

# Disable annoying info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model
model_to_load = 'CunnyDetectV1(2.2M)' # Default is CunnyDetectV1(2.2M)
model = load_model(os.path.join('models', model_to_load + '.h5'))

# Preprocess images
image_folder = 'images_to_predict'
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
image_size = (256, 256) # Adjust this to match the model's input size. Default is (256, 256)

# Load images for prediction
results = []
for img in image_files:
    image_path = os.path.join(image_folder, img)
    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image)

    # Predict
    y_pred = model.predict(np.expand_dims(image_array / 255, axis=0))

    # Interpret
    if y_pred < 0.475:
        results.append(f'({img} {y_pred}) Not cunny 😇')
    elif y_pred < 0.50:
        results.append(f'({img} {y_pred}) Not sure (not cunny?) 🤔')
    elif y_pred < 0.525:
        results.append(f'({img} {y_pred}) Not sure (cunny?) 🤔')
    else:
        results.append(f'({img} {y_pred}) Uohhhhhhhhh! 😭')

# Print the results 
print('')      
for res in results:
    print(res)