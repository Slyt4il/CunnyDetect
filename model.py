import tensorflow as tf
import os
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy

# Disable annoying info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Prevent Tensorflow from using up all memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load data and create dataset
data = tf.keras.utils.image_dataset_from_directory('training_data', batch_size=8, image_size=(256, 256))

# Scale data
data = data.map(lambda x, y: (x / 255, y))

# Split data
data_size = int(len(data))
train_size, val_size = int(data_size * 0.7), int(data_size * 0.2)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(data_size - (train_size + val_size))

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(256, 256, 3), padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), 1, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), 1, activation='relu', padding='same'))
model.add(AveragePooling2D())
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Early Stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, mode='auto', restore_best_weights=True)

# Train model
hist = model.fit(train, epochs=50, validation_data=val, callbacks=callback)

# Visualize losses (maybe later)

# Evaluate model
pre, re, acc = Precision(), Recall(), BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    y_pred = model.predict(X)
    pre.update_state(y, y_pred)
    re.update_state(y, y_pred)
    acc.update_state(y, y_pred)
print('=' * os.get_terminal_size().columns)
print(model.summary())
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
print('=' * os.get_terminal_size().columns)

# Save model
model.save(os.path.join('models', 'new_model.h5'))