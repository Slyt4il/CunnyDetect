import tensorflow as tf
import os

# Disable annoying info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Prevent Tensorflow from using up all memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load data and create dataset
train_data = tf.keras.utils.image_dataset_from_directory('training_data', validation_split=0.2, subset='training', seed=696969, batch_size=32)
val_data = tf.keras.utils.image_dataset_from_directory('training_data', validation_split=0.2, subset='validation', seed=696969, batch_size=32)

# Prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

# Preprocessing layers
inputs = tf.keras.Input(shape=(None, None, 3))
x = tf.keras.layers.Resizing(224, 224)(inputs)

# Build model
base_model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, minimalistic=False, input_shape=(224, 224, 3))
base_model.trainable = False
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# Train model
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, mode='auto', restore_best_weights=True)
model.fit(train_data, epochs=90, validation_data=val_data, callbacks=callback)

model.save(os.path.join('models', 'new_model.h5'))

# Fine-tuning
base_model.trainable = True
for layer in base_model.layers[:120]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
callbackft = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto', restore_best_weights=True)
model.fit(train_data, epochs=15, validation_data=val_data, callbacks=callbackft)

model.save(os.path.join('models', 'new_model_ft.h5'))

# Model summary
print('=' * os.get_terminal_size().columns)
print(model.summary())
print('=' * os.get_terminal_size().columns)