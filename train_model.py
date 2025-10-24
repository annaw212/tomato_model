import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and preprocess data (assuming images in 'data/train' and 'data/val')
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE=64

train_ds = tf.keras.utils.image_dataset_from_directory(
  'data/train',
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  'data/val',
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE
) 

class_names = train_ds.class_names
num_classes = len(class_names)

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 2. Define the model
model = models.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

# 3. Compile the Model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# 4. Train the Model
epochs = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 5. Evaluate the model (on a separate test if available)
# test_loss, test_acc = model.evaluate(test_ds)
# print(test_loss, test_acc)

# 6. Save the model
model.save("tomato_model.keras")
