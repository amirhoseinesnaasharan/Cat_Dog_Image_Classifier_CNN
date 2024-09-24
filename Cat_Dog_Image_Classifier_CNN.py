import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# ------------------------------------------------------------------------

# Define dataset
train_dir = "E:\\python\\code python\\Cats and Dogs image classification\\train"
test_dir = "E:\\python\\code python\\Cats and Dogs image classification\\test"

# Set random seed for reproducibility
seed_value = 42
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training and test data
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # set the image size as needed
    batch_size=32,
    class_mode='binary'  # Binary classification (dog or cat)
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output for yes/no
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,  # Set the number of epochs as needed
    validation_data=test_generator,
    validation_steps=len(test_generator),
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

# Save trained weights
model.save_weights('dog_cat_classification_weights.h5')

# Predict on a single image (replace with your image path)
img_path = "E:\\python\\code python\\Cats and Dogs image classification\\test\\cats\\cat_124.jpg"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

# Use predict for binary classification
prediction_probs = model.predict(img_array)[0]

# Round the probability to the nearest integer (0 or 1)
prediction = round(prediction_probs[0])

if prediction == 1:
    print("Predicted: Dog")
else:
    print("Predicted: Cat")

"""
Number of train parameters = (filter width*filter height*input channels+1)*number of filters

Number of train parameters for each layer in your CNN:
Conv2D layer :
Filter size: 3*3
Input channels:3 (RGB images)
Number of filters: 32
(3*3*3+1)*32= 896

MaxPooling2D layer (first and second layer)= no parametre

Conv2D layer (2nd layer):
Filter size: 3*3
Input channels:32
Number of filters:64
(3*3*32+1)*64= 18496

first fully connected layer:
Input size: 64
Number of neurons: 64
(64*64+1)*64= 4160

second fully connected layer:
Input size: 64
Number of neurons: 1
(64+1)*1=65

896 + 18496 + 4160 + 65 = 23117
"""
