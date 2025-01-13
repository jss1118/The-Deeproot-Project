import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import customtkinter as ctk
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from colorama import *

print("Num GPUs available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Path to training and test datasets
model_save_path = '/Users/joshua.stanley/Desktop'

# Leaf disease detection dataset (Segment 2) ----------------------------------------------------------------------------------------------------------------------------------

try:
    training_data = '/Users/joshua.stanley/Desktop/Science Research/Datasets/Original Dataset'
    test_data = '/Users/joshua.stanley/Desktop/Science Research/Datasets/Original Dataset/test/test'
except:
    print(Fore.RED + 'Could not locate files for training model.')

# Identifying and loading the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    training_data,
    image_size=(256, 256),
    labels='inferred',
    batch_size=32,  # Increase batch size for better gradient estimation
    label_mode='categorical',
    seed=123,
    validation_split=0.2,
    subset='training'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    training_data,
    image_size=(256, 256),
    labels='inferred',
    batch_size=32,
    label_mode='categorical',
    seed=123,
    validation_split=0.2,
    subset='validation'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_data,
    image_size=(256, 256),
    labels='inferred',
    batch_size=32,
    label_mode='categorical',
    seed=123
)

# Prefetch datasets for improved training performance


# Model definition for multi-class classification
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(256, 256, 3)),

    # Enhanced Data Augmentation Layer
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.3),  # Increased rotation range
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomContrast(0.2),  # Added random contrast augmentation
    tf.keras.layers.RandomBrightness(0.3),  # Increased brightness range
    tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),  # Random translations

    # Convolutional Layers
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # Increased filter size
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),  # Added deeper layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Fully Connected Layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization
    tf.keras.layers.Dense(256, activation='relu'),  # Added a dense layer
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(38, activation='softmax')  # Multi-class output layer
])

# Model compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate for stability
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model training
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,  # Increased epochs for better training
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),  # Early stopping
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)  # Learning rate reduction on plateau
    ]
)

# Model evaluation
model.evaluate(test_dataset)

# Save the model
model.save(model_save_path)
