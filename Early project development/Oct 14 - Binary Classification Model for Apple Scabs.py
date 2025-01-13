#IF YOU WANT THIS CODE TO RUN, REMOVE CLASS_C FOLDER FROM DIRECTORY
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Path to training and test datasets
training_data = '/Users/joshua.stanley/Desktop/Final Train'
test_data = '/Users/joshua.stanley/Desktop/test'

# Identifying and loading the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    training_data,
    image_size=(256, 256),
    labels='inferred',
    class_names=['class_a', 'class_b'],
    batch_size=32,
    label_mode='binary',  
    seed=123,
    validation_split=0.2,
    subset='training'
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_data,
    image_size=(256, 256),
    labels='inferred',
    class_names=['class_a', 'class_b'],
    batch_size=32,
    label_mode='binary', 
    seed=123
)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)  # Binary output
])

# Model compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model training
model.fit(train_dataset, epochs=3)

# Model evaluation
model.evaluate(test_dataset)

# Model prediction on test dataset
classifications = model.predict(test_dataset)

# Class names mapping
class_names = ['Scabbed', 'Healthy']

# Plotting first 10 images and their predicted labels
plt.figure(figsize=(10, 10))
for images, labels in test_dataset.take(1):  # Take one batch from the dataset
    predictions = model.predict(images)      # Predict the batch
    for i in range(10):  # Display first 10 images
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  # Display the image
        predicted_class = int(np.round(predictions[i]))  # Fix: predictions[i] is already a float
        true_label = int(labels[i].numpy())  # True label
        plt.title(f'Pred: {class_names[predicted_class]}, True: {class_names[true_label]}', 
                  fontsize=7)  # Adjust fontsize here
        plt.axis('off')
plt.tight_layout()  # Automatically adjust subplot params to give specified padding
plt.show()
