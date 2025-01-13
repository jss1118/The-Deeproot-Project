import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, classification_report
import datetime
import os

# ----------------------------------------------------------------------
# 1) Set the base directory that holds multiple folders (each folder = a dataset)
#    Example structure:
#    base_dir/
#       ├─ apple/
#       │    ├─ classA/
#       │    ├─ classB/
#       ├─ corn/
#       │    ├─ classA/
#       │    ├─ classB/
#       ...
# ----------------------------------------------------------------------
base_dir = "/media/josh/SanDisk SSD/Python Projects/new datasets/"  # Change to your own path

# ----------------------------------------------------------------------
# 2) Iterate through each folder in base_dir
# ----------------------------------------------------------------------
for plant_folder in os.listdir(base_dir):
    plant_path = os.path.join(base_dir, plant_folder)

    # Skip if it's not a directory
    if not os.path.isdir(plant_path):
        continue
    
    # Identify how many classes by checking subfolders
    subfolders = [
        sf for sf in os.listdir(plant_path) 
        if os.path.isdir(os.path.join(plant_path, sf))
    ]
    class_num = len(subfolders)

    # Determine the loss mode based on class count
    if class_num == 2:
        mode = 'binary'
    else:
        mode = 'categorical'

    print(f"\n--- Training on folder: {plant_folder} ---")
    print(f"Path: {plant_path}")
    print(f"Detected {class_num} classes. Using '{mode}_crossentropy'.")

    # ----------------------------------------------------------------------
    # 3) Create training and validation sets from the current folder
    # ----------------------------------------------------------------------
    try:
        training_set = tf.keras.utils.image_dataset_from_directory(
            plant_path,
            labels="inferred",
            label_mode="categorical",   # or "binary" if exactly 2 classes
            color_mode="rgb",
            batch_size=8,
            image_size=(128, 128),
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset='training'
        )

        validation_set = tf.keras.utils.image_dataset_from_directory(
            plant_path,
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb",
            batch_size=8,
            image_size=(128, 128),
            shuffle=True,
            seed=123,
            validation_split=0.2,
            subset='validation'
        )
    except ValueError as e:
        # If there's an error (e.g., not enough images in subfolders), skip this folder
        print(f"Skipping {plant_folder} due to error: {e}")
        continue

    # ----------------------------------------------------------------------
    # 4) Build Model (Updated CNN Architecture)
    #
    #    Incorporates:
    #      - Batch Normalization (after each Conv layer)
    #      - Dropout in each block to reduce overfitting
    #      - 'he_uniform' initializer for better convergence with ReLU
    # ----------------------------------------------------------------------
    cnn = tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_uniform',
                               input_shape=[128, 128, 3]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3,
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.25),

        # Block 2
        tf.keras.layers.Conv2D(64, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3,
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.25),

        # Block 3
        tf.keras.layers.Conv2D(128, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3,
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.25),

        # Block 4
        tf.keras.layers.Conv2D(256, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3,
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.25),

        # Block 5
        tf.keras.layers.Conv2D(512, 3, padding='same',
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(512, 3,
                               activation='relu',
                               kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.25),

        # Flatten + Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1500, activation='relu',
                              kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(class_num, activation='softmax')  # match class_num
    ])

    # ----------------------------------------------------------------------
    # 5) Compile the model
    # ----------------------------------------------------------------------
    cnn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=f'{mode}_crossentropy',
        metrics=['accuracy']
    )

    # Optional: You can experiment with a learning-rate scheduler if desired:
    # from tensorflow.keras.callbacks import ReduceLROnPlateau
    # lr_scheduler = ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.5, patience=3, verbose=1
    # )

    cnn.summary()

    # ----------------------------------------------------------------------
    # 6) Setup TensorBoard logging (Optional)
    # ----------------------------------------------------------------------
    log_dir = os.path.join(
        "logs",
        f"{plant_folder}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1  # set to 1 to record histograms
    )

    # ----------------------------------------------------------------------
    # 7) Early Stopping Callback
    #    This stops training if validation accuracy does not improve by 2% (0.02).
    # ----------------------------------------------------------------------
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.005,   # 2% improvement threshold
        patience=1,       # number of epochs to wait
        verbose=1,
        restore_best_weights=True
    )

    # ----------------------------------------------------------------------
    # 8) Train the model
    # ----------------------------------------------------------------------
    print(f"Starting training for {plant_folder} ...")
    training_history = cnn.fit(
        x=training_set,
        validation_data=validation_set,
        epochs=11,
        callbacks=[tensorboard_callback, early_stopping]
        # If using LR scheduler:
        # callbacks=[tensorboard_callback, early_stopping, lr_scheduler]
    )

    # ----------------------------------------------------------------------
    # 9) Evaluate the model
    # ----------------------------------------------------------------------
    train_loss, train_acc = cnn.evaluate(training_set)
    print('Training accuracy:', train_acc)
    val_loss, val_acc = cnn.evaluate(validation_set)
    print('Validation accuracy:', val_acc)

    # ----------------------------------------------------------------------
    # 10) (Optional) Save the model
    # ----------------------------------------------------------------------
    # model_save_path = f"my_model_{plant_folder}.h5"
    # cnn.save(model_save_path)

    # -------------------
