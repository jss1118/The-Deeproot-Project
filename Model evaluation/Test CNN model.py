import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt  # <-- For plotting

accuracy_full = 0
number_of_tests = 0

plant_names = []      # Keep track of the plant (model) names
plant_accuracies = [] # Keep track of each plant's accuracy

print("Num GPUs available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Root directory containing all plant subfolders
root_dir = '/Volumes/SanDisk SSD/saved from project/archive'
# Directory containing all of the saved models (one per plant)
saved_model_dir = '/Users/joshua.stanley/Desktop/Science Research/Saved Models/category'

# Iterate over all folders (plants) in the root directory
for plant in sorted(os.listdir(root_dir)):
    plant_path = os.path.join(root_dir, plant)
    
    # Only proceed if this is actually a folder
    if not os.path.isdir(plant_path):
        continue

    print("────────────────────────────────────────────")
    print(f"Evaluating plant folder: {plant}")

    # Build the path to the model for this plant
    model_path = os.path.join(saved_model_dir, f"model{plant}.keras")
    if not os.path.exists(model_path):
        print(f"Model for '{plant}' not found at:\n  {model_path}\nSkipping...")
        continue

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Get test class names from subfolders in this plant's directory
    test_class_names = sorted(next(os.walk(plant_path))[1])  # [1] => subfolders
    print(f"Test classes ({len(test_class_names)}): {test_class_names}")

    # Create a test dataset
    test_set = tf.keras.utils.image_dataset_from_directory(
        plant_path,
        labels="inferred",
        label_mode="categorical",
        batch_size=1,
        image_size=(128, 128),
        shuffle=False
    )
    
    # We will test on exactly 16 images
    num_images_to_test = 16
    images_to_test = []
    for images, labels in test_set:
        for img, label in zip(images.numpy(), labels.numpy()):
            images_to_test.append((img, label))
            if len(images_to_test) == num_images_to_test:
                break
        if len(images_to_test) == num_images_to_test:
            break

    accuracy = 0

    # Evaluate each of the 16 images
    for i, (image, label) in enumerate(images_to_test):
        image_batch = np.expand_dims(image, axis=0)
        prediction = model.predict(image_batch)
        predicted_class_index = np.argmax(prediction)

        true_class_name = test_class_names[np.argmax(label)]
        predicted_class_name = test_class_names[predicted_class_index]

        print(f"  True: {true_class_name} | Pred: {predicted_class_name}")

        if true_class_name == predicted_class_name:
            accuracy += 100
            
    number_of_tests += 1
    
    # Calculate the final accuracy for this plant (out of 16 images)
    accuracy = accuracy / num_images_to_test
    accuracy_full += accuracy
    
    # Print the plant's accuracy
    print(f"Accuracy for {plant} = {accuracy}%\n")
    
    # Store the plant name and accuracy for the bar chart
    plant_names.append(plant)
    plant_accuracies.append(accuracy)

# Print overall average accuracy (assuming you have 19 plants total)
print(f'Full accuracy = {accuracy_full / 19}')

# ──────────────────────────────────────────────────────────────────
# Generate a bar chart for each plant's accuracy
plt.figure(figsize=(10, 6))
plt.bar(plant_names, plant_accuracies, color='skyblue')
plt.xlabel('Plant')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Model (16 images tested per plant)')
plt.xticks(rotation=90)
plt.tight_layout()  # Helps avoid label cutoff
plt.show()
