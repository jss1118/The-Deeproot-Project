import os
import tensorflow as tf
import coremltools as ct

# Define a custom InputLayer that remaps 'batch_shape' to 'batch_input_shape'
def CustomInputLayer(**kwargs):
    if 'batch_shape' in kwargs:
        kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
    return tf.keras.layers.InputLayer(**kwargs)

# Register the missing DTypePolicy
custom_objects = {
    'InputLayer': CustomInputLayer,
    'DTypePolicy': tf.keras.mixed_precision.Policy,
}

# Directory containing the .h5 models
models_dir = "/Users/joshua.stanley/Desktop/Science Research/Saved Models/category"

# Iterate over files in the directory
for filename in os.listdir(models_dir):
    if filename.endswith(".h5"):
        model_path = os.path.join(models_dir, filename)
        print(f"Processing model: {model_path}")

        # Load the model without the optimizer state
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        print("Model loaded successfully (without optimizer)!")

        # Convert the model to Core ML format
        mlmodel = ct.convert(model)

        # Create a new filename for the converted model (e.g., modelapple.mlpackage)
        mlmodel_filename = os.path.splitext(filename)[0] + ".mlpackage"
        mlmodel_save_path = os.path.join(models_dir, mlmodel_filename)

        # Save the converted Core ML model
        mlmodel.save(mlmodel_save_path)
        print(f"Model saved as {mlmodel_save_path}\n")
