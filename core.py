import coremltools as ct
import tensorflow as tf
crop = "apple"
# Load your Keras model (this could be a .keras or .h5 file)
model = tf.keras.models.load_model(f"/Users/joshua.stanley/Desktop/Science Research/Saved Models/category/model{crop}.keras")

# Get the expected input shape (e.g., (None, 224, 224, 3))
input_shape = model.input.shape  # Typically (None, height, width, channels)

# Define an image input type.
# Adjust 'name' if needed (it should match the model's input name).
image_input = ct.ImageType(
    name=model.input_names[0],        # The input layer name (usually "input_1" or similar)
    shape=input_shape,                # E.g., (None, 224, 224, 3)
    scale=1/255.0,                    # If your model expects normalized pixel values
    bias=[0, 0, 0]                    # Adjust bias if necessary
)

# Convert the Keras model to Core ML format, specifying the image input.
coreml_model = ct.convert(model, inputs=[image_input])

# Save the converted model.
coreml_model.save(f"model{crop}.mlmodel")
