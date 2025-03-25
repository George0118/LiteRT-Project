import tensorflow as tf
import numpy as np
import cv2
import os
import subprocess

image_folder = '../images'
converted_images_folder = '../converted_images'
input_shape = (32, 32, 3)

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[0], input_shape[1]))  # Resize to model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to create a representative dataset (returns batches of images)
def representative_dataset():
    # List all images in the folder
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
    
    for image_path in image_paths:
        image = load_and_preprocess_image(image_path)
        yield [image.astype(np.float32)]  
        
        
# Paths for input models and output converted models
model_paths = [f"./model_test/model_{i}" for i in range(10)]
converted_model_paths = [f"./model_result/model_{i}" for i in range(10)]
input_shape = (32,32,3)

for i, path in enumerate(model_paths):
    # Convert the TensorFlow model to a TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.target_spec.supported_types = [tf.int8]
    tflite_model = converter.convert()

    # Create directory if it doesn't exist
    os.makedirs(converted_model_paths[i], exist_ok=True)

    # Define model file names
    tflite_filename = f"{converted_model_paths[i]}/converted_model.tflite"

    # Save the TFLite model
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
        
    cc_filename = f"{converted_model_paths[i]}/model_{i}_data.cc"

    # Generate the .cc file using xxd
    with open(cc_filename, "w") as cc_file:
        subprocess.run(
            ["C:/Program Files/Git/usr/bin/xxd.exe", "-i", tflite_filename],
            stdout=cc_file,
            check=True
        )

image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])

for i,image_path in enumerate(image_paths):
    image = load_and_preprocess_image(image_path)
    image = (image[0]).astype(np.uint8)
    converted_image_path = converted_images_folder+f"/converted_image_{i}.jpg"
    cv2.imwrite(converted_image_path, image)
    
print("Conversion complete!")
